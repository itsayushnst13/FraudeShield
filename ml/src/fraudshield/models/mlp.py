"""PyTorch MLP for tabular fraud classification.

Architecture: Linear -> BatchNorm -> ReLU -> Dropout blocks into a single logit.
Trained with ``BCEWithLogitsLoss(pos_weight=...)`` so the rare positive class is
upweighted in the loss instead of being resampled - no synthetic rows, no
leakage risk. Early stopping monitors validation PR-AUC (not loss), because loss
is dominated by the majority class and improves while fraud recall degrades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    """Stack Linear/BatchNorm/ReLU/Dropout blocks ending in a single logit."""
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers += [
            nn.Linear(previous, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


@dataclass
class TrainingHistory:
    """Per-epoch curves used for the training plots and MLflow logging."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_pr_auc": self.val_pr_auc,
        }


class TorchMLPClassifier:
    """sklearn-style wrapper around the MLP so it drops into the shared pipeline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 2048,
        max_epochs: int = 60,
        early_stopping_patience: int = 8,
        grad_clip: float = 5.0,
        pos_weight: float | None = None,
        seed: int = 42,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip
        self.pos_weight = pos_weight
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = build_mlp(input_dim, self.hidden_dims, dropout).to(self.device)
        self.history = TrainingHistory()
        self.best_epoch_ = -1
        self.best_val_pr_auc_ = -1.0

    def _loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(np.asarray(X, dtype=np.float32)),
            torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1),
        )
        # drop_last avoids a BatchNorm crash on a trailing batch of size 1.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) > self.batch_size,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TorchMLPClassifier:
        """Train with early stopping on validation PR-AUC, restoring the best epoch."""
        pos_weight = self.pos_weight
        if pos_weight is None:
            positives = max(1.0, float(np.sum(y_train == 1)))
            pos_weight = float(np.sum(y_train == 0)) / positives
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                running += loss.item() * xb.size(0)
            train_loss = running / max(1, len(train_loader.dataset))

            val_loss, val_probs = self._evaluate(val_loader, criterion)
            y_val_arr = np.asarray(y_val).astype(int).ravel()
            val_pr_auc = (
                float(average_precision_score(y_val_arr, val_probs))
                if 0 < y_val_arr.sum() < len(y_val_arr)
                else 0.0
            )

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.val_pr_auc.append(val_pr_auc)

            if val_pr_auc > self.best_val_pr_auc_ + 1e-6:
                self.best_val_pr_auc_ = val_pr_auc
                self.best_epoch_ = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
                if self.checkpoint_path:
                    self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, self.checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "MLP epoch %02d | train_loss %.5f | val_loss %.5f | val PR-AUC %.4f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_pr_auc,
                )
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info("Early stopping at epoch %d (best epoch %d).", epoch, self.best_epoch_)
                break

        self.model.load_state_dict(best_state)
        return self

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray]:
        self.model.eval()
        total, probs = 0.0, []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                total += criterion(logits, yb).item() * xb.size(0)
                probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        n = max(1, len(loader.dataset))
        return total / n, np.concatenate(probs) if probs else np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return an (n, 2) probability matrix to match the sklearn interface."""
        self.model.eval()
        tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            positive = torch.sigmoid(self.model(tensor)).cpu().numpy().ravel()
        return np.column_stack([1.0 - positive, positive])

    def save(self, path: str | Path) -> Path:
        """Persist weights plus the hyper-parameters needed to rebuild the module."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "history": self.history.to_dict(),
                "best_epoch": self.best_epoch_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> TorchMLPClassifier:
        """Rebuild a classifier from a checkpoint written by :meth:`save`."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        instance = cls(
            input_dim=blob["input_dim"],
            hidden_dims=blob["hidden_dims"],
            dropout=blob["dropout"],
            device=device,
        )
        instance.model.load_state_dict(blob["state_dict"])
        instance.model.eval()
        return instance
