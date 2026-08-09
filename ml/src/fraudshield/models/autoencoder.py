"""Autoencoder-based anomaly detection.

A deliberately different modelling philosophy from the supervised models: train
a compression bottleneck on LEGITIMATE transactions only, then score a new
transaction by how badly it reconstructs. High reconstruction error means the
transaction does not look like normal traffic.

Why keep it alongside XGBoost? The supervised models can only recognise fraud
patterns resembling the 492 labelled positives they were trained on. The
autoencoder never sees a fraud label, so it can flag genuinely novel attack
patterns - the case where a supervised model silently fails. In production the
two run side by side: supervised for known fraud, autoencoder as a drift and
novelty tripwire.

The anomaly threshold is derived from a VALIDATION percentile. Test data is
never consulted when choosing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _build_encoder_decoder(
    input_dim: int, encoder_dims: list[int], latent_dim: int, dropout: float
) -> tuple[nn.Sequential, nn.Sequential]:
    encoder_layers: list[nn.Module] = []
    previous = input_dim
    for width in encoder_dims:
        encoder_layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
        previous = width
    encoder_layers.append(nn.Linear(previous, latent_dim))

    decoder_layers: list[nn.Module] = []
    previous = latent_dim
    for width in reversed(encoder_dims):
        decoder_layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
        previous = width
    decoder_layers.append(nn.Linear(previous, input_dim))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


class AutoencoderAnomalyDetector:
    """Reconstruction-error anomaly detector with an sklearn-style surface."""

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int] | None = None,
        latent_dim: int = 16,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 2048,
        max_epochs: int = 60,
        early_stopping_patience: int = 8,
        seed: int = 42,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims or [64, 32]
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.encoder, self.decoder = _build_encoder_decoder(
            input_dim, self.encoder_dims, latent_dim, dropout
        )
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.error_threshold_: float | None = None
        self.error_scale_: float = 1.0

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X_train_legit: np.ndarray,
        X_val_legit: np.ndarray,
    ) -> AutoencoderAnomalyDetector:
        """Train on legitimate rows only; early stop on validation reconstruction loss."""
        criterion = nn.MSELoss()
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        def loader(X: np.ndarray, shuffle: bool) -> DataLoader:
            dataset = TensorDataset(torch.tensor(np.asarray(X, dtype=np.float32)))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        train_loader, val_loader = loader(X_train_legit, True), loader(X_val_legit, False)
        best_loss, best_state, stale = float("inf"), None, 0

        for epoch in range(1, self.max_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            running = 0.0
            for (xb,) in train_loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._forward(xb), xb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
            train_loss = running / max(1, len(train_loader.dataset))

            self.encoder.eval()
            self.decoder.eval()
            val_running = 0.0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(self.device)
                    val_running += criterion(self._forward(xb), xb).item() * xb.size(0)
            val_loss = val_running / max(1, len(val_loader.dataset))

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < best_loss - 1e-7:
                best_loss, stale = val_loss, 0
                best_state = (
                    {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                    {k: v.detach().clone() for k, v in self.decoder.state_dict().items()},
                )
            else:
                stale += 1
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "AE epoch %02d | train MSE %.6f | val MSE %.6f", epoch, train_loss, val_loss
                )
            if stale >= self.early_stopping_patience:
                logger.info("Autoencoder early stopping at epoch %d.", epoch)
                break

        if best_state is not None:
            self.encoder.load_state_dict(best_state[0])
            self.decoder.load_state_dict(best_state[1])
        if self.checkpoint_path:
            self.save(self.checkpoint_path)
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-row mean squared reconstruction error."""
        self.encoder.eval()
        self.decoder.eval()
        tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            reconstructed = self._forward(tensor)
            errors = torch.mean((tensor - reconstructed) ** 2, dim=1)
        return errors.cpu().numpy().ravel()

    def calibrate(self, X_val: np.ndarray, percentile: float = 99.0) -> float:
        """Set the anomaly threshold and score scale from VALIDATION data only.

        ``percentile`` is taken over validation reconstruction errors, so the
        alert rate is pinned to roughly ``100 - percentile`` percent of traffic.
        """
        errors = self.reconstruction_error(X_val)
        self.error_threshold_ = float(np.percentile(errors, percentile))
        self.error_scale_ = float(max(np.percentile(errors, 99.9), 1e-9))
        logger.info(
            "Autoencoder calibrated: threshold=%.6f (p%.1f of validation errors)",
            self.error_threshold_,
            percentile,
        )
        return self.error_threshold_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Map reconstruction error into a bounded pseudo-probability.

        This is a monotone rank score, NOT a calibrated probability - it is only
        comparable to itself. PR-AUC is rank-based, so model comparison remains
        valid, but these values must not be read as fraud likelihoods.
        """
        errors = self.reconstruction_error(X)
        scores = np.clip(errors / self.error_scale_, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "input_dim": self.input_dim,
                "encoder_dims": self.encoder_dims,
                "latent_dim": self.latent_dim,
                "dropout": self.dropout,
                "error_threshold": self.error_threshold_,
                "error_scale": self.error_scale_,
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> AutoencoderAnomalyDetector:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        instance = cls(
            input_dim=blob["input_dim"],
            encoder_dims=blob["encoder_dims"],
            latent_dim=blob["latent_dim"],
            dropout=blob["dropout"],
            device=device,
        )
        instance.encoder.load_state_dict(blob["encoder"])
        instance.decoder.load_state_dict(blob["decoder"])
        instance.error_threshold_ = blob.get("error_threshold")
        instance.error_scale_ = blob.get("error_scale", 1.0)
        instance.history = blob.get("history", instance.history)
        return instance
