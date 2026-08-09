"""Individual model trainers.

Each trainer takes preprocessed arrays and returns ``(fitted_model, val_probs)``.
Keeping them uniform is what lets the pipeline treat logistic regression, XGBoost
and a PyTorch network as interchangeable candidates in the same comparison table.

Invariant enforced everywhere in this module: any imbalance handling touches
``X_train``/``y_train`` only. ``X_val`` is never resampled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..imbalance import ResampleResult, apply_strategy
from ..logging_utils import get_logger
from ..models.autoencoder import AutoencoderAnomalyDetector
from ..models.baselines import build_dummy, build_logistic_regression
from ..models.mlp import TorchMLPClassifier
from ..models.trees import build_lightgbm, build_random_forest, build_xgboost

logger = get_logger(__name__)


@dataclass
class TrainedCandidate:
    """A trained model plus everything needed to evaluate and register it."""

    name: str
    model_type: str
    model: Any
    val_probs: np.ndarray
    imbalance_strategy: str
    params: dict[str, Any]
    history: dict[str, list[float]] | None = None
    tuning: Any = None


def _positive_probs(model: Any, X: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X))
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()


def train_sklearn_candidate(
    name: str,
    model_type: str,
    builder: Callable[[ResampleResult], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    strategy: str,
    cfg: Config,
    use_sample_weight: bool = False,
    tune: bool = False,
) -> TrainedCandidate:
    """Fit an sklearn-compatible estimator under a given imbalance strategy.

    When ``tune`` is set, a randomised search runs on the resampled TRAINING
    fold before the final fit. The validation split stays untouched.
    """
    resampled = apply_strategy(X_train, y_train, strategy, cfg)
    model = builder(resampled)
    tuning = None
    if tune:
        from .hpo import tune_estimator

        model, tuning = tune_estimator(name, model, resampled.X, resampled.y, cfg)
    if use_sample_weight and resampled.sample_weight is not None:
        model.fit(resampled.X, resampled.y, sample_weight=resampled.sample_weight)
    else:
        model.fit(resampled.X, resampled.y)
    logger.info("Trained %s [%s] on %d rows", name, strategy, len(resampled.y))
    return TrainedCandidate(
        name=name,
        model_type=model_type,
        model=model,
        val_probs=_positive_probs(model, X_val),
        imbalance_strategy=strategy,
        params=getattr(model, "get_params", dict)() if hasattr(model, "get_params") else {},
        tuning=tuning,
    )


def train_baselines(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, cfg: Config
) -> list[TrainedCandidate]:
    """Dummy and logistic-regression controls."""
    return [
        train_sklearn_candidate(
            "dummy", "dummy", lambda _r: build_dummy(cfg), X_train, y_train, X_val, "none", cfg
        ),
        train_sklearn_candidate(
            "logistic_regression",
            "logistic_regression",
            lambda _r: build_logistic_regression(cfg, balanced=True),
            X_train,
            y_train,
            X_val,
            "class_weight",
            cfg,
        ),
    ]


def train_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    cfg: Config,
    strategy: str = "class_weight",
    tune: bool = False,
) -> list[TrainedCandidate]:
    """Random forest, XGBoost and (when installed) LightGBM."""
    candidates = [
        train_sklearn_candidate(
            "random_forest",
            "random_forest",
            lambda _r: build_random_forest(cfg, balanced=True),
            X_train,
            y_train,
            X_val,
            strategy,
            cfg,
            tune=tune,
        ),
        train_sklearn_candidate(
            "xgboost",
            "xgboost",
            lambda r: build_xgboost(cfg, scale_pos_weight=r.scale_pos_weight),
            X_train,
            y_train,
            X_val,
            strategy,
            cfg,
            tune=tune,
        ),
    ]
    if build_lightgbm(cfg, scale_pos_weight=1.0) is not None:
        candidates.append(
            train_sklearn_candidate(
                "lightgbm",
                "lightgbm",
                lambda r: build_lightgbm(cfg, scale_pos_weight=r.scale_pos_weight),
                X_train,
                y_train,
                X_val,
                strategy,
                cfg,
                tune=tune,
            )
        )
    return candidates


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    checkpoint_dir: Path | None = None,
) -> TrainedCandidate:
    """Train the PyTorch MLP with weighted BCE and PR-AUC early stopping."""
    params = dict(cfg.get("models.mlp", {}) or {})
    checkpoint = (checkpoint_dir / "mlp_best.pt") if checkpoint_dir else None
    model = TorchMLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=list(params.get("hidden_dims", [128, 64])),
        dropout=float(params.get("dropout", 0.3)),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-5)),
        batch_size=int(params.get("batch_size", 2048)),
        max_epochs=int(params.get("max_epochs", 60)),
        early_stopping_patience=int(params.get("early_stopping_patience", 8)),
        grad_clip=float(params.get("grad_clip", 5.0)),
        seed=cfg.seed,
        checkpoint_path=checkpoint,
    )
    model.fit(X_train, np.asarray(y_train), X_val, np.asarray(y_val))
    return TrainedCandidate(
        name="torch_mlp",
        model_type="torch_mlp",
        model=model,
        val_probs=_positive_probs(model, X_val),
        imbalance_strategy="class_weight(pos_weight)",
        params=params,
        history=model.history.to_dict(),
    )


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    cfg: Config,
    checkpoint_dir: Path | None = None,
) -> TrainedCandidate:
    """Train the autoencoder on legitimate training rows only, then calibrate on val."""
    params = dict(cfg.get("models.autoencoder", {}) or {})
    y_train = np.asarray(y_train).astype(int)
    legit_train = X_train[y_train == 0]

    model = AutoencoderAnomalyDetector(
        input_dim=X_train.shape[1],
        encoder_dims=list(params.get("encoder_dims", [64, 32])),
        latent_dim=int(params.get("latent_dim", 16)),
        dropout=float(params.get("dropout", 0.1)),
        lr=float(params.get("lr", 1e-3)),
        batch_size=int(params.get("batch_size", 2048)),
        max_epochs=int(params.get("max_epochs", 60)),
        early_stopping_patience=int(params.get("early_stopping_patience", 8)),
        seed=cfg.seed,
        checkpoint_path=(checkpoint_dir / "autoencoder_best.pt") if checkpoint_dir else None,
    )
    # Hold out a slice of legitimate training rows for the AE's own early stopping,
    # so the labelled validation split is not consumed by unsupervised training.
    split_at = max(1, int(0.9 * len(legit_train)))
    model.fit(legit_train[:split_at], legit_train[split_at:])
    model.calibrate(X_val, percentile=99.0)

    return TrainedCandidate(
        name="autoencoder",
        model_type="autoencoder",
        model=model,
        val_probs=_positive_probs(model, X_val),
        imbalance_strategy="trained_on_legitimate_only",
        params=params,
        history=model.history,
    )
