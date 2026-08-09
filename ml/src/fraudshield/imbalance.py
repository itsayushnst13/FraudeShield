"""Class-imbalance handling.

The Kaggle dataset is roughly 0.172% fraud (492 positives in 284,807 rows). Three
families of remedy are supported, all of which are applied to the TRAINING FOLD
ONLY:

``none``          - train on the raw distribution (useful as a control).
``class_weight``  - reweight the loss so positives count more. No data is
                    synthesised, so there is no risk of leaking synthetic
                    neighbours across folds. Usually the safest default.
``undersample``   - discard majority rows. Cheap and fast, but throws away real
                    signal and inflates variance.
``smote``         - synthesise minority neighbours in feature space.

Resampling validation or test data would be a correctness bug, not a tuning
choice: it changes the base rate and makes precision meaningless. The functions
below therefore only ever receive training arrays, and
:func:`assert_no_resampling_on_eval` is available as an explicit guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

VALID_STRATEGIES = ("none", "class_weight", "undersample", "smote")


@dataclass
class ResampleResult:
    """Training arrays after a resampling strategy, plus optional loss weights."""

    X: np.ndarray
    y: np.ndarray
    strategy: str
    sample_weight: np.ndarray | None = None
    scale_pos_weight: float | None = None

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())

    @property
    def n_negative(self) -> int:
        return int(len(self.y) - self.y.sum())


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, used by XGBoost/LightGBM."""
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0:
        raise ValueError("Cannot compute scale_pos_weight with zero positive samples.")
    return negatives / positives


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights (sklearn's 'balanced' heuristic)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = {
        c: len(y) / (len(classes) * n) for c, n in zip(classes, counts, strict=True)
    }
    return np.array([weight_per_class[label] for label in y], dtype=float)


def apply_strategy(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "class_weight",
    cfg: Config | None = None,
) -> ResampleResult:
    """Apply an imbalance strategy to TRAINING data only.

    Returns resampled arrays and, where the strategy is weight-based, the weights
    the estimator should consume.
    """
    cfg = cfg or load_config()
    strategy = (strategy or "none").lower()
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'. Valid: {VALID_STRATEGIES}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        raise ValueError("Training data contains no positive samples.")

    if strategy == "none":
        return ResampleResult(X, y, strategy)

    if strategy == "class_weight":
        return ResampleResult(
            X,
            y,
            strategy,
            sample_weight=compute_sample_weights(y),
            scale_pos_weight=compute_scale_pos_weight(y),
        )

    seed = cfg.seed
    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        ratio = float(cfg.get("imbalance.undersample_ratio", 0.1))
        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    else:  # smote
        from imblearn.over_sampling import SMOTE

        k = int(cfg.get("imbalance.smote_k_neighbors", 5))
        k = max(1, min(k, int(y.sum()) - 1)) if y.sum() > 1 else 1
        sampler = SMOTE(random_state=seed, k_neighbors=k)

    X_res, y_res = sampler.fit_resample(X, y)
    logger.info(
        "Imbalance '%s': %d -> %d rows | positives %d -> %d",
        strategy,
        len(y),
        len(y_res),
        int(y.sum()),
        int(y_res.sum()),
    )
    return ResampleResult(np.asarray(X_res), np.asarray(y_res).astype(int), strategy)


def assert_no_resampling_on_eval(y_original: np.ndarray, y_used: np.ndarray, split: str) -> None:
    """Guard that an evaluation split was not resampled.

    Raises if the class distribution of ``y_used`` differs from ``y_original``.
    """
    orig_rate = float(np.mean(y_original))
    used_rate = float(np.mean(y_used))
    if len(y_original) != len(y_used) or not np.isclose(orig_rate, used_rate, atol=1e-12):
        raise AssertionError(
            f"Leakage guard tripped: '{split}' split appears resampled "
            f"(rows {len(y_original)}->{len(y_used)}, rate {orig_rate:.6f}->{used_rate:.6f})."
        )
