"""Baseline estimators.

Every serious model must beat these. The dummy classifier exists specifically to
demonstrate the accuracy trap: it achieves ~99.83% accuracy on the real dataset
while its PR-AUC collapses to the base rate.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from ..config import Config, load_config


def build_dummy(cfg: Config | None = None) -> DummyClassifier:
    """Always-majority classifier used as the floor for every metric."""
    cfg = cfg or load_config()
    return DummyClassifier(strategy="most_frequent", random_state=cfg.seed)


def build_logistic_regression(
    cfg: Config | None = None, balanced: bool = True
) -> LogisticRegression:
    """Regularised linear baseline; interpretable and fast to retrain."""
    cfg = cfg or load_config()
    params = dict(cfg.get("models.logistic_regression", {}) or {})
    return LogisticRegression(
        max_iter=int(params.get("max_iter", 2000)),
        C=float(params.get("C", 1.0)),
        solver=str(params.get("solver", "lbfgs")),
        class_weight="balanced" if balanced else None,
        random_state=cfg.seed,
        n_jobs=None,
    )
