"""Hyperparameter optimisation.

Randomised search rather than grid search: with 6-7 hyper-parameters the grid is
combinatorially large and most dimensions barely move the score, so a fixed
budget of random draws explores the space far more efficiently than an
exhaustive sweep of a coarser grid.

Two properties make this leakage-safe:

* The search runs **inside the training fold only**, using stratified k-fold
  cross-validation. The validation split is never touched, so it remains a clean
  basis for model selection and threshold choice afterwards.
* Scoring is ``average_precision`` (PR-AUC), matching the project's selection
  metric. Tuning on accuracy or ROC-AUC here would optimise for the wrong thing
  and quietly undo the rest of the design.

Stratification within CV matters at a 0.17% base rate: an unstratified fold can
easily contain too few positives for average precision to be meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from ..config import Config
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Search spaces are centred on the configured defaults rather than spanning the
# whole plausible range, so a short budget is spent refining a sane model rather
# than rediscovering that a depth-1 tree is bad.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "xgboost": {
        "max_depth": randint(3, 9),
        "learning_rate": loguniform(0.02, 0.3),
        "n_estimators": randint(150, 600),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda": loguniform(0.1, 10.0),
        "min_child_weight": randint(1, 10),
    },
    "lightgbm": {
        "num_leaves": randint(15, 80),
        "learning_rate": loguniform(0.02, 0.3),
        "n_estimators": randint(150, 600),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 40),
    },
    "random_forest": {
        "n_estimators": randint(100, 400),
        "max_depth": randint(6, 24),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", None],
    },
    "logistic_regression": {
        "C": loguniform(0.01, 100.0),
    },
}


@dataclass
class TuningResult:
    """Outcome of a randomised search for one estimator."""

    model_name: str
    best_params: dict[str, Any]
    best_cv_score: float
    baseline_cv_score: float
    n_candidates: int
    scoring: str = "average_precision"
    improved: bool = field(default=False)

    def summary(self) -> str:
        delta = self.best_cv_score - self.baseline_cv_score
        verdict = "kept tuned params" if self.improved else "kept configured defaults"
        return (
            f"{self.model_name}: CV {self.scoring} {self.baseline_cv_score:.4f} -> "
            f"{self.best_cv_score:.4f} ({delta:+.4f}) over {self.n_candidates} candidates; "
            f"{verdict}."
        )


def _cross_val_score(estimator: Any, X: np.ndarray, y: np.ndarray, cv: Any) -> float:
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(estimator, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    return float(np.mean(scores))


def tune_estimator(
    model_name: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, TuningResult | None]:
    """Search hyper-parameters for one estimator on the training fold.

    Returns the estimator to use (tuned or original) and the tuning record. The
    configured defaults are kept when the search fails to beat them, so tuning
    can never make the pipeline worse.
    """
    space = SEARCH_SPACES.get(model_name)
    if space is None:
        logger.info("No search space defined for '%s'; using configured defaults.", model_name)
        return estimator, None

    n_iter = int(cfg.get("evaluation.hpo_n_iter", 12))
    n_splits = int(cfg.get("evaluation.cv_folds", 3))

    # Stratified CV needs at least one positive per fold.
    n_positive = int(np.sum(y_train == 1))
    if n_positive < n_splits * 2:
        logger.warning(
            "Only %d positives in the training fold; skipping HPO for '%s' "
            "(too few for %d-fold CV to be meaningful).",
            n_positive,
            model_name,
            n_splits,
        )
        return estimator, None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    baseline = _cross_val_score(estimator, X_train, y_train, cv)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=cfg.seed,
        n_jobs=1,  # the estimators already parallelise internally
        refit=False,  # the pipeline refits on the full training fold itself
        error_score=0.0,  # a failed candidate scores zero rather than aborting
    )
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    search.fit(X_train, y_train, **fit_kwargs)

    improved = bool(search.best_score_ > baseline)
    result = TuningResult(
        model_name=model_name,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        baseline_cv_score=baseline,
        n_candidates=n_iter,
        improved=improved,
    )
    logger.info(result.summary())

    if not improved:
        return estimator, result

    tuned = estimator.__class__(**{**estimator.get_params(), **search.best_params_})
    return tuned, result
