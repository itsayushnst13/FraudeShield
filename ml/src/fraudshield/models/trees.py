"""Tree-ensemble models.

Gradient-boosted trees are the workhorse for tabular fraud detection: they
capture the non-linear interactions among the PCA components without feature
engineering, handle unscaled inputs, train in seconds, and expose exact
TreeSHAP attributions for per-decision explanations.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def build_random_forest(cfg: Config | None = None, balanced: bool = True) -> RandomForestClassifier:
    """Bagged trees with balanced subsample weighting."""
    cfg = cfg or load_config()
    params = dict(cfg.get("models.random_forest", {}) or {})
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        n_jobs=int(params.get("n_jobs", -1)),
        class_weight="balanced_subsample" if balanced else None,
        random_state=cfg.seed,
    )


def build_xgboost(cfg: Config | None = None, scale_pos_weight: float | None = None) -> Any:
    """XGBoost classifier configured for imbalanced binary classification."""
    from xgboost import XGBClassifier

    cfg = cfg or load_config()
    params = dict(cfg.get("models.xgboost", {}) or {})
    return XGBClassifier(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        tree_method=str(params.get("tree_method", "hist")),
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=cfg.seed,
        n_jobs=-1,
    )


def build_lightgbm(cfg: Config | None = None, scale_pos_weight: float | None = None) -> Any:
    """LightGBM classifier; returns None when the package is unavailable.

    ``scale_pos_weight`` is honoured only when ``use_scale_pos_weight`` is set in
    config. LightGBM's leaf-wise growth collapses under the ~578:1 weight this
    dataset implies - measured PR-AUC fell from 0.84 to 0.06 - so imbalance is
    handled through L2 regularisation and leaf-size floors instead.
    """
    if not lightgbm_available():
        logger.warning("LightGBM is not installed - skipping this model.")
        return None
    from lightgbm import LGBMClassifier

    cfg = cfg or load_config()
    params = dict(cfg.get("models.lightgbm", {}) or {})
    use_spw = bool(params.get("use_scale_pos_weight", False))
    if scale_pos_weight is not None and not use_spw:
        logger.info(
            "LightGBM: ignoring scale_pos_weight=%.1f "
            "(disabled in config; it destabilises leaf-wise growth here).",
            scale_pos_weight,
        )
    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        num_leaves=int(params.get("num_leaves", 16)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        min_child_samples=int(params.get("min_child_samples", 30)),
        reg_lambda=float(params.get("reg_lambda", 5.0)),
        verbose=int(params.get("verbose", -1)),
        scale_pos_weight=scale_pos_weight if use_spw else None,
        random_state=cfg.seed,
        n_jobs=-1,
    )
