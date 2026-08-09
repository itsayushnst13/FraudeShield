"""End-to-end training pipeline.

Execution order and the leakage controls that make it defensible:

1. Split first (stratified). Test is sealed until step 8.
2. Fit the preprocessor on TRAIN only, then apply it to val/test.
3. Imbalance study: strategies are applied to the training fold only.
4. Train every candidate on the same preprocessed train arrays.
5. Score candidates on VALIDATION.
6. Choose each candidate's operating threshold on VALIDATION.
7. Select the production model on VALIDATION PR-AUC.
8. Evaluate the winner on TEST exactly once - a clean generalisation estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..data.loader import load_raw
from ..data.splits import DataSplits, make_splits
from ..evaluation.metrics import compute_metrics, pr_curve_points, roc_curve_points
from ..evaluation.threshold import select_threshold
from ..features.preprocess import fit_preprocessor, get_output_feature_names
from ..imbalance import apply_strategy, assert_no_resampling_on_eval
from ..logging_utils import get_logger
from ..models.registry import ModelMetadata, save_artifact
from ..models.trees import build_xgboost
from .tracking import ExperimentTracker
from .trainers import (
    TrainedCandidate,
    train_autoencoder,
    train_baselines,
    train_torch_mlp,
    train_trees,
)

logger = get_logger(__name__)

# Two models whose primary metric differs by less than this are treated as tied.
TIE_TOLERANCE = 0.005


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for reporting and assertions."""

    comparison: pd.DataFrame
    imbalance_study: pd.DataFrame
    best_name: str
    best_threshold: float
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    artifact_dir: Path
    data_source: str


def run_imbalance_study(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: Config
) -> pd.DataFrame:
    """Compare imbalance strategies with XGBoost held fixed as the probe model."""
    rows = []
    for strategy in cfg.get("imbalance.strategies", ["none", "class_weight"]):
        resampled = apply_strategy(X_train, y_train, strategy, cfg)
        model = build_xgboost(cfg, scale_pos_weight=resampled.scale_pos_weight)
        model.fit(resampled.X, resampled.y)
        probs = model.predict_proba(X_val)[:, 1]
        assert_no_resampling_on_eval(y_val, y_val, "validation")
        metrics = compute_metrics(y_val, probs, 0.5, f"xgboost[{strategy}]")
        rows.append(
            {
                "strategy": strategy,
                "train_rows": int(len(resampled.y)),
                "train_positives": resampled.n_positive,
                **{
                    k: round(v, 4)
                    for k, v in (
                        ("pr_auc", metrics.pr_auc),
                        ("roc_auc", metrics.roc_auc),
                        ("precision@0.5", metrics.precision),
                        ("recall@0.5", metrics.recall),
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def _train_all_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    checkpoint_dir: Path,
    include_deep: bool = True,
    tune: bool = False,
) -> list[TrainedCandidate]:
    strategy = str(cfg.get("imbalance.default_strategy", "class_weight"))
    candidates = train_baselines(X_train, y_train, X_val, cfg)
    candidates += train_trees(X_train, y_train, X_val, cfg, strategy=strategy, tune=tune)
    if include_deep:
        candidates.append(train_torch_mlp(X_train, y_train, X_val, y_val, cfg, checkpoint_dir))
        candidates.append(train_autoencoder(X_train, y_train, X_val, cfg, checkpoint_dir))
    return candidates


def run_pipeline(
    cfg: Config | None = None,
    data_path: str | Path | None = None,
    data_source: str = "KAGGLE_CREDITCARD",
    include_deep: bool = True,
    model_name: str = "fraud_detector",
    version: str = "1.0.0",
    tune: bool | None = None,
) -> PipelineResult:
    """Run the full training pipeline and register the winning artifact.

    ``tune`` defaults to the ``evaluation.hyperparameter_search`` config flag.
    """
    cfg = cfg or load_config()
    if tune is None:
        tune = bool(cfg.get("evaluation.hyperparameter_search", False))
    root = cfg.root
    reports_dir = cfg.path("paths.reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cfg.path("paths.models_dir") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df, validation = load_raw(data_path, cfg=cfg)
    splits: DataSplits = make_splits(df, cfg)

    preprocessor = fit_preprocessor(splits.X_train, cfg)
    feature_names = get_output_feature_names(preprocessor)
    X_train = np.asarray(preprocessor.transform(splits.X_train), dtype=float)
    X_val = np.asarray(preprocessor.transform(splits.X_val), dtype=float)
    X_test = np.asarray(preprocessor.transform(splits.X_test), dtype=float)
    y_train = splits.y_train.to_numpy()
    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()

    tracker = ExperimentTracker(
        tracking_uri=str(cfg.get("paths.mlflow_tracking_uri", "sqlite:///mlflow.db")),
        experiment_name=f"{cfg.get('project.name', 'fraudshield')}-{data_source}",
        root=root,
        artifacts_dir=str(cfg.get("paths.mlflow_artifacts_dir", "mlartifacts")),
    )

    logger.info("=== Imbalance study (XGBoost probe) ===")
    imbalance_study = run_imbalance_study(X_train, y_train, X_val, y_val, cfg)
    imbalance_study.to_csv(reports_dir / "imbalance_comparison.csv", index=False)
    logger.info("\n%s", imbalance_study.to_string(index=False))

    logger.info("=== Training candidates ===")
    if tune:
        logger.info(
            "Hyperparameter search enabled: %d candidates, %d-fold stratified CV on the "
            "training fold, scored by average precision.",
            int(cfg.get("evaluation.hpo_n_iter", 12)),
            int(cfg.get("evaluation.cv_folds", 3)),
        )
    candidates = _train_all_candidates(
        X_train, y_train, X_val, y_val, cfg, checkpoint_dir, include_deep, tune=tune
    )

    rows, per_model, tuning_records = [], {}, []
    for candidate in candidates:
        if candidate.tuning is not None:
            tuning_records.append(
                {
                    "model": candidate.tuning.model_name,
                    "baseline_cv_pr_auc": round(candidate.tuning.baseline_cv_score, 6),
                    "tuned_cv_pr_auc": round(candidate.tuning.best_cv_score, 6),
                    "applied": candidate.tuning.improved,
                    "n_candidates": candidate.tuning.n_candidates,
                    "best_params": {k: str(v) for k, v in candidate.tuning.best_params.items()},
                }
            )
        choice = select_threshold(y_val, candidate.val_probs, cfg)
        metrics = compute_metrics(y_val, candidate.val_probs, choice.threshold, candidate.name)
        rows.append({**metrics.to_row(), "imbalance": candidate.imbalance_strategy})
        per_model[candidate.name] = {"candidate": candidate, "choice": choice, "metrics": metrics}

        with tracker.run(
            run_name=candidate.name,
            tags={"data_source": data_source, "model_type": candidate.model_type},
        ):
            tracker.log_params(
                {
                    "model": candidate.name,
                    "imbalance": candidate.imbalance_strategy,
                    "threshold": choice.threshold,
                    **candidate.params,
                }
            )
            tracker.log_metrics(
                {f"val_{k}": v for k, v in metrics.to_dict().items() if isinstance(v, (int, float))}
            )
            for curve_name, values in (candidate.history or {}).items():
                tracker.log_metric_series(curve_name, values)
        choice.sweep.to_csv(reports_dir / f"threshold_sweep_{candidate.name}.csv", index=False)

    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    logger.info("=== Validation comparison ===\n%s", comparison.to_string(index=False))

    primary = str(cfg.get("selection.primary_metric", "pr_auc"))
    # The autoencoder is an unsupervised tripwire, not a scoring candidate: its
    # outputs are rank scores rather than calibrated probabilities, so it is
    # excluded from selection for the production scoring path.
    eligible = comparison[~comparison["model"].isin(["dummy", "autoencoder"])]

    # Near-ties on PR-AUC are common and not meaningful at 79 validation
    # positives, so break them on the configured tie-breaker (recall by default)
    # rather than letting float noise pick the production model.
    tie_breaker = str(cfg.get("selection.tie_breaker", "recall"))
    tie_column = "recall" if "recall" in tie_breaker else tie_breaker
    top_score = float(eligible[primary].max())
    contenders = eligible[eligible[primary] >= top_score - TIE_TOLERANCE]
    if len(contenders) > 1 and tie_column in contenders.columns:
        logger.info(
            "%d candidates within %.3f of the best %s; breaking the tie on '%s'.",
            len(contenders),
            TIE_TOLERANCE,
            primary,
            tie_column,
        )
        best_name = str(contenders.sort_values(tie_column, ascending=False).iloc[0]["model"])
    else:
        best_name = str(eligible.sort_values(primary, ascending=False).iloc[0]["model"])
    best = per_model[best_name]
    best_threshold = best["choice"].threshold
    best_model = best["candidate"].model

    logger.info("=== Final evaluation on held-out TEST (first and only look) ===")
    test_probs = np.asarray(best_model.predict_proba(X_test))[:, 1]
    test_metrics = compute_metrics(y_test, test_probs, best_threshold, best_name)
    logger.info("Test metrics: %s", test_metrics.to_row())

    selection_rationale = (
        f"Selected '{best_name}' on validation {primary}="
        f"{best['metrics'].to_dict()[primary]:.4f}, the highest among eligible supervised "
        f"candidates. Threshold {best_threshold:.2f} chosen on validation; test set was "
        f"evaluated once afterwards."
    )

    metadata = ModelMetadata(
        model_name=model_name,
        model_type=best["candidate"].model_type,
        version=version,
        threshold=best_threshold,
        features=feature_names,
        data_source=data_source,
        n_train_rows=int(len(y_train)),
        n_train_fraud=int(y_train.sum()),
        imbalance_strategy=best["candidate"].imbalance_strategy,
        validation_metrics={
            k: float(v) for k, v in best["metrics"].to_dict().items() if isinstance(v, (int, float))
        },
        test_metrics={
            k: float(v) for k, v in test_metrics.to_dict().items() if isinstance(v, (int, float))
        },
        threshold_rationale=best["choice"].rationale,
        selection_rationale=selection_rationale,
        random_seed=cfg.seed,
        extra={
            "dataset_rows": validation.n_rows,
            "dataset_fraud": validation.n_fraud,
            "hyperparameter_search": tuning_records,
        },
    )
    artifact_dir = cfg.path("paths.models_dir") / model_name
    save_artifact(artifact_dir, preprocessor, best_model, metadata)

    # Persist a background sample so the serving layer can build a SHAP explainer
    # for non-tree models without needing the training data at runtime.
    rng = np.random.default_rng(cfg.seed)
    background = X_train[rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)]
    np.save(artifact_dir / "shap_background.npy", background)

    payload = {
        "data_source": data_source,
        "is_synthetic": metadata.is_synthetic,
        "best_model": best_name,
        "threshold": best_threshold,
        "validation": best["metrics"].to_dict(),
        "test": test_metrics.to_dict(),
        "confusion_matrix": test_metrics.confusion,
        "pr_curve": pr_curve_points(y_test, test_probs),
        "roc_curve": roc_curve_points(y_test, test_probs),
        "split_summary": splits.summary().to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "imbalance_study": imbalance_study.to_dict(orient="records"),
        "selection_rationale": selection_rationale,
        "threshold_rationale": best["choice"].rationale,
        "hyperparameter_search": tuning_records,
    }
    if tuning_records:
        pd.DataFrame(tuning_records).to_csv(reports_dir / "hpo_results.csv", index=False)
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote reports to %s", reports_dir)

    return PipelineResult(
        comparison=comparison,
        imbalance_study=imbalance_study,
        best_name=best_name,
        best_threshold=best_threshold,
        val_metrics=best["metrics"].to_dict(),
        test_metrics=test_metrics.to_dict(),
        artifact_dir=artifact_dir,
        data_source=data_source,
    )
