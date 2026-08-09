"""End-to-end pipeline test: training through artifact registration."""

from __future__ import annotations

import json

import pytest

from fraudshield.config import load_config
from fraudshield.models.registry import load_artifact
from fraudshield.training.pipeline import run_imbalance_study, run_pipeline


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory, frame_for_pipeline):
    """Run the real pipeline once against a temp workspace."""
    workspace = tmp_path_factory.mktemp("run")
    data_path = workspace / "creditcard.csv"
    frame_for_pipeline.to_csv(data_path, index=False)

    cfg = load_config()
    cfg._data["paths"]["models_dir"] = str(workspace / "models")
    cfg._data["paths"]["reports_dir"] = str(workspace / "reports")
    cfg._data["models"]["mlp"].update({"max_epochs": 3, "batch_size": 256})
    cfg._data["models"]["autoencoder"].update({"max_epochs": 3, "batch_size": 256})
    cfg._data["models"]["random_forest"]["n_estimators"] = 40
    cfg._data["models"]["xgboost"]["n_estimators"] = 40
    cfg._data["models"]["lightgbm"]["n_estimators"] = 40

    # HPO is exercised by ml/tests/test_hpo.py; disabled here to keep the
    # integration test fast.
    result = run_pipeline(cfg=cfg, data_path=data_path, data_source="SYNTHETIC_SMOKE", tune=False)
    return result, workspace, cfg


@pytest.fixture(scope="module")
def frame_for_pipeline():
    from fraudshield.data.synthetic import generate_synthetic_frame

    return generate_synthetic_frame(n_rows=4000, fraud_rate=0.03, seed=13)


def test_pipeline_trains_every_required_model(pipeline_run):
    result, _, _ = pipeline_run
    trained = set(result.comparison["model"])
    assert {
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "torch_mlp",
        "autoencoder",
    }.issubset(trained)


def test_selected_model_is_not_the_dummy_or_autoencoder(pipeline_run):
    result, _, _ = pipeline_run
    assert result.best_name not in ("dummy", "autoencoder")


def test_dummy_baseline_is_beaten_on_pr_auc(pipeline_run):
    result, _, _ = pipeline_run
    table = result.comparison.set_index("model")
    assert table.loc[result.best_name, "pr_auc"] > table.loc["dummy", "pr_auc"]


def test_threshold_is_selected_rather_than_defaulted(pipeline_run):
    result, _, cfg = pipeline_run
    assert result.best_threshold in cfg.get("threshold.grid")


def test_artifact_bundle_is_registered_and_loadable(pipeline_run):
    result, _, _ = pipeline_run
    artifact = load_artifact(result.artifact_dir)
    assert artifact.metadata.model_name == "fraud_detector"
    assert artifact.metadata.threshold == result.best_threshold
    assert artifact.metadata.is_synthetic is True
    assert len(artifact.metadata.features) == 31
    assert artifact.metadata.selection_rationale and artifact.metadata.threshold_rationale


def test_test_metrics_are_recorded_and_plausible(pipeline_run):
    result, _, _ = pipeline_run
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        assert 0.0 <= result.test_metrics[key] <= 1.0
    assert result.test_metrics["n_positive"] > 0


def test_reports_are_written(pipeline_run):
    _, workspace, _ = pipeline_run
    reports = workspace / "reports"
    assert (reports / "model_comparison.csv").is_file()
    assert (reports / "imbalance_comparison.csv").is_file()
    payload = json.loads((reports / "metrics.json").read_text())
    assert payload["is_synthetic"] is True
    assert len(payload["confusion_matrix"]) == 2
    assert payload["pr_curve"]["precision"] and payload["roc_curve"]["fpr"]


def test_imbalance_study_compares_every_configured_strategy(frame_for_pipeline):
    import numpy as np

    from fraudshield.data.splits import make_splits
    from fraudshield.features.preprocess import fit_preprocessor

    cfg = load_config()
    splits = make_splits(frame_for_pipeline, cfg)
    pre = fit_preprocessor(splits.X_train, cfg)
    study = run_imbalance_study(
        np.asarray(pre.transform(splits.X_train), dtype=float),
        splits.y_train.to_numpy(),
        np.asarray(pre.transform(splits.X_val), dtype=float),
        splits.y_val.to_numpy(),
        cfg,
    )
    assert set(study["strategy"]) == set(cfg.get("imbalance.strategies"))
    assert (study["pr_auc"] >= 0).all()


def test_pipeline_records_hyperparameter_search_when_enabled(tmp_path, frame_for_pipeline):
    """With tuning on, the run must record what was searched and whether it helped."""
    import json

    workspace = tmp_path / "tuned"
    data_path = workspace / "creditcard.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame_for_pipeline.to_csv(data_path, index=False)

    cfg = load_config()
    cfg._data["paths"]["models_dir"] = str(workspace / "models")
    cfg._data["paths"]["reports_dir"] = str(workspace / "reports")
    cfg._data["evaluation"]["hpo_n_iter"] = 3
    cfg._data["evaluation"]["cv_folds"] = 3
    cfg._data["models"]["random_forest"]["n_estimators"] = 30
    cfg._data["models"]["xgboost"]["n_estimators"] = 30
    cfg._data["models"]["lightgbm"]["n_estimators"] = 30

    run_pipeline(
        cfg=cfg,
        data_path=data_path,
        data_source="SYNTHETIC_SMOKE",
        include_deep=False,
        tune=True,
    )

    payload = json.loads((workspace / "reports" / "metrics.json").read_text())
    records = payload["hyperparameter_search"]
    assert records, "tuning was enabled but nothing was recorded"
    assert (workspace / "reports" / "hpo_results.csv").is_file()
    for record in records:
        assert record["n_candidates"] == 3
        assert "baseline_cv_pr_auc" in record and "tuned_cv_pr_auc" in record
        assert isinstance(record["applied"], bool)


def test_selection_breaks_near_ties_on_the_configured_metric():
    """Float noise on PR-AUC must not decide which model ships."""
    import pandas as pd

    from fraudshield.training.pipeline import TIE_TOLERANCE

    comparison = pd.DataFrame(
        [
            {"model": "xgboost", "pr_auc": 0.8330, "recall": 0.8101},
            {"model": "lightgbm", "pr_auc": 0.8332, "recall": 0.7595},
            {"model": "dummy", "pr_auc": 0.0017, "recall": 0.0},
        ]
    )
    eligible = comparison[~comparison["model"].isin(["dummy", "autoencoder"])]
    top = float(eligible["pr_auc"].max())
    contenders = eligible[eligible["pr_auc"] >= top - TIE_TOLERANCE]
    assert len(contenders) == 2, "0.0002 apart must count as a tie"
    winner = contenders.sort_values("recall", ascending=False).iloc[0]["model"]
    assert winner == "xgboost", "the higher-recall model should win a PR-AUC tie"
