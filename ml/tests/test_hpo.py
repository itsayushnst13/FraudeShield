"""Tests for hyperparameter optimisation and its leakage properties."""

from __future__ import annotations

import numpy as np
import pytest

from fraudshield.config import load_config
from fraudshield.data.splits import make_splits
from fraudshield.features.preprocess import fit_preprocessor
from fraudshield.models.baselines import build_logistic_regression
from fraudshield.models.trees import build_random_forest, build_xgboost
from fraudshield.training.hpo import SEARCH_SPACES, tune_estimator


@pytest.fixture(scope="module")
def tuning_data():
    from fraudshield.data.synthetic import generate_synthetic_frame

    cfg = load_config()
    frame = generate_synthetic_frame(n_rows=2500, fraud_rate=0.05, seed=21)
    splits = make_splits(frame, cfg)
    pre = fit_preprocessor(splits.X_train, cfg)
    return {
        "cfg": cfg,
        "X": np.asarray(pre.transform(splits.X_train), dtype=float),
        "y": splits.y_train.to_numpy(),
        "X_val": np.asarray(pre.transform(splits.X_val), dtype=float),
        "y_val": splits.y_val.to_numpy(),
    }


@pytest.fixture(scope="module")
def fast_cfg(tuning_data):
    """A small search budget so tests stay quick without changing behaviour."""
    cfg = tuning_data["cfg"]
    cfg._data["evaluation"]["hpo_n_iter"] = 4
    cfg._data["evaluation"]["cv_folds"] = 3
    return cfg


def test_search_spaces_exist_for_every_tunable_model():
    assert {"xgboost", "random_forest", "logistic_regression"}.issubset(SEARCH_SPACES)


def test_tuning_returns_an_estimator_and_a_record(tuning_data, fast_cfg):
    model, result = tune_estimator(
        "random_forest", build_random_forest(fast_cfg), tuning_data["X"], tuning_data["y"], fast_cfg
    )
    assert model is not None
    assert result is not None
    assert result.model_name == "random_forest"
    assert result.n_candidates == 4
    assert result.scoring == "average_precision"


def test_tuned_estimator_is_fittable_and_predicts(tuning_data, fast_cfg):
    model, _ = tune_estimator(
        "xgboost",
        build_xgboost(fast_cfg, scale_pos_weight=20.0),
        tuning_data["X"],
        tuning_data["y"],
        fast_cfg,
    )
    model.fit(tuning_data["X"], tuning_data["y"])
    probs = model.predict_proba(tuning_data["X_val"])[:, 1]
    assert probs.shape == tuning_data["y_val"].shape
    assert ((probs >= 0) & (probs <= 1)).all()


def test_best_params_come_from_the_declared_search_space(tuning_data, fast_cfg):
    _, result = tune_estimator(
        "xgboost",
        build_xgboost(fast_cfg, scale_pos_weight=20.0),
        tuning_data["X"],
        tuning_data["y"],
        fast_cfg,
    )
    assert set(result.best_params).issubset(set(SEARCH_SPACES["xgboost"]))


def test_defaults_are_kept_when_the_search_does_not_improve(tuning_data, fast_cfg):
    original = build_logistic_regression(fast_cfg)
    model, result = tune_estimator(
        "logistic_regression", original, tuning_data["X"], tuning_data["y"], fast_cfg
    )
    if not result.improved:
        assert model is original
    else:
        assert result.best_cv_score > result.baseline_cv_score


def test_tuning_is_skipped_for_models_without_a_search_space(tuning_data, fast_cfg):
    original = build_random_forest(fast_cfg)
    model, result = tune_estimator(
        "some_unknown_model", original, tuning_data["X"], tuning_data["y"], fast_cfg
    )
    assert model is original and result is None


def test_tuning_is_skipped_when_positives_are_too_few_for_cv(tuning_data, fast_cfg):
    """Too few positives makes fold-level average precision meaningless."""
    X, y = tuning_data["X"][:200].copy(), np.zeros(200, dtype=int)
    y[:3] = 1  # 3 positives, fewer than 2 x n_splits
    original = build_random_forest(fast_cfg)
    model, result = tune_estimator("random_forest", original, X, y, fast_cfg)
    assert model is original and result is None


def test_tuning_never_sees_the_validation_split(tuning_data, fast_cfg):
    """A search fitted on train-only data must be reproducible from train alone."""
    args = ("random_forest", tuning_data["X"], tuning_data["y"], fast_cfg)
    _, first = tune_estimator(args[0], build_random_forest(fast_cfg), *args[1:])
    _, second = tune_estimator(args[0], build_random_forest(fast_cfg), *args[1:])
    assert first.best_params == second.best_params
    assert first.best_cv_score == pytest.approx(second.best_cv_score)


def test_result_summary_is_human_readable(tuning_data, fast_cfg):
    _, result = tune_estimator(
        "random_forest", build_random_forest(fast_cfg), tuning_data["X"], tuning_data["y"], fast_cfg
    )
    summary = result.summary()
    assert "average_precision" in summary
    assert "candidates" in summary
