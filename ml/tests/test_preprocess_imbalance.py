"""Tests for preprocessing and class-imbalance handling, focused on leakage."""

from __future__ import annotations

import numpy as np
import pytest

from fraudshield.data.splits import make_splits
from fraudshield.features.preprocess import (
    build_preprocessor,
    fit_preprocessor,
    get_output_feature_names,
)
from fraudshield.imbalance import (
    apply_strategy,
    assert_no_resampling_on_eval,
    compute_sample_weights,
    compute_scale_pos_weight,
)


def test_preprocessor_drops_raw_time_and_adds_cyclical_hour(frame):
    pipeline = fit_preprocessor(frame.drop(columns=["Class"]))
    names = get_output_feature_names(pipeline)
    assert "Time" not in names and "time_raw" not in names
    assert {"hour_sin", "hour_cos", "amount_log"}.issubset(set(names))
    assert len([n for n in names if n.startswith("V")]) == 28


def test_preprocessor_output_shape_and_finiteness(frame):
    X = frame.drop(columns=["Class"])
    out = fit_preprocessor(X).transform(X)
    assert out.shape == (len(X), 31)
    assert np.isfinite(out).all()


def test_scaler_is_fitted_on_training_data_only(frame):
    """Val/test statistics must not influence the scaler."""
    s = make_splits(frame)
    pipeline = fit_preprocessor(s.X_train)
    train_out = pipeline.transform(s.X_train)
    # Training data standardises to ~zero mean; held-out data generally will not.
    assert abs(float(train_out.mean())) < 0.05
    fitted_on_all = fit_preprocessor(frame.drop(columns=["Class"]))
    assert not np.allclose(
        pipeline.named_steps["scaler"].scale_, fitted_on_all.named_steps["scaler"].scale_
    )


def test_transform_is_column_order_invariant(frame):
    X = frame.drop(columns=["Class"])
    pipeline = fit_preprocessor(X)
    shuffled = X[list(reversed(X.columns))]
    np.testing.assert_allclose(pipeline.transform(X), pipeline.transform(shuffled))


def test_missing_columns_are_filled_not_crashed(frame):
    X = frame.drop(columns=["Class"])
    pipeline = fit_preprocessor(X)
    out = pipeline.transform(X.drop(columns=["V3", "V7"]))
    assert out.shape[1] == 31


def test_unfitted_preprocessor_is_a_pipeline():
    pipeline = build_preprocessor()
    assert hasattr(pipeline, "fit") and hasattr(pipeline, "transform")


def test_scale_pos_weight_matches_class_ratio():
    y = np.array([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_scale_pos_weight_requires_positives():
    with pytest.raises(ValueError):
        compute_scale_pos_weight(np.zeros(10, dtype=int))


def test_sample_weights_upweight_the_minority():
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_sample_weights(y)
    assert weights[y == 1].mean() > weights[y == 0].mean()


@pytest.mark.parametrize("strategy", ["none", "class_weight", "undersample", "smote"])
def test_every_strategy_returns_usable_training_arrays(strategy, frame, cfg):
    s = make_splits(frame)
    X = fit_preprocessor(s.X_train).transform(s.X_train)
    result = apply_strategy(X, s.y_train.to_numpy(), strategy, cfg)
    assert result.n_positive > 0 and len(result.X) == len(result.y)
    assert result.strategy == strategy


def test_smote_adds_synthetic_positives_only_to_training(frame, cfg):
    s = make_splits(frame)
    X = fit_preprocessor(s.X_train).transform(s.X_train)
    y = s.y_train.to_numpy()
    result = apply_strategy(X, y, "smote", cfg)
    assert result.n_positive > int(y.sum())
    # The validation labels object is untouched by resampling.
    assert len(s.y_val) == len(s.X_val)


def test_class_weight_strategy_does_not_resample(frame, cfg):
    s = make_splits(frame)
    X = fit_preprocessor(s.X_train).transform(s.X_train)
    result = apply_strategy(X, s.y_train.to_numpy(), "class_weight", cfg)
    assert len(result.y) == len(s.y_train)
    assert result.sample_weight is not None and result.scale_pos_weight is not None


def test_undersample_shrinks_the_majority(frame, cfg):
    s = make_splits(frame)
    X = fit_preprocessor(s.X_train).transform(s.X_train)
    result = apply_strategy(X, s.y_train.to_numpy(), "undersample", cfg)
    assert result.n_negative < int((s.y_train == 0).sum())


def test_unknown_strategy_is_rejected(frame, cfg):
    s = make_splits(frame)
    X = fit_preprocessor(s.X_train).transform(s.X_train)
    with pytest.raises(ValueError):
        apply_strategy(X, s.y_train.to_numpy(), "magic", cfg)


def test_leakage_guard_passes_on_untouched_eval_split():
    y = np.array([0] * 95 + [1] * 5)
    assert_no_resampling_on_eval(y, y, "validation")


def test_leakage_guard_trips_when_eval_split_is_resampled():
    y = np.array([0] * 95 + [1] * 5)
    resampled = np.concatenate([y, np.ones(20, dtype=int)])
    with pytest.raises(AssertionError):
        assert_no_resampling_on_eval(y, resampled, "validation")
