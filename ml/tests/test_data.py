"""Tests for schema validation, loading and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.data.loader import DatasetNotFoundError, load_raw
from fraudshield.data.schema import ALL_COLUMNS, TARGET_COLUMN, validate_dataframe
from fraudshield.data.splits import make_splits


def test_synthetic_frame_matches_expected_schema(frame):
    assert list(frame.columns) == ALL_COLUMNS
    assert frame[TARGET_COLUMN].isin([0, 1]).all()
    assert len(frame) == 3000


def test_validation_accepts_wellformed_frame(frame):
    result = validate_dataframe(frame)
    assert result.ok and not result.errors
    assert result.n_fraud > 0
    assert 0 < result.fraud_rate < 1


def test_validation_rejects_missing_columns(frame):
    result = validate_dataframe(frame.drop(columns=["V5"]))
    assert not result.ok
    assert any("V5" in e for e in result.errors)


def test_validation_rejects_nulls_and_negative_amounts(frame):
    broken = frame.copy()
    broken.loc[0, "V1"] = np.nan
    broken.loc[1, "Amount"] = -10.0
    result = validate_dataframe(broken)
    assert not result.ok
    assert any("null" in e for e in result.errors)
    assert any("Negative" in e for e in result.errors)


def test_validation_rejects_non_binary_target(frame):
    broken = frame.copy()
    broken.loc[0, TARGET_COLUMN] = 7
    result = validate_dataframe(broken)
    assert not result.ok


def test_validation_rejects_frame_with_no_fraud(frame):
    result = validate_dataframe(frame.assign(Class=0))
    assert not result.ok
    assert any("positive" in e for e in result.errors)


def test_raise_if_invalid_raises(frame):
    with pytest.raises(ValueError):
        validate_dataframe(frame.drop(columns=["Amount"])).raise_if_invalid()


def test_missing_dataset_raises_with_setup_instructions(tmp_path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_raw(tmp_path / "absent.csv")
    assert "kaggle" in str(exc.value).lower()


def test_load_raw_reads_and_validates(tmp_path, frame):
    path = tmp_path / "creditcard.csv"
    frame.to_csv(path, index=False)
    loaded, result = load_raw(path)
    assert result.ok
    assert len(loaded) == len(frame)


def test_splits_are_disjoint_and_cover_all_rows(frame):
    s = make_splits(frame)
    total = len(s.X_train) + len(s.X_val) + len(s.X_test)
    assert total == len(frame)
    indices = [set(x.index) for x in (s.X_train, s.X_val, s.X_test)]
    assert not (indices[0] & indices[1]) and not (indices[0] & indices[2])
    assert not (indices[1] & indices[2])


def test_splits_preserve_fraud_rate(frame):
    s = make_splits(frame)
    base = frame["Class"].mean()
    for y in (s.y_train, s.y_val, s.y_test):
        assert y.mean() == pytest.approx(base, abs=0.01)


def test_splits_exclude_target_from_features(frame):
    s = make_splits(frame)
    assert TARGET_COLUMN not in s.X_train.columns


def test_splits_are_reproducible(frame):
    a, b = make_splits(frame), make_splits(frame)
    pd.testing.assert_index_equal(a.X_test.index, b.X_test.index)
