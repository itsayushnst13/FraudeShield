"""Tests for metrics, threshold optimisation and the accuracy trap."""

from __future__ import annotations

import numpy as np
import pytest

from fraudshield.evaluation.metrics import compute_metrics, pr_curve_points, roc_curve_points
from fraudshield.evaluation.threshold import select_threshold, sweep_thresholds


def test_confusion_counts_are_exact():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    m = compute_metrics(y, p, threshold=0.5)
    assert (m.true_positives, m.false_positives, m.true_negatives, m.false_negatives) == (
        1,
        1,
        1,
        1,
    )
    assert m.precision == pytest.approx(0.5) and m.recall == pytest.approx(0.5)


def test_perfect_and_inverted_rankings():
    y = np.array([0] * 50 + [1] * 50)
    perfect = compute_metrics(y, np.concatenate([np.zeros(50), np.ones(50)]), 0.5)
    assert perfect.pr_auc == pytest.approx(1.0) and perfect.roc_auc == pytest.approx(1.0)
    inverted = compute_metrics(y, np.concatenate([np.ones(50), np.zeros(50)]), 0.5)
    assert inverted.roc_auc == pytest.approx(0.0)


def test_accuracy_is_misleading_under_extreme_imbalance():
    """The core motivation for PR-AUC: a useless model still scores ~99.8% accuracy."""
    y = np.concatenate([np.zeros(9983, dtype=int), np.ones(17, dtype=int)])
    always_legit = np.zeros(10000)
    m = compute_metrics(y, always_legit, threshold=0.5)
    assert m.accuracy > 0.998
    assert m.recall == 0.0 and m.f1 == 0.0
    assert m.pr_auc < 0.01


def test_f1_is_zero_when_nothing_is_flagged():
    y = np.array([0, 0, 1])
    m = compute_metrics(y, np.array([0.1, 0.1, 0.2]), threshold=0.9)
    assert m.precision == 0.0 and m.f1 == 0.0


def test_metrics_reject_shape_mismatch_and_empty_input():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))


def test_threshold_changes_precision_recall_tradeoff(scores):
    y, p = scores
    low = compute_metrics(y, p, 0.1)
    high = compute_metrics(y, p, 0.9)
    assert low.recall >= high.recall
    assert high.precision >= low.precision


def test_pr_auc_is_threshold_independent(scores):
    y, p = scores
    assert compute_metrics(y, p, 0.2).pr_auc == pytest.approx(compute_metrics(y, p, 0.8).pr_auc)


def test_sweep_covers_grid_and_recall_is_monotone_decreasing(scores, cfg):
    y, p = scores
    sweep = sweep_thresholds(y, p, cfg)
    assert len(sweep) == len(cfg.get("threshold.grid"))
    recalls = sweep.sort_values("threshold")["recall"].to_numpy()
    assert np.all(np.diff(recalls) <= 1e-9)


def test_expected_cost_weights_missed_fraud_more_heavily(scores, cfg):
    y, p = scores
    sweep = sweep_thresholds(y, p, cfg)
    row = sweep.iloc[0]
    expected = (
        cfg.get("threshold.cost_false_negative") * row["false_negatives"]
        + cfg.get("threshold.cost_false_positive") * row["false_positives"]
    )
    assert row["expected_cost"] == pytest.approx(expected)


def test_selected_threshold_is_in_grid_and_beats_default(scores, cfg):
    y, p = scores
    choice = select_threshold(y, p, cfg)
    assert choice.threshold in cfg.get("threshold.grid")
    assert choice.rationale
    sweep = choice.sweep.set_index("threshold")
    assert sweep.loc[choice.threshold, "expected_cost"] <= sweep["expected_cost"].max()


def test_selection_respects_min_precision_guardrail(scores, cfg):
    y, p = scores
    choice = select_threshold(y, p, cfg)
    row = choice.sweep.set_index("threshold").loc[choice.threshold]
    assert row["precision"] >= cfg.get("threshold.min_precision")


def test_selection_is_deterministic(scores, cfg):
    y, p = scores
    assert select_threshold(y, p, cfg).threshold == select_threshold(y, p, cfg).threshold


def test_curve_points_are_serialisable(scores):
    y, p = scores
    pr, roc = pr_curve_points(y, p), roc_curve_points(y, p)
    assert len(pr["precision"]) == len(pr["recall"]) > 0
    assert len(roc["fpr"]) == len(roc["tpr"]) > 0
    assert all(isinstance(v, float) for v in pr["precision"])
