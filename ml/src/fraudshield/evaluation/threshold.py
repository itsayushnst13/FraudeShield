"""Threshold optimisation driven by an explicit business objective.

0.5 is an arbitrary default that assumes balanced classes and symmetric error
costs. Neither holds in card fraud: a missed fraud costs the chargeback plus
write-off, while a false alarm costs a few minutes of analyst review. The
optimiser below sweeps candidate thresholds and picks the one minimising

    expected_cost = C_fn * FN + C_fp * FP

subject to a minimum-precision guardrail that keeps alert volume inside what a
review team can absorb. All costs and the guardrail live in configs/config.yaml.

The sweep runs on the VALIDATION split only. Choosing a threshold on test data
would leak the test set into a model decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger
from .metrics import compute_metrics

logger = get_logger(__name__)


@dataclass
class ThresholdChoice:
    """Selected operating point plus the full sweep that justified it."""

    threshold: float
    objective: str
    rationale: str
    sweep: pd.DataFrame
    selected_row: dict[str, Any]


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Evaluate every candidate threshold and return a tidy trade-off table."""
    cfg = cfg or load_config()
    grid = list(cfg.get("threshold.grid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    cost_fn = float(cfg.get("threshold.cost_false_negative", 500.0))
    cost_fp = float(cfg.get("threshold.cost_false_positive", 5.0))

    rows = []
    for threshold in grid:
        m = compute_metrics(y_true, y_prob, threshold=threshold)
        expected_cost = cost_fn * m.false_negatives + cost_fp * m.false_positives
        rows.append(
            {
                "threshold": float(threshold),
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
                "alerts": m.true_positives + m.false_positives,
                "alert_rate": (m.true_positives + m.false_positives) / m.n_samples,
                "expected_cost": expected_cost,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: Config | None = None,
) -> ThresholdChoice:
    """Choose the operating threshold according to the configured objective."""
    cfg = cfg or load_config()
    sweep = sweep_thresholds(y_true, y_prob, cfg)
    objective = str(cfg.get("threshold.objective", "expected_cost"))
    min_precision = float(cfg.get("threshold.min_precision", 0.0))
    cost_fn = float(cfg.get("threshold.cost_false_negative", 500.0))
    cost_fp = float(cfg.get("threshold.cost_false_positive", 5.0))

    eligible = sweep[sweep["precision"] >= min_precision]
    guardrail_applied = not eligible.empty
    if not guardrail_applied:
        logger.warning(
            "No threshold satisfies min_precision=%.3f; falling back to the full grid.",
            min_precision,
        )
        eligible = sweep

    if objective == "f1":
        best = eligible.loc[eligible["f1"].idxmax()]
        rationale = f"Maximised F1 ({best['f1']:.4f}) over the candidate grid."
    elif objective == "max_recall_at_precision":
        best = eligible.loc[eligible["recall"].idxmax()]
        rationale = (
            f"Maximised recall ({best['recall']:.4f}) subject to precision >= {min_precision:.2f}."
        )
    else:
        min_cost = float(eligible["expected_cost"].min())
        tolerance = float(cfg.get("threshold.cost_tolerance", 0.0))
        near_optimal = eligible[eligible["expected_cost"] <= min_cost * (1.0 + tolerance)]
        # Prefer the most precise operating point among those that are
        # statistically indistinguishable on cost.
        best = near_optimal.loc[near_optimal["precision"].idxmax()]
        tie_note = (
            f" {len(near_optimal)} thresholds fell within {tolerance:.0%} of the minimum "
            f"cost; the most precise was chosen."
            if len(near_optimal) > 1
            else ""
        )
        rationale = (
            f"Minimised expected cost (FN={cost_fn:.0f}, FP={cost_fp:.0f} per event) "
            f"at {best['expected_cost']:.0f}, subject to precision >= {min_precision:.2f}"
            f"{'' if guardrail_applied else ' (guardrail not satisfiable, relaxed)'}. "
            f"Catches {int(best['true_positives'])} fraud, misses "
            f"{int(best['false_negatives'])}, raises {int(best['alerts'])} alerts." + tie_note
        )

    choice = ThresholdChoice(
        threshold=float(best["threshold"]),
        objective=objective,
        rationale=rationale,
        sweep=sweep,
        selected_row=best.to_dict(),
    )
    logger.info("Selected threshold %.3f | %s", choice.threshold, rationale)
    return choice
