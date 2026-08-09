"""Evaluation metrics tuned for extreme class imbalance.

Accuracy is deliberately computed but never used for selection: predicting
"legitimate" for every transaction scores 99.83% accuracy on this dataset while
catching zero fraud. PR-AUC (average precision) is the headline metric because
it summarises the precision/recall trade-off across all thresholds and, unlike
ROC-AUC, is not dominated by the vast true-negative pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    """Threshold-free and threshold-dependent metrics for one model."""

    model: str
    threshold: float
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    brier: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    n_samples: int
    n_positive: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def confusion(self) -> list[list[int]]:
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_row(self) -> dict[str, Any]:
        """Compact row for the model comparison table."""
        return {
            "model": self.model,
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model",
) -> ClassificationMetrics:
    """Compute the full metric set for predicted probabilities at a threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    n_positive = int(y_true.sum())
    both_classes = 0 < n_positive < len(y_true)
    pr_auc = float(average_precision_score(y_true, y_prob)) if both_classes else 0.0
    roc_auc = float(roc_auc_score(y_true, y_prob)) if both_classes else 0.5

    return ClassificationMetrics(
        model=model_name,
        threshold=float(threshold),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=_safe_divide(tp + tn, len(y_true)),
        brier=float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        n_samples=int(len(y_true)),
        n_positive=n_positive,
    )


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray, max_points: int = 200) -> dict:
    """Downsampled precision-recall curve, JSON-serialisable for the frontend."""
    precision, recall, _ = precision_recall_curve(np.asarray(y_true).astype(int), y_prob)
    idx = np.linspace(0, len(precision) - 1, min(max_points, len(precision))).astype(int)
    return {"precision": precision[idx].tolist(), "recall": recall[idx].tolist()}


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray, max_points: int = 200) -> dict:
    """Downsampled ROC curve, JSON-serialisable for the frontend."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
