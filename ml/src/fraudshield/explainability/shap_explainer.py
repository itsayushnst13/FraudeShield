"""SHAP explanations for individual fraud decisions.

Why SHAP rather than global feature importance: a fraud analyst reviewing a held
transaction needs to know why *this* transaction scored 0.94, not which features
matter on average. SHAP attributes the gap between the model's base rate and this
prediction across features, with contributions that sum to that gap - which is
exactly the evidence a reviewer (and, later, a regulator asking for adverse-action
reasoning) needs.

TreeSHAP is used for tree ensembles because it is exact and fast enough to run
inline in the request path. Non-tree models fall back to a sampling explainer
over a small background set.

Nothing here is templated: every number returned comes from a computed SHAP value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)

TREE_MODEL_HINTS = ("XGB", "LGBM", "RandomForest", "GradientBoosting", "DecisionTree")


@dataclass
class FeatureContribution:
    """One feature's signed contribution to a single prediction."""

    feature: str
    shap_value: float
    impact: float
    direction: str
    feature_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_tree_model(model: Any) -> bool:
    return any(hint in type(model).__name__ for hint in TREE_MODEL_HINTS)


class ShapExplainer:
    """Wraps a fitted model and produces per-prediction attributions."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background: np.ndarray | None = None,
        max_background: int = 100,
        seed: int = 42,
    ) -> None:
        import shap

        self.model = model
        self.feature_names = list(feature_names)
        self.explainer_type = "tree" if _is_tree_model(model) else "sampling"

        if self.explainer_type == "tree":
            self._explainer = shap.TreeExplainer(model)
        else:
            if background is None or len(background) == 0:
                raise ValueError("A background sample is required for non-tree explainers.")
            rng = np.random.default_rng(seed)
            idx = rng.choice(
                len(background), size=min(max_background, len(background)), replace=False
            )
            summary = shap.kmeans(np.asarray(background)[idx], min(10, len(idx)))
            self._explainer = shap.KernelExplainer(
                lambda data: np.asarray(model.predict_proba(data))[:, 1], summary
            )
        logger.info("SHAP explainer initialised (%s).", self.explainer_type)

    def _raw_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return an (n_samples, n_features) array of positive-class SHAP values."""
        values = self._explainer.shap_values(X)
        values = np.asarray(values)
        # Binary classifiers may return a list/stack per class; take the positive class.
        if values.ndim == 3:
            values = values[..., -1] if values.shape[-1] == 2 else values[-1]
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values

    def explain(self, X: np.ndarray, top_k: int = 5) -> list[list[FeatureContribution]]:
        """Compute ranked contributions for each row of preprocessed features."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        values = self._raw_shap_values(X)
        results: list[list[FeatureContribution]] = []

        for row_idx in range(X.shape[0]):
            row = values[row_idx]
            order = np.argsort(-np.abs(row))[:top_k]
            contributions = [
                FeatureContribution(
                    feature=self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                    shap_value=float(row[i]),
                    impact=float(abs(row[i])),
                    direction="increases_fraud_risk" if row[i] > 0 else "decreases_fraud_risk",
                    feature_value=float(X[row_idx, i]),
                )
                for i in order
            ]
            results.append(contributions)
        return results

    def explain_one(self, x: np.ndarray, top_k: int = 5) -> list[FeatureContribution]:
        """Convenience wrapper for a single transaction."""
        return self.explain(np.atleast_2d(x), top_k=top_k)[0]

    def global_importance(self, X: np.ndarray, top_k: int = 15) -> list[dict[str, Any]]:
        """Mean absolute SHAP value per feature over a sample - the global view."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean_abs = np.abs(self._raw_shap_values(X)).mean(axis=0)
        order = np.argsort(-mean_abs)[:top_k]
        return [
            {
                "feature": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                "mean_abs_shap": float(mean_abs[i]),
            }
            for i in order
        ]


def describe_contributions(contributions: list[FeatureContribution], probability: float) -> str:
    """Render computed SHAP values as an analyst-readable narrative.

    The wording is derived from the magnitudes present in ``contributions``; no
    explanation text is hardcoded per feature.
    """
    if not contributions:
        return f"Fraud probability {probability:.1%}. No feature attributions available."

    magnitudes = [c.impact for c in contributions]
    largest = max(magnitudes) or 1.0

    def strength(impact: float) -> str:
        ratio = impact / largest
        if ratio >= 0.66:
            return "strongly"
        if ratio >= 0.33:
            return "moderately"
        return "slightly"

    lines = [f"Fraud probability: {probability:.1%}", "", "Top contributing features:"]
    for c in contributions:
        verb = "increased" if c.shap_value > 0 else "decreased"
        lines.append(
            f"  {c.feature} (value {c.feature_value:+.3f}) -> {strength(c.impact)} "
            f"{verb} fraud probability (SHAP {c.shap_value:+.4f})"
        )
    return "\n".join(lines)
