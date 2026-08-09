"""Model serving layer.

Loads the registered artifact bundle once at process start and reuses it for
every request. The SAME preprocessor object that was fitted during training is
used at inference time, which is what keeps training/serving skew out of the
system.

The service returns raw scoring output; risk banding happens in the risk engine
and report writing happens in the agent, so each concern stays testable alone.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fraudshield.data.schema import FEATURE_COLUMNS
from fraudshield.explainability.shap_explainer import ShapExplainer, describe_contributions
from fraudshield.logging_utils import get_logger
from fraudshield.models.registry import LoadedArtifact, load_artifact

logger = get_logger(__name__)


@dataclass
class ScoringResult:
    """Model output for a single transaction, before risk banding."""

    probability: float
    is_fraud: bool
    threshold: float
    contributions: list[dict[str, Any]]
    explanation_text: str
    warnings: list[str]


class ModelService:
    """Thread-safe holder for the loaded artifact and its SHAP explainer."""

    def __init__(self, model_dir: Path, shap_top_k: int = 5) -> None:
        self.model_dir = Path(model_dir)
        self.shap_top_k = shap_top_k
        self._lock = threading.Lock()
        self._artifact: LoadedArtifact | None = None
        self._explainer: ShapExplainer | None = None
        self._explainer_error: str | None = None
        self._load_error: str | None = None

    # ---------------------------------------------------------------- loading
    def load(self) -> bool:
        """Load the artifact bundle. Returns False (without raising) on failure.

        A missing model must degrade to an unhealthy service, not a crash loop:
        /health should be able to report *why* the service cannot score.
        """
        with self._lock:
            try:
                self._artifact = load_artifact(self.model_dir)
                self._load_error = None
            except Exception as exc:
                self._artifact = None
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.error("Model load failed from %s: %s", self.model_dir, self._load_error)
                return False
            self._init_explainer()
            return True

    def _init_explainer(self) -> None:
        assert self._artifact is not None
        background_path = self.model_dir / "shap_background.npy"
        background = np.load(background_path) if background_path.is_file() else None
        try:
            self._explainer = ShapExplainer(
                model=self._artifact.model,
                feature_names=self._artifact.metadata.features,
                background=background,
            )
            self._explainer_error = None
        except Exception as exc:
            self._explainer = None
            self._explainer_error = f"SHAP unavailable ({type(exc).__name__}: {exc})"
            logger.warning(self._explainer_error)

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def metadata(self):
        if self._artifact is None:
            raise RuntimeError("Model is not loaded.")
        return self._artifact.metadata

    # --------------------------------------------------------------- scoring
    @staticmethod
    def to_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a model-input frame, filling absent feature columns with 0.0."""
        frame = pd.DataFrame(payloads)
        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        return frame[FEATURE_COLUMNS].astype(float)

    def predict(self, payloads: list[dict[str, Any]], explain: bool = True) -> list[ScoringResult]:
        """Score one or more transactions and attach SHAP attributions."""
        if self._artifact is None:
            raise RuntimeError("Model is not loaded; cannot score transactions.")

        frame = self.to_frame(payloads)
        transformed = np.asarray(self._artifact.preprocessor.transform(frame), dtype=float)
        probabilities = np.asarray(self._artifact.model.predict_proba(transformed))[:, 1]
        threshold = float(self._artifact.metadata.threshold)

        base_warnings: list[str] = []
        if self._artifact.metadata.is_synthetic:
            base_warnings.append(
                "Model was trained on SYNTHETIC smoke data; predictions are not meaningful."
            )

        explanations: list[list[dict[str, Any]]] = [[] for _ in range(len(frame))]
        if explain and self._explainer is not None:
            try:
                computed = self._explainer.explain(transformed, top_k=self.shap_top_k)
                explanations = [[c.to_dict() for c in row] for row in computed]
            except Exception as exc:
                logger.warning("SHAP computation failed: %s", exc)
                base_warnings.append(f"Explanation unavailable ({type(exc).__name__}).")
        elif explain and self._explainer_error:
            base_warnings.append(self._explainer_error)

        results = []
        for index, probability in enumerate(probabilities):
            probability = float(np.clip(probability, 0.0, 1.0))
            contributions = explanations[index]
            text = (
                describe_contributions(_rehydrate(contributions), probability)
                if contributions
                else f"Fraud probability: {probability:.1%}. No feature attributions available."
            )
            results.append(
                ScoringResult(
                    probability=probability,
                    is_fraud=bool(probability >= threshold),
                    threshold=threshold,
                    contributions=contributions,
                    explanation_text=text,
                    warnings=list(base_warnings),
                )
            )
        return results


def _rehydrate(contributions: list[dict[str, Any]]):
    """Turn serialised contributions back into objects for the text renderer."""
    from fraudshield.explainability.shap_explainer import FeatureContribution

    return [FeatureContribution(**c) for c in contributions]


_service: ModelService | None = None


def init_model_service(model_dir: Path, shap_top_k: int = 5) -> ModelService:
    """Create and load the process-wide model service."""
    global _service
    _service = ModelService(model_dir, shap_top_k=shap_top_k)
    _service.load()
    return _service


def get_model_service() -> ModelService:
    if _service is None:
        raise RuntimeError("Model service has not been initialised.")
    return _service
