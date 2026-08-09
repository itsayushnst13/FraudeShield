"""Model artifact persistence and versioning.

A deployable artifact is the *bundle*, not just the estimator: the fitted
preprocessor, the estimator, the selected decision threshold and the metadata
describing how it was produced. Serving the estimator without the exact
preprocessor that was fitted alongside it is a classic training/serving skew bug,
so the two are always written and loaded together.

Layout on disk::

    models/fraud_detector/
      metadata.json
      preprocessor.joblib
      model.joblib        (sklearn / xgboost / lightgbm)
      model.pt            (torch MLP or autoencoder)
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ..logging_utils import get_logger

logger = get_logger(__name__)

TORCH_TYPES = {"torch_mlp", "autoencoder"}
METADATA_FILE = "metadata.json"
PREPROCESSOR_FILE = "preprocessor.joblib"


@dataclass
class ModelMetadata:
    """Everything needed to identify, audit and reproduce a deployed model."""

    model_name: str
    model_type: str
    version: str
    threshold: float
    features: list[str]
    training_date: str = ""
    data_source: str = "UNKNOWN"
    n_train_rows: int = 0
    n_train_fraud: int = 0
    imbalance_strategy: str = "none"
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    threshold_rationale: str = ""
    selection_rationale: str = ""
    random_seed: int = 42
    python_version: str = field(default_factory=platform.python_version)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.training_date:
            self.training_date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def is_synthetic(self) -> bool:
        """True when the artifact was trained on the synthetic smoke fixture."""
        return "SYNTHETIC" in str(self.data_source).upper()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["is_synthetic"] = self.is_synthetic
        return payload


def save_artifact(
    directory: str | Path,
    preprocessor: Any,
    model: Any,
    metadata: ModelMetadata,
) -> Path:
    """Write preprocessor, estimator and metadata into a versioned bundle directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, directory / PREPROCESSOR_FILE)
    if metadata.model_type in TORCH_TYPES:
        model.save(directory / "model.pt")
    else:
        joblib.dump(model, directory / "model.joblib")

    (directory / METADATA_FILE).write_text(
        json.dumps(metadata.to_dict(), indent=2), encoding="utf-8"
    )
    logger.info("Saved artifact '%s' v%s -> %s", metadata.model_name, metadata.version, directory)
    return directory


@dataclass
class LoadedArtifact:
    """A ready-to-serve bundle."""

    preprocessor: Any
    model: Any
    metadata: ModelMetadata

    def predict_proba(self, X) -> Any:
        """Preprocess raw transaction columns and return fraud probabilities."""
        transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(transformed)[:, 1]


def load_metadata(directory: str | Path) -> ModelMetadata:
    """Read metadata.json, ignoring derived keys that are not constructor fields."""
    path = Path(directory) / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"No model metadata at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    valid = set(ModelMetadata.__dataclass_fields__.keys())
    return ModelMetadata(**{k: v for k, v in raw.items() if k in valid})


def load_artifact(directory: str | Path) -> LoadedArtifact:
    """Load a bundle previously written by :func:`save_artifact`."""
    directory = Path(directory)
    metadata = load_metadata(directory)
    preprocessor = joblib.load(directory / PREPROCESSOR_FILE)

    if metadata.model_type == "torch_mlp":
        from .mlp import TorchMLPClassifier

        model = TorchMLPClassifier.load(directory / "model.pt")
    elif metadata.model_type == "autoencoder":
        from .autoencoder import AutoencoderAnomalyDetector

        model = AutoencoderAnomalyDetector.load(directory / "model.pt")
    else:
        model = joblib.load(directory / "model.joblib")

    logger.info(
        "Loaded model '%s' v%s (%s)", metadata.model_name, metadata.version, metadata.model_type
    )
    return LoadedArtifact(preprocessor=preprocessor, model=model, metadata=metadata)


def default_model_dir(repo_root: Path, model_name: str = "fraud_detector") -> Path:
    return repo_root / "models" / model_name
