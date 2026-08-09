"""Fixtures for backend tests.

A real (tiny) model artifact is trained once per session into a temp directory so
API tests exercise genuine scoring and genuine SHAP values rather than mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fraudshield.data.synthetic import generate_synthetic_frame  # noqa: E402
from fraudshield.features.preprocess import fit_preprocessor, get_output_feature_names  # noqa: E402
from fraudshield.models.registry import ModelMetadata, save_artifact  # noqa: E402
from fraudshield.models.trees import build_random_forest  # noqa: E402


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("model") / "fraud_detector"
    frame = generate_synthetic_frame(n_rows=2500, fraud_rate=0.04, seed=3)
    X, y = frame.drop(columns=["Class"]), frame["Class"].to_numpy()

    preprocessor = fit_preprocessor(X)
    transformed = np.asarray(preprocessor.transform(X), dtype=float)
    model = build_random_forest().fit(transformed, y)

    save_artifact(
        directory, preprocessor, model,
        ModelMetadata(
            model_name="fraud_detector", model_type="random_forest", version="0.0.1-test",
            threshold=0.30, features=get_output_feature_names(preprocessor),
            data_source="SYNTHETIC_SMOKE", n_train_rows=len(y), n_train_fraud=int(y.sum()),
            imbalance_strategy="class_weight",
            validation_metrics={"pr_auc": 0.9, "recall": 0.9},
            test_metrics={"pr_auc": 0.9, "recall": 0.9},
        ),
    )
    np.save(directory / "shap_background.npy", transformed[:100])
    return directory


@pytest.fixture(scope="session")
def sample_rows():
    frame = generate_synthetic_frame(n_rows=600, fraud_rate=0.1, seed=5)
    fraud = frame[frame.Class == 1].head(1).drop(columns=["Class"]).to_dict("records")[0]
    legit = frame[frame.Class == 0].head(1).drop(columns=["Class"]).to_dict("records")[0]
    return {"fraud": fraud, "legit": legit}


@pytest.fixture
def client(artifact_dir, monkeypatch):
    """TestClient wired to the temp artifact, with settings cache cleared."""
    from app.core.config import get_settings

    monkeypatch.setenv("MODEL_DIR", str(artifact_dir))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()

    from app.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def unloaded_client(monkeypatch, tmp_path):
    """TestClient pointed at a directory with no model, to test 503 behaviour."""
    from app.core.config import get_settings

    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()

    from app.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
