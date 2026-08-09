"""Tests for model training, deep-learning components, SHAP and the registry."""

from __future__ import annotations

import numpy as np
import pytest

from fraudshield.data.splits import make_splits
from fraudshield.explainability.shap_explainer import ShapExplainer, describe_contributions
from fraudshield.features.preprocess import fit_preprocessor, get_output_feature_names
from fraudshield.models.autoencoder import AutoencoderAnomalyDetector
from fraudshield.models.baselines import build_dummy, build_logistic_regression
from fraudshield.models.mlp import TorchMLPClassifier, build_mlp
from fraudshield.models.registry import ModelMetadata, load_artifact, save_artifact
from fraudshield.models.trees import build_random_forest, build_xgboost


@pytest.fixture(scope="module")
def prepared(frame_module):
    s = make_splits(frame_module)
    pre = fit_preprocessor(s.X_train)
    return {
        "pre": pre,
        "names": get_output_feature_names(pre),
        "X_train": np.asarray(pre.transform(s.X_train), dtype=float),
        "X_val": np.asarray(pre.transform(s.X_val), dtype=float),
        "y_train": s.y_train.to_numpy(),
        "y_val": s.y_val.to_numpy(),
        "X_train_raw": s.X_train,
    }


@pytest.fixture(scope="module")
def frame_module():
    from fraudshield.data.synthetic import generate_synthetic_frame

    return generate_synthetic_frame(n_rows=3000, fraud_rate=0.03, seed=7)


def test_dummy_classifier_never_flags_fraud(prepared):
    model = build_dummy().fit(prepared["X_train"], prepared["y_train"])
    assert model.predict(prepared["X_val"]).sum() == 0


def test_logistic_regression_beats_random_ranking(prepared):
    from sklearn.metrics import roc_auc_score

    model = build_logistic_regression().fit(prepared["X_train"], prepared["y_train"])
    probs = model.predict_proba(prepared["X_val"])[:, 1]
    assert roc_auc_score(prepared["y_val"], probs) > 0.7


def test_xgboost_produces_valid_probabilities(prepared):
    model = build_xgboost(scale_pos_weight=10.0).fit(prepared["X_train"], prepared["y_train"])
    probs = model.predict_proba(prepared["X_val"])[:, 1]
    assert probs.shape == prepared["y_val"].shape
    assert ((probs >= 0) & (probs <= 1)).all()


def test_mlp_architecture_has_expected_block_structure():
    import torch.nn as nn

    net = build_mlp(input_dim=31, hidden_dims=[16, 8], dropout=0.2)
    kinds = [type(layer) for layer in net]
    assert kinds[:4] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout]
    assert kinds[-1] is nn.Linear
    assert net[-1].out_features == 1


def test_mlp_trains_records_history_and_predicts(prepared):
    model = TorchMLPClassifier(
        input_dim=prepared["X_train"].shape[1],
        hidden_dims=[16, 8],
        max_epochs=4,
        batch_size=256,
        early_stopping_patience=2,
    )
    model.fit(prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"])
    assert len(model.history.train_loss) == len(model.history.val_pr_auc) >= 1
    probs = model.predict_proba(prepared["X_val"])
    assert probs.shape == (len(prepared["y_val"]), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_mlp_early_stopping_respects_patience(prepared):
    model = TorchMLPClassifier(
        input_dim=prepared["X_train"].shape[1],
        hidden_dims=[8],
        max_epochs=50,
        batch_size=256,
        early_stopping_patience=1,
    )
    model.fit(prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"])
    assert len(model.history.train_loss) < 50


def test_mlp_roundtrips_through_disk(prepared, tmp_path):
    model = TorchMLPClassifier(
        input_dim=prepared["X_train"].shape[1],
        hidden_dims=[8],
        max_epochs=2,
        batch_size=256,
    )
    model.fit(prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"])
    before = model.predict_proba(prepared["X_val"])[:, 1]
    reloaded = TorchMLPClassifier.load(model.save(tmp_path / "mlp.pt"))
    np.testing.assert_allclose(before, reloaded.predict_proba(prepared["X_val"])[:, 1], atol=1e-5)


def test_autoencoder_reconstructs_legitimate_better_than_fraud(prepared):
    y_train, y_val = prepared["y_train"], prepared["y_val"]
    legit = prepared["X_train"][y_train == 0]
    ae = AutoencoderAnomalyDetector(
        input_dim=legit.shape[1],
        encoder_dims=[16],
        latent_dim=8,
        max_epochs=25,
        batch_size=256,
        early_stopping_patience=5,
    )
    ae.fit(legit[: int(0.9 * len(legit))], legit[int(0.9 * len(legit)) :])
    errors = ae.reconstruction_error(prepared["X_val"])
    assert errors[y_val == 1].mean() > errors[y_val == 0].mean()


def test_autoencoder_threshold_comes_from_validation_percentile(prepared):
    ae = AutoencoderAnomalyDetector(
        input_dim=prepared["X_train"].shape[1],
        encoder_dims=[16],
        latent_dim=8,
        max_epochs=3,
        batch_size=256,
    )
    legit = prepared["X_train"][prepared["y_train"] == 0]
    ae.fit(legit[:-50], legit[-50:])
    threshold = ae.calibrate(prepared["X_val"], percentile=95.0)
    errors = ae.reconstruction_error(prepared["X_val"])
    assert (errors > threshold).mean() == pytest.approx(0.05, abs=0.02)


def test_shap_values_are_real_and_ranked(prepared):
    model = build_random_forest().fit(prepared["X_train"], prepared["y_train"])
    explainer = ShapExplainer(model, prepared["names"])
    contributions = explainer.explain_one(prepared["X_val"][0], top_k=5)
    assert len(contributions) == 5
    impacts = [c.impact for c in contributions]
    assert impacts == sorted(impacts, reverse=True)
    assert any(c.shap_value != 0 for c in contributions)
    assert all(
        c.direction in ("increases_fraud_risk", "decreases_fraud_risk") for c in contributions
    )


def test_shap_global_importance_returns_ranked_features(prepared):
    model = build_random_forest().fit(prepared["X_train"], prepared["y_train"])
    top = ShapExplainer(model, prepared["names"]).global_importance(prepared["X_val"][:50], top_k=5)
    assert len(top) == 5
    assert top[0]["mean_abs_shap"] >= top[-1]["mean_abs_shap"]


def test_explanation_text_reflects_computed_values(prepared):
    model = build_random_forest().fit(prepared["X_train"], prepared["y_train"])
    contributions = ShapExplainer(model, prepared["names"]).explain_one(prepared["X_val"][0])
    text = describe_contributions(contributions, 0.87)
    assert "87.0%" in text
    assert contributions[0].feature in text


def test_explanation_text_handles_absent_attributions():
    assert "No feature attributions" in describe_contributions([], 0.5)


def test_artifact_roundtrip_preserves_predictions_and_metadata(prepared, tmp_path):
    model = build_random_forest().fit(prepared["X_train"], prepared["y_train"])
    metadata = ModelMetadata(
        model_name="test_model",
        model_type="random_forest",
        version="0.1.0",
        threshold=0.35,
        features=prepared["names"],
        data_source="SYNTHETIC_SMOKE",
    )
    save_artifact(tmp_path / "bundle", prepared["pre"], model, metadata)
    loaded = load_artifact(tmp_path / "bundle")
    assert loaded.metadata.threshold == 0.35
    assert loaded.metadata.is_synthetic is True
    assert loaded.metadata.training_date
    np.testing.assert_allclose(
        loaded.predict_proba(prepared["X_train_raw"].head(20)),
        model.predict_proba(prepared["X_train"][:20])[:, 1],
    )


def test_metadata_flags_real_data_as_non_synthetic():
    metadata = ModelMetadata("m", "xgboost", "1.0.0", 0.5, ["a"], data_source="KAGGLE_CREDITCARD")
    assert metadata.is_synthetic is False


def test_loading_a_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope")


def test_lightgbm_ignores_scale_pos_weight_by_default():
    """Extreme positive weighting collapses LightGBM's leaf-wise growth here."""
    from fraudshield.models.trees import build_lightgbm

    model = build_lightgbm(scale_pos_weight=577.7)
    assert model is not None
    assert model.get_params()["scale_pos_weight"] is None


def test_lightgbm_is_regularised():
    """Omitting L2 was the root cause of a 0.43 PR-AUC collapse; guard against it."""
    from fraudshield.models.trees import build_lightgbm

    params = build_lightgbm().get_params()
    assert params["reg_lambda"] > 0
    assert params["min_child_samples"] >= 20
