"""API endpoint tests exercising real scoring against a real artifact."""

from __future__ import annotations

import pytest


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_health_is_degraded_without_a_model(unloaded_client):
    response = unloaded_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded" and body["model_loaded"] is False
    assert body["detail"]


def test_scoring_returns_503_without_a_model(unloaded_client, sample_rows):
    assert unloaded_client.post("/predict", json=sample_rows["legit"]).status_code == 503
    assert unloaded_client.get("/model-info").status_code == 503


def test_model_info_exposes_provenance(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "fraud_detector"
    assert body["model_type"] == "random_forest"
    assert body["n_features"] == len(body["features"]) == 31
    assert body["is_synthetic"] is True
    assert len(body["risk_policy"]) == 4


def test_predict_flags_a_fraudulent_transaction(client, sample_rows):
    body = client.post("/predict", json=sample_rows["fraud"]).json()
    assert body["prediction"] == "FRAUD"
    assert body["fraud_probability"] >= body["threshold"]
    assert body["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")


def test_predict_clears_a_legitimate_transaction(client, sample_rows):
    body = client.post("/predict", json=sample_rows["legit"]).json()
    assert body["prediction"] == "LEGITIMATE"
    assert body["fraud_probability"] < body["threshold"]


def test_prediction_agrees_with_the_threshold_rule(client, sample_rows):
    for row in sample_rows.values():
        body = client.post("/predict", json=row).json()
        expected = "FRAUD" if body["fraud_probability"] >= body["threshold"] else "LEGITIMATE"
        assert body["prediction"] == expected


def test_predict_returns_real_shap_attributions(client, sample_rows):
    body = client.post("/predict", json=sample_rows["fraud"]).json()
    explanation = body["explanation"]
    assert len(explanation) == 5
    impacts = [item["impact"] for item in explanation]
    assert impacts == sorted(impacts, reverse=True)
    assert any(item["shap_value"] != 0 for item in explanation)
    assert body["explanation_text"] and explanation[0]["feature"] in body["explanation_text"]


def test_synthetic_model_warns_on_every_prediction(client, sample_rows):
    body = client.post("/predict", json=sample_rows["legit"]).json()
    assert any("SYNTHETIC" in w for w in body["warnings"])


def test_transaction_id_is_echoed_back(client, sample_rows):
    payload = {**sample_rows["legit"], "transaction_id": "txn_abc"}
    assert client.post("/predict", json=payload).json()["transaction_id"] == "txn_abc"


@pytest.mark.parametrize(
    "payload",
    [
        {"Amount": -1.0},
        {"Amount": "not-a-number"},
        {},
        {"Amount": 10.0, "unexpected_field": 1},
    ],
)
def test_invalid_payloads_are_rejected_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_pca_components_default_to_zero(client):
    response = client.post("/predict", json={"Amount": 50.0, "Time": 1000.0})
    assert response.status_code == 200
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


def test_batch_predict_scores_every_row(client, sample_rows):
    rows = [sample_rows["fraud"], sample_rows["legit"], sample_rows["fraud"]]
    body = client.post("/batch-predict", json={"transactions": rows}).json()
    assert body["total"] == 3 and body["succeeded"] == 3 and body["failed"] == 0
    assert body["fraud_detected"] == 2
    assert [r["index"] for r in body["results"]] == [0, 1, 2]


def test_batch_predict_matches_single_predict(client, sample_rows):
    single = client.post("/predict", json=sample_rows["fraud"]).json()
    batch = client.post(
        "/batch-predict", json={"transactions": [sample_rows["fraud"]]}
    ).json()["results"][0]["prediction"]
    assert batch["fraud_probability"] == pytest.approx(single["fraud_probability"])


def test_batch_predict_rejects_an_empty_batch(client):
    assert client.post("/batch-predict", json={"transactions": []}).status_code == 422


def test_batch_predict_rejects_a_batch_with_an_invalid_row(client, sample_rows):
    rows = [sample_rows["legit"], {"Amount": -5}]
    assert client.post("/batch-predict", json={"transactions": rows}).status_code == 422


def test_batch_predict_enforces_the_size_limit(client, sample_rows, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "max_batch_size", 2, raising=False)
    rows = [sample_rows["legit"]] * 3
    assert client.post("/batch-predict", json={"transactions": rows}).status_code == 413


def test_metrics_counts_scored_transactions(client, sample_rows):
    before = client.get("/metrics").json()["total_transactions_scored"]
    client.post("/predict", json=sample_rows["legit"])
    after = client.get("/metrics").json()
    assert after["total_transactions_scored"] == before + 1
    assert 0.0 <= after["fraud_rate"] <= 1.0
    assert after["is_synthetic"] is True


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for endpoint in ("/health", "/model-info", "/predict", "/batch-predict",
                     "/investigate", "/metrics"):
        assert endpoint in paths
