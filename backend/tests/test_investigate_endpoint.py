"""Integration tests for POST /investigate."""

from __future__ import annotations


def payload(row, history=None):
    return {"transaction": {**row, "transaction_id": "t9", "customer_id": "c9"},
            "history": history or []}


def test_investigate_returns_prediction_report_and_evidence(client, sample_rows):
    body = client.post("/investigate", json=payload(sample_rows["fraud"])).json()
    assert set(body) == {"prediction", "report", "evidence"}
    assert body["prediction"]["prediction"] in ("FRAUD", "LEGITIMATE")
    assert len(body["evidence"]) == 6


def test_report_action_matches_the_risk_engine(client, sample_rows):
    body = client.post("/investigate", json=payload(sample_rows["fraud"])).json()
    assert body["report"]["recommended_action"] == body["prediction"]["recommended_action"]


def test_all_six_tools_are_called(client, sample_rows):
    report = client.post("/investigate", json=payload(sample_rows["legit"])).json()["report"]
    assert len(report["tools_called"]) == 6
    assert "get_shap_explanation" in report["tools_called"]


def test_absent_history_is_reported_as_unavailable(client, sample_rows):
    report = client.post("/investigate", json=payload(sample_rows["fraud"])).json()["report"]
    assert "get_customer_transaction_history" in report["unavailable_evidence"]


def test_supplied_history_is_used(client, sample_rows):
    history = [{"Amount": 40.0, "Time": 10.0}, {"Amount": 60.0, "Time": 20.0}]
    body = client.post("/investigate", json=payload(sample_rows["fraud"], history)).json()
    assert "get_customer_transaction_history" not in body["report"]["unavailable_evidence"]
    tool = next(e for e in body["evidence"] if e["tool"] == "get_customer_transaction_history")
    assert tool["data"]["n_transactions"] == 2 and tool["data"]["mean_amount"] == 50.0


def test_report_uses_the_deterministic_path_when_no_llm_is_configured(client, sample_rows):
    report = client.post("/investigate", json=payload(sample_rows["fraud"])).json()["report"]
    assert report["generated_by"] == "deterministic_fallback"
    assert report["llm_provider"] == "none"


def test_report_grounds_its_reasoning_in_shap_features(client, sample_rows):
    body = client.post("/investigate", json=payload(sample_rows["fraud"])).json()
    top_feature = body["prediction"]["explanation"][0]["feature"]
    assert top_feature in body["report"]["why_flagged"]


def test_investigate_rejects_an_invalid_transaction(client):
    assert client.post("/investigate", json={"transaction": {"Amount": -1}}).status_code == 422


def test_investigate_requires_a_transaction(client):
    assert client.post("/investigate", json={"history": []}).status_code == 422


def test_investigate_returns_503_without_a_model(unloaded_client, sample_rows):
    assert unloaded_client.post(
        "/investigate", json=payload(sample_rows["legit"])
    ).status_code == 503
