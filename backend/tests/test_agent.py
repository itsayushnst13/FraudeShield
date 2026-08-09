"""Tests for the investigation agent, its tool whitelist and its guardrails."""

from __future__ import annotations

import json

import pytest
from app.services.agent import (
    InvestigationAgent,
    _parse_json_object,
    _validate_narrative,
    build_deterministic_report,
)
from app.services.agent_tools import InvestigationToolkit, TransactionHistoryStore
from app.services.llm import LLMResponse, NullLLMClient, build_llm_client

SCORING = {
    "prediction": "FRAUD",
    "fraud_probability": 0.947,
    "threshold": 0.35,
    "model_name": "fraud_detector",
    "model_version": "1.0.0",
    "model_type": "xgboost",
    "explanation": [
        {"feature": "V14", "shap_value": 0.42, "impact": 0.42,
         "direction": "increases_fraud_risk", "feature_value": -3.1},
        {"feature": "V10", "shap_value": 0.31, "impact": 0.31,
         "direction": "increases_fraud_risk", "feature_value": -2.4},
    ],
}
RISK = {"risk_level": "CRITICAL", "recommended_action": "BLOCK_AND_ESCALATE"}
POLICY = [{"risk_level": "CRITICAL", "min_probability": 0.9, "action": "BLOCK_AND_ESCALATE"}]


class StubLLM:
    """Returns a canned payload so guardrails can be tested deterministically."""

    provider, model, available = "stub", "stub-1", True

    def __init__(self, payload, ok=True, error=None):
        self.payload = payload
        self.ok = ok
        self.error = error
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResponse(text=text, provider=self.provider, model=self.model,
                           ok=self.ok, error=self.error)


VALID_PAYLOAD = {
    "risk_summary": "Model scored this transaction as fraudulent with high confidence.",
    "why_flagged": "V14 and V10 contributed most strongly to the score.",
    "key_evidence": ["Probability 94.7%", "Risk band CRITICAL"],
    "recommended_action": "Block the transaction",
    "confidence_and_limitations": "PCA components are anonymised.",
}


def make_toolkit(transaction=None, history=None):
    transaction = transaction or {"transaction_id": "t1", "customer_id": "c1", "Amount": 250.0}
    store = TransactionHistoryStore()
    for row in history or []:
        store.add(str(transaction["customer_id"]), row)
    return InvestigationToolkit(
        transaction=transaction, scoring=SCORING, risk=RISK, risk_policy=POLICY,
        model_info={"model_name": "fraud_detector"}, history_store=store,
    )


# ------------------------------------------------------------------ tool layer
def test_toolkit_exposes_exactly_the_whitelisted_tools():
    assert set(make_toolkit().registry) == {
        "get_transaction_details", "get_customer_transaction_history",
        "get_model_prediction", "get_shap_explanation", "get_risk_policy", "get_model_info",
    }


def test_non_whitelisted_tool_calls_are_blocked():
    result = make_toolkit().invoke("read_database")
    assert result.available is False and result.note == "Tool not permitted."


def test_history_tool_reports_unavailable_without_records():
    result = make_toolkit().get_customer_transaction_history()
    assert result.available is False
    assert "No prior transactions" in result.note


def test_history_tool_reports_unavailable_without_customer_id():
    toolkit = make_toolkit(transaction={"transaction_id": "t1", "Amount": 10.0})
    assert toolkit.get_customer_transaction_history().available is False


def test_history_tool_summarises_available_records():
    toolkit = make_toolkit(history=[{"Amount": 100.0}, {"Amount": 200.0}])
    result = toolkit.get_customer_transaction_history()
    assert result.available is True
    assert result.data["n_transactions"] == 2
    assert result.data["mean_amount"] == 150.0


def test_transaction_details_do_not_leak_raw_pca_values():
    toolkit = make_toolkit(transaction={"transaction_id": "t1", "Amount": 5.0, "V1": 9.9})
    data = toolkit.get_transaction_details().data
    assert "V1" not in data and data["pca_components_provided"] == 1


def test_shap_tool_reports_unavailable_when_attributions_are_missing():
    toolkit = InvestigationToolkit({"Amount": 1.0}, {**SCORING, "explanation": []},
                                   RISK, POLICY, {})
    assert toolkit.get_shap_explanation().available is False


def test_collect_evidence_runs_every_tool_and_logs_calls():
    evidence = make_toolkit().collect_evidence()
    assert len(evidence["evidence"]) == 6
    assert len(evidence["tools_called"]) == 6
    assert "get_customer_transaction_history" in evidence["unavailable"]


# --------------------------------------------------------------- LLM selection
def test_no_provider_yields_a_null_client():
    assert isinstance(build_llm_client("none", "", ""), NullLLMClient)


def test_missing_api_key_yields_a_null_client():
    assert build_llm_client("anthropic", "", "claude").available is False


def test_unknown_provider_falls_back_to_null_client():
    assert isinstance(build_llm_client("mystery-vendor", "key", "m"), NullLLMClient)


def test_known_providers_are_constructed_with_their_key():
    for provider in ("anthropic", "openai"):
        client = build_llm_client(provider, "secret", "model-x")
        assert client.available is True and client.provider == provider


# ------------------------------------------------------------------- guardrails
def test_deterministic_report_is_used_without_an_llm():
    report = InvestigationAgent(NullLLMClient()).investigate(make_toolkit(), SCORING, RISK)
    assert report.generated_by == "deterministic_fallback"
    assert report.recommended_action == "BLOCK_AND_ESCALATE"
    assert "V14" in report.why_flagged


def test_deterministic_report_says_insufficient_evidence_for_missing_history():
    toolkit = make_toolkit()
    report = build_deterministic_report(SCORING, RISK, toolkit.collect_evidence(), "test")
    assert any("Insufficient evidence." in bullet for bullet in report.key_evidence)


def test_valid_llm_narrative_is_accepted():
    report = InvestigationAgent(StubLLM(VALID_PAYLOAD)).investigate(make_toolkit(), SCORING, RISK)
    assert report.generated_by == "llm"
    assert report.risk_summary == VALID_PAYLOAD["risk_summary"]


def test_llm_cannot_override_the_recommended_action():
    payload = {**VALID_PAYLOAD, "recommended_action": "ALLOW - this is definitely fine"}
    report = InvestigationAgent(StubLLM(payload)).investigate(make_toolkit(), SCORING, RISK)
    assert report.recommended_action == "BLOCK_AND_ESCALATE"


@pytest.mark.parametrize(
    "contradiction",
    ["This transaction is legitimate.", "No action needed here.", "There is no fraud present."],
)
def test_narratives_contradicting_the_ml_verdict_are_rejected(contradiction):
    payload = {**VALID_PAYLOAD, "risk_summary": contradiction}
    report = InvestigationAgent(StubLLM(payload)).investigate(make_toolkit(), SCORING, RISK)
    assert report.generated_by == "deterministic_fallback"
    assert report.warnings


def test_incomplete_narratives_are_rejected():
    report = InvestigationAgent(StubLLM({"risk_summary": "only this"})).investigate(
        make_toolkit(), SCORING, RISK
    )
    assert report.generated_by == "deterministic_fallback"


def test_malformed_json_falls_back_deterministically():
    report = InvestigationAgent(StubLLM("this is not json")).investigate(
        make_toolkit(), SCORING, RISK
    )
    assert report.generated_by == "deterministic_fallback"


def test_llm_transport_failure_falls_back_deterministically():
    stub = StubLLM(VALID_PAYLOAD, ok=False, error="timeout")
    report = InvestigationAgent(stub).investigate(make_toolkit(), SCORING, RISK)
    assert report.generated_by == "deterministic_fallback"
    assert "timeout" in report.warnings[0]


def test_evidence_pack_is_passed_to_the_llm_prompt():
    stub = StubLLM(VALID_PAYLOAD)
    InvestigationAgent(stub).investigate(make_toolkit(), SCORING, RISK)
    system_prompt, user_prompt = stub.calls[0]
    assert "Insufficient evidence." in system_prompt
    assert "never invent" in system_prompt.lower()
    assert "get_shap_explanation" in user_prompt


def test_json_parser_tolerates_code_fences():
    assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_object("preamble {\"a\": 2} trailing") == {"a": 2}
    assert _parse_json_object("no object here") is None


def test_validator_accepts_a_wellformed_payload():
    assert _validate_narrative(VALID_PAYLOAD, "FRAUD") == []


def test_validator_flags_fraud_claims_on_cleared_transactions():
    payload = {**VALID_PAYLOAD, "why_flagged": "This is confirmed fraud."}
    assert _validate_narrative(payload, "LEGITIMATE")
