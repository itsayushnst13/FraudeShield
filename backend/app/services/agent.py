"""GenAI fraud investigation agent.

Position in the architecture::

    Transaction -> ML model -> probability -> SHAP -> Investigation Agent -> report

The LLM is a *writer*, not a *decider*. It receives an evidence pack assembled by
the whitelisted toolkit and turns it into an analyst-readable narrative. It
cannot change the fraud probability, the risk band or the recommended action -
those are copied verbatim from the ML model and the deterministic risk engine
into the response, outside anything the model can influence.

Three guardrails:

1. The system prompt forbids inventing evidence and mandates the exact phrase
   "Insufficient evidence." when a tool reported nothing.
2. :func:`_validate_narrative` rejects narratives that contradict the ML verdict,
   and the deterministic report is used instead.
3. Authoritative fields are never parsed back out of LLM text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fraudshield.logging_utils import get_logger

from .agent_tools import InvestigationToolkit
from .llm import LLMClient

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a fraud investigation analyst assistant working inside a \
payments risk team.

A machine-learning model has ALREADY scored the transaction. Your job is to explain \
its decision to a human reviewer, not to re-decide it.

Hard rules:
1. The ML fraud probability and risk level are authoritative. Never contradict, \
override, re-score or second-guess them. Do not state that a flagged transaction is \
safe, or that a cleared transaction is fraudulent.
2. Use ONLY the evidence supplied below. Never invent transaction history, merchant \
names, locations, device data, cardholder behaviour or statistics.
3. If a piece of evidence is marked unavailable, write exactly: Insufficient evidence.
4. SHAP features (V1-V28) are anonymised PCA components. Describe their direction and \
magnitude of contribution. Do not claim to know what they represent semantically.
5. Be concise and factual. No filler, no speculation, no reassurance.

Respond in valid JSON with exactly these keys:
{
  "risk_summary": "2-3 sentences stating the verdict and headline evidence",
  "why_flagged": "what drove the score, grounded in the SHAP attributions",
  "key_evidence": ["bullet", "points", "from supplied evidence only"],
  "recommended_action": "restate the action mandated by the risk policy",
  "confidence_and_limitations": "what is uncertain or missing from the evidence"
}
Output the JSON object only, with no surrounding text or code fences."""

CONTRADICTION_PATTERNS = {
    "FRAUD": [
        "not fraudulent",
        "not fraud",
        "is legitimate",
        "appears legitimate",
        "no fraud",
        "safe transaction",
        "no action needed",
        "no action required",
    ],
    "LEGITIMATE": ["is fraudulent", "confirmed fraud", "this is fraud"],
}


@dataclass
class InvestigationReport:
    """Structured investigation output returned to the API layer."""

    risk_summary: str
    why_flagged: str
    key_evidence: list[str]
    recommended_action: str
    confidence_and_limitations: str
    generated_by: str
    llm_provider: str
    llm_model: str
    tools_called: list[str] = field(default_factory=list)
    unavailable_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_summary": self.risk_summary,
            "why_flagged": self.why_flagged,
            "key_evidence": self.key_evidence,
            "recommended_action": self.recommended_action,
            "confidence_and_limitations": self.confidence_and_limitations,
            "generated_by": self.generated_by,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tools_called": self.tools_called,
            "unavailable_evidence": self.unavailable_evidence,
            "warnings": self.warnings,
        }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a model response, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_narrative(payload: dict[str, Any], verdict: str) -> list[str]:
    """Return the reasons a narrative must be rejected, if any."""
    problems: list[str] = []
    required = {
        "risk_summary",
        "why_flagged",
        "key_evidence",
        "recommended_action",
        "confidence_and_limitations",
    }
    missing = required - set(payload)
    if missing:
        problems.append(f"Missing keys: {sorted(missing)}")

    blob = " ".join(
        str(v).lower() if not isinstance(v, list) else " ".join(map(str, v)).lower()
        for v in payload.values()
    )
    for phrase in CONTRADICTION_PATTERNS.get(verdict, []):
        if phrase in blob:
            problems.append(f"Narrative contradicts the ML verdict ({verdict}): '{phrase}'")
            break
    return problems


def build_deterministic_report(
    scoring: dict[str, Any], risk: dict[str, Any], evidence: dict[str, Any], reason: str
) -> InvestigationReport:
    """Assemble a report from evidence alone, with no LLM involved.

    Used when no provider is configured or the narrative fails validation. Every
    sentence is derived from the supplied numbers, so this path can never
    hallucinate - it is simply less fluent.
    """
    probability = float(scoring.get("fraud_probability", 0.0))
    verdict = str(scoring.get("prediction", "UNKNOWN"))
    features = scoring.get("explanation", []) or []

    drivers = ", ".join(
        f"{c['feature']} ({'+' if c['shap_value'] > 0 else ''}{c['shap_value']:.4f})"
        for c in features[:3]
    )
    why = (
        f"The strongest attributions were {drivers}."
        if drivers
        else "Insufficient evidence. SHAP attributions were not available."
    )

    bullets = [
        f"Model fraud probability: {probability:.1%} (decision threshold "
        f"{scoring.get('threshold')})",
        f"Risk band: {risk.get('risk_level')} -> {risk.get('recommended_action')}",
        f"Scored by: {scoring.get('model_name')} v{scoring.get('model_version')} "
        f"({scoring.get('model_type')})",
    ]
    for item in evidence.get("evidence", []):
        if item["tool"] == "get_customer_transaction_history":
            bullets.append(
                f"Account history: {item['data'].get('n_transactions')} prior transactions, "
                f"mean amount {item['data'].get('mean_amount')}"
                if item["available"]
                else f"Account history: Insufficient evidence. {item.get('note', '')}".strip()
            )

    return InvestigationReport(
        risk_summary=(
            f"The model classified this transaction as {verdict} with a fraud probability of "
            f"{probability:.1%}, placing it in the {risk.get('risk_level')} risk band."
        ),
        why_flagged=why,
        key_evidence=bullets,
        recommended_action=str(risk.get("recommended_action", "REVIEW")),
        confidence_and_limitations=(
            "This report was generated deterministically from model output without an LLM "
            f"({reason}). V1-V28 are anonymised PCA components, so attributions indicate "
            "direction and magnitude only, not business meaning."
        ),
        generated_by="deterministic_fallback",
        llm_provider="none",
        llm_model="none",
        tools_called=evidence.get("tools_called", []),
        unavailable_evidence=evidence.get("unavailable", []),
        warnings=[f"LLM narrative not used: {reason}"],
    )


class InvestigationAgent:
    """Orchestrates evidence collection, LLM narration and validation."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def investigate(
        self,
        toolkit: InvestigationToolkit,
        scoring: dict[str, Any],
        risk: dict[str, Any],
    ) -> InvestigationReport:
        """Produce an investigation report for one scored transaction."""
        evidence = toolkit.collect_evidence()
        verdict = str(scoring.get("prediction", "UNKNOWN"))

        if not getattr(self.llm, "available", False):
            return build_deterministic_report(scoring, risk, evidence, "no LLM provider configured")

        user_prompt = (
            "EVIDENCE PACK (the only information you may use):\n"
            + json.dumps(evidence, indent=2, default=str)
            + "\n\nAUTHORITATIVE ML DECISION (do not alter):\n"
            + json.dumps({"scoring": scoring, "risk": risk}, indent=2, default=str)
        )
        response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        if not response.ok or not response.text:
            return build_deterministic_report(
                scoring, risk, evidence, response.error or "empty LLM response"
            )

        payload = _parse_json_object(response.text)
        if payload is None:
            return build_deterministic_report(
                scoring, risk, evidence, "LLM response was not valid JSON"
            )

        problems = _validate_narrative(payload, verdict)
        if problems:
            logger.warning("Rejected LLM narrative: %s", problems)
            return build_deterministic_report(scoring, risk, evidence, "; ".join(problems))

        key_evidence = payload.get("key_evidence", [])
        if isinstance(key_evidence, str):
            key_evidence = [key_evidence]

        return InvestigationReport(
            risk_summary=str(payload["risk_summary"]),
            why_flagged=str(payload["why_flagged"]),
            key_evidence=[str(item) for item in key_evidence],
            # Restated from the risk engine, never taken from the LLM.
            recommended_action=str(risk.get("recommended_action", "REVIEW")),
            confidence_and_limitations=str(payload["confidence_and_limitations"]),
            generated_by="llm",
            llm_provider=response.provider,
            llm_model=response.model,
            tools_called=evidence.get("tools_called", []),
            unavailable_evidence=evidence.get("unavailable", []),
        )
