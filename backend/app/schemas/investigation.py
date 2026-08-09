"""Schemas for the AI investigation endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .transaction import PredictionResponse, Transaction


class HistoricalTransaction(BaseModel):
    """A prior transaction supplied by the caller as investigation context."""

    model_config = ConfigDict(extra="allow")

    transaction_id: str | None = None
    Amount: float = Field(..., ge=0)
    Time: float = Field(default=0.0, ge=0)
    label: str | None = Field(default=None, description="Known outcome, if any.")


class InvestigationRequest(BaseModel):
    """Investigate one transaction, optionally with caller-supplied history."""

    transaction: Transaction  # type: ignore[valid-type]
    history: list[HistoricalTransaction] = Field(
        default_factory=list,
        description="Prior transactions for this customer. Absent history yields "
        "'Insufficient evidence.' rather than invented context.",
    )


class InvestigationReportOut(BaseModel):
    """LLM-written (or deterministically assembled) investigation narrative."""

    risk_summary: str
    why_flagged: str
    key_evidence: list[str]
    recommended_action: str
    confidence_and_limitations: str
    generated_by: str
    llm_provider: str
    llm_model: str
    tools_called: list[str]
    unavailable_evidence: list[str]
    warnings: list[str] = Field(default_factory=list)


class InvestigationResponse(BaseModel):
    """Full investigation payload: authoritative ML decision plus narrative."""

    prediction: PredictionResponse
    report: InvestigationReportOut
    evidence: list[dict[str, Any]]


class ServiceMetricsResponse(BaseModel):
    """Aggregated service counters plus stored offline evaluation metrics."""

    model_config = ConfigDict(protected_namespaces=())

    total_transactions_scored: int
    fraud_detected: int
    fraud_rate: float
    model_name: str
    model_version: str
    model_type: str
    is_synthetic: bool
    offline_metrics: dict[str, Any]
