"""Pydantic schemas for the transaction scoring API.

The 28 PCA fields are declared via ``create_model`` rather than 28 hand-written
lines. This still produces a fully-typed OpenAPI schema, but keeps the source
short and impossible to get out of sync with the dataset contract.
"""

from __future__ import annotations

from typing import Any

from fraudshield.data.schema import PCA_COLUMNS
from pydantic import BaseModel, ConfigDict, Field, create_model


class TransactionBase(BaseModel):
    """Non-PCA transaction attributes."""

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(
        default=0.0, ge=0, description="Seconds elapsed since the first transaction on record."
    )
    Amount: float = Field(..., ge=0, description="Transaction amount.")
    transaction_id: str | None = Field(
        default=None, description="Optional client-side identifier, echoed back in the response."
    )
    customer_id: str | None = Field(
        default=None, description="Optional customer identifier used by the investigation agent."
    )


_pca_fields: dict[str, Any] = {
    name: (float, Field(default=0.0, description=f"PCA component {name}.")) for name in PCA_COLUMNS
}

Transaction = create_model("Transaction", __base__=TransactionBase, **_pca_fields)
Transaction.__doc__ = "A single transaction: Time, Amount and PCA components V1..V28."


class BatchTransactionRequest(BaseModel):
    """Batch scoring payload."""

    transactions: list[Transaction] = Field(..., min_length=1)  # type: ignore[valid-type]


class FeatureContributionOut(BaseModel):
    """One SHAP attribution, as returned to clients."""

    feature: str
    shap_value: float
    impact: float
    direction: str
    feature_value: float


class PredictionResponse(BaseModel):
    """Scoring result for one transaction."""

    prediction: str = Field(..., description="FRAUD or LEGITIMATE.")
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    recommended_action: str
    threshold: float
    model_name: str
    model_version: str
    model_type: str
    transaction_id: str | None = None
    explanation: list[FeatureContributionOut] = Field(default_factory=list)
    explanation_text: str | None = None
    warnings: list[str] = Field(default_factory=list)


class BatchItemResult(BaseModel):
    """One row of a batch response; either a prediction or a per-row error."""

    index: int
    ok: bool
    prediction: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    """Batch scoring result with a per-row success/failure breakdown."""

    total: int
    succeeded: int
    failed: int
    fraud_detected: int
    results: list[BatchItemResult]


class HealthResponse(BaseModel):
    """Liveness and model-readiness probe."""

    status: str
    model_loaded: bool
    api_version: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Deployed model provenance, surfaced at GET /model-info."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_type: str
    version: str
    threshold: float
    training_date: str
    data_source: str
    is_synthetic: bool
    n_features: int
    features: list[str]
    n_train_rows: int
    n_train_fraud: int
    imbalance_strategy: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    threshold_rationale: str
    selection_rationale: str
    risk_policy: list[dict]
