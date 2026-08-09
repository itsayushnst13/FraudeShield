"""HTTP API routes.

Each route is thin: validate, delegate to a service, shape the response. The
scoring, risk and narration logic all live in ``app/services`` and
``app/core`` so they can be tested without an HTTP client.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from ..core.config import get_settings
from ..core.risk_engine import get_risk_engine
from ..core.state import counters, load_offline_metrics
from ..schemas.investigation import (
    InvestigationReportOut,
    InvestigationRequest,
    InvestigationResponse,
    ServiceMetricsResponse,
)
from ..schemas.transaction import (
    BatchItemResult,
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    Transaction,
)
from ..services.agent import InvestigationAgent
from ..services.agent_tools import InvestigationToolkit, TransactionHistoryStore
from ..services.llm import build_llm_client
from ..services.model_service import ScoringResult, get_model_service

router = APIRouter()


def _require_model():
    """Return the loaded model service or fail with 503."""
    service = get_model_service()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. Train and register one first "
                f"({service.load_error or 'artifact missing'})."
            ),
        )
    return service


def _to_response(
    result: ScoringResult, metadata: Any, transaction_id: str | None
) -> PredictionResponse:
    """Combine scoring output with the deterministic risk assessment."""
    assessment = get_risk_engine().assess(result.probability)
    return PredictionResponse(
        prediction="FRAUD" if result.is_fraud else "LEGITIMATE",
        fraud_probability=round(result.probability, 6),
        risk_level=assessment.risk_level,
        recommended_action=assessment.recommended_action,
        threshold=result.threshold,
        model_name=metadata.model_name,
        model_version=metadata.version,
        model_type=metadata.model_type,
        transaction_id=transaction_id,
        explanation=result.contributions,
        explanation_text=result.explanation_text,
        warnings=result.warnings,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe that also reports whether a model is ready to score."""
    service = get_model_service()
    settings = get_settings()
    loaded = service.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        api_version=settings.api_version,
        detail=None if loaded else (service.load_error or "No model artifact found."),
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    """Provenance of the deployed model, including how its threshold was chosen."""
    service = _require_model()
    meta = service.metadata
    return ModelInfoResponse(
        model_name=meta.model_name,
        model_type=meta.model_type,
        version=meta.version,
        threshold=meta.threshold,
        training_date=meta.training_date,
        data_source=meta.data_source,
        is_synthetic=meta.is_synthetic,
        n_features=len(meta.features),
        features=meta.features,
        n_train_rows=meta.n_train_rows,
        n_train_fraud=meta.n_train_fraud,
        imbalance_strategy=meta.imbalance_strategy,
        validation_metrics=meta.validation_metrics,
        test_metrics=meta.test_metrics,
        threshold_rationale=meta.threshold_rationale,
        selection_rationale=meta.selection_rationale,
        risk_policy=get_risk_engine().policy_description(),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(transaction: Transaction) -> PredictionResponse:  # type: ignore[valid-type]
    """Score a single transaction and return a SHAP-backed explanation."""
    service = _require_model()
    payload = transaction.model_dump()
    try:
        result = service.predict([payload])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    counters.record(1, int(result.is_fraud))
    return _to_response(result, service.metadata, payload.get("transaction_id"))


@router.post("/batch-predict", response_model=BatchPredictionResponse, tags=["scoring"])
def batch_predict(request: BatchTransactionRequest) -> BatchPredictionResponse:
    """Score many transactions, isolating per-row failures instead of failing the batch."""
    service = _require_model()
    settings = get_settings()
    if len(request.transactions) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch exceeds max size of {settings.max_batch_size}.",
        )

    payloads = [t.model_dump() for t in request.transactions]
    results: list[BatchItemResult] = []
    try:
        scored = service.predict(payloads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {exc}") from exc

    fraud_count = 0
    for index, (payload, result) in enumerate(zip(payloads, scored, strict=True)):
        try:
            response = _to_response(result, service.metadata, payload.get("transaction_id"))
            fraud_count += int(result.is_fraud)
            results.append(BatchItemResult(index=index, ok=True, prediction=response))
        except Exception as exc:
            results.append(BatchItemResult(index=index, ok=False, error=str(exc)))

    succeeded = sum(1 for r in results if r.ok)
    counters.record(succeeded, fraud_count)
    return BatchPredictionResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        fraud_detected=fraud_count,
        results=results,
    )


@router.post("/investigate", response_model=InvestigationResponse, tags=["investigation"])
def investigate(request: InvestigationRequest) -> InvestigationResponse:
    """Score a transaction, then generate a guarded investigation report."""
    service = _require_model()
    settings = get_settings()
    payload = request.transaction.model_dump()

    try:
        result = service.predict([payload])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    counters.record(1, int(result.is_fraud))
    prediction = _to_response(result, service.metadata, payload.get("transaction_id"))
    assessment = get_risk_engine().assess(result.probability)

    history_store = TransactionHistoryStore()
    customer_id = payload.get("customer_id")
    if customer_id:
        for item in request.history:
            history_store.add(str(customer_id), item.model_dump())

    toolkit = InvestigationToolkit(
        transaction=payload,
        scoring=prediction.model_dump(),
        risk={
            "risk_level": assessment.risk_level,
            "recommended_action": assessment.recommended_action,
            "band_floor": assessment.band_floor,
            "band_ceiling": assessment.band_ceiling,
        },
        risk_policy=get_risk_engine().policy_description(),
        model_info={
            "model_name": service.metadata.model_name,
            "model_type": service.metadata.model_type,
            "version": service.metadata.version,
            "data_source": service.metadata.data_source,
            "is_synthetic": service.metadata.is_synthetic,
        },
        history_store=history_store,
    )

    agent = InvestigationAgent(
        build_llm_client(
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
        )
    )
    report = agent.investigate(
        toolkit,
        scoring=prediction.model_dump(),
        risk={
            "risk_level": assessment.risk_level,
            "recommended_action": assessment.recommended_action,
        },
    )
    evidence = toolkit.collect_evidence()

    return InvestigationResponse(
        prediction=prediction,
        report=InvestigationReportOut(**report.to_dict()),
        evidence=evidence["evidence"],
    )


@router.get("/metrics", response_model=ServiceMetricsResponse, tags=["system"])
def metrics() -> ServiceMetricsResponse:
    """Live service counters plus the offline evaluation metrics of the model."""
    service = _require_model()
    settings = get_settings()
    snapshot = counters.snapshot()
    return ServiceMetricsResponse(
        **snapshot,
        model_name=service.metadata.model_name,
        model_version=service.metadata.version,
        model_type=service.metadata.model_type,
        is_synthetic=service.metadata.is_synthetic,
        offline_metrics=load_offline_metrics(settings.model_path),
    )
