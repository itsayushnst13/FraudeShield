"""FastAPI application entrypoint for FraudShield AI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fraudshield.logging_utils import get_logger, setup_logging

from .api.routes import router
from .core.config import get_settings
from .services.model_service import init_model_service

logger = get_logger(__name__)

DESCRIPTION = """
Credit-card fraud detection and investigation API.

**Pipeline:** transaction -> preprocessing -> ML model -> probability -> deterministic
risk band -> SHAP attributions -> guarded LLM investigation report.

The ML model owns the fraud decision. The LLM only narrates the evidence and can
never alter the probability, the risk band or the recommended action.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup rather than per request."""
    settings = get_settings()
    setup_logging(settings.log_level)
    service = init_model_service(settings.model_path, shap_top_k=settings.shap_top_k)
    if service.is_loaded:
        logger.info(
            "Model ready: %s v%s (%s), threshold %.3f",
            service.metadata.model_name,
            service.metadata.version,
            service.metadata.model_type,
            service.metadata.threshold,
        )
        if service.metadata.is_synthetic:
            logger.warning(
                "Serving a model trained on SYNTHETIC data - predictions are not meaningful."
            )
    else:
        logger.warning("Started WITHOUT a model; scoring endpoints will return 503.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Application factory - keeps tests free of import-time side effects."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
