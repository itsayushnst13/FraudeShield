"""Backend settings, sourced from environment variables.

Secrets (LLM API keys in particular) are read from the environment and never
committed. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "configs" / "config.yaml").is_file():
            return candidate
    return here.parents[3]


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI service."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    app_name: str = "FraudShield AI"
    api_version: str = "1.0.0"
    log_level: str = "INFO"

    repo_root: Path = _default_repo_root()
    model_dir: str = "models/fraud_detector"
    ml_config_path: str = "configs/config.yaml"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    max_batch_size: int = 1000
    shap_top_k: int = 5

    # GenAI layer - provider agnostic. Absent key => deterministic fallback report.
    llm_provider: str = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: int = 30
    llm_max_tokens: int = 900

    @property
    def model_path(self) -> Path:
        path = Path(self.model_dir)
        return path if path.is_absolute() else self.repo_root / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
