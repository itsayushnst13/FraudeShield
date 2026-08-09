"""Provider-agnostic LLM access.

The investigation agent depends on this interface, never on a concrete vendor
SDK, so swapping providers is a configuration change (``LLM_PROVIDER``,
``LLM_API_KEY``, ``LLM_MODEL``) rather than a code change.

Adapters speak raw HTTP via ``httpx`` rather than pulling in vendor SDKs, which
keeps the backend image small and the dependency surface auditable.

When no provider is configured the client reports ``available = False`` and the
agent falls back to a deterministic, template-free report assembled from the ML
evidence. The system must never be unable to produce an investigation just
because an API key is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from fraudshield.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Result of a completion request."""

    text: str
    provider: str
    model: str
    ok: bool
    error: str | None = None


class LLMClient(Protocol):
    """Minimal contract the investigation agent relies on."""

    provider: str
    model: str
    available: bool

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


class NullLLMClient:
    """Used when no provider is configured; always reports unavailability."""

    provider = "none"
    model = "none"
    available = False

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(
            text="",
            provider=self.provider,
            model=self.model,
            ok=False,
            error="No LLM provider configured (set LLM_PROVIDER and LLM_API_KEY).",
        )


class _HTTPLLMClient:
    """Shared HTTP plumbing for hosted chat-completion APIs."""

    provider = "http"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.available = bool(api_key and model)

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> LLMResponse:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return LLMResponse(
                    text=self._extract(response.json()),
                    provider=self.provider,
                    model=self.model,
                    ok=True,
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("LLM request failed (%s): %s", self.provider, message)
            return LLMResponse(
                text="", provider=self.provider, model=self.model, ok=False, error=message
            )

    def _extract(self, data: dict[str, Any]) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class AnthropicClient(_HTTPLLMClient):
    """Adapter for the Anthropic Messages API."""

    provider = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int, max_tokens: int):
        super().__init__(
            api_key,
            model or "claude-sonnet-4-5",
            base_url or "https://api.anthropic.com",
            timeout,
            max_tokens,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self._post(
            f"{self.base_url}/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )

    def _extract(self, data: dict[str, Any]) -> str:
        blocks = data.get("content", []) or []
        return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


class OpenAIClient(_HTTPLLMClient):
    """Adapter for OpenAI-compatible chat-completions endpoints."""

    provider = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int, max_tokens: int):
        super().__init__(
            api_key,
            model or "gpt-4o-mini",
            base_url or "https://api.openai.com/v1",
            timeout,
            max_tokens,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self._post(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )

    def _extract(self, data: dict[str, Any]) -> str:
        choices = data.get("choices", []) or []
        if not choices:
            return ""
        return (choices[0].get("message", {}) or {}).get("content", "").strip()


_REGISTRY = {"anthropic": AnthropicClient, "openai": OpenAIClient}


def build_llm_client(
    provider: str,
    api_key: str,
    model: str,
    base_url: str = "",
    timeout: int = 30,
    max_tokens: int = 900,
) -> LLMClient:
    """Instantiate the configured provider, or a null client when unset."""
    key = (provider or "none").strip().lower()
    if key in ("", "none", "disabled"):
        return NullLLMClient()
    factory = _REGISTRY.get(key)
    if factory is None:
        logger.warning(
            "Unknown LLM_PROVIDER '%s'; falling back to deterministic reports.", provider
        )
        return NullLLMClient()
    if not api_key:
        logger.warning("LLM_PROVIDER='%s' set but LLM_API_KEY is empty; using fallback.", key)
        return NullLLMClient()
    return factory(api_key, model, base_url, timeout, max_tokens)
