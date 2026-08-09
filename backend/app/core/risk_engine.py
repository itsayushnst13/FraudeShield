"""Deterministic probability -> risk band mapping.

Kept deliberately dumb and deterministic: given the same probability it always
returns the same band and the same recommended action. Bands and actions live in
configs/config.yaml, so risk appetite can be retuned without a code change, and
the LLM investigation layer can never alter a risk decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fraudshield.config import load_config
from fraudshield.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BANDS = [
    {"name": "LOW", "min_probability": 0.0},
    {"name": "MEDIUM", "min_probability": 0.3},
    {"name": "HIGH", "min_probability": 0.7},
    {"name": "CRITICAL", "min_probability": 0.9},
]
DEFAULT_ACTIONS = {
    "LOW": "ALLOW",
    "MEDIUM": "ALLOW_WITH_MONITORING",
    "HIGH": "HOLD_FOR_MANUAL_REVIEW",
    "CRITICAL": "BLOCK_AND_ESCALATE",
}


@dataclass(frozen=True)
class RiskAssessment:
    """Outcome of scoring one probability against the configured bands."""

    risk_level: str
    recommended_action: str
    band_floor: float
    band_ceiling: float | None


class RiskEngine:
    """Maps a fraud probability onto a named risk band and recommended action."""

    def __init__(self, bands: list[dict] | None = None, actions: dict[str, str] | None = None):
        raw = bands or DEFAULT_BANDS
        self.bands = sorted(raw, key=lambda b: float(b["min_probability"]))
        self.actions = actions or DEFAULT_ACTIONS

    @property
    def levels(self) -> list[str]:
        return [str(b["name"]) for b in self.bands]

    def assess(self, probability: float) -> RiskAssessment:
        """Return the band containing ``probability`` (validated to [0, 1])."""
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"Probability must be within [0, 1], got {probability}.")

        selected_index = 0
        for index, band in enumerate(self.bands):
            if float(probability) >= float(band["min_probability"]):
                selected_index = index
        band = self.bands[selected_index]
        ceiling = (
            float(self.bands[selected_index + 1]["min_probability"])
            if selected_index + 1 < len(self.bands)
            else None
        )
        name = str(band["name"])
        return RiskAssessment(
            risk_level=name,
            recommended_action=self.actions.get(name, "REVIEW"),
            band_floor=float(band["min_probability"]),
            band_ceiling=ceiling,
        )

    def policy_description(self) -> list[dict]:
        """Human/agent-readable description of the active policy."""
        return [
            {
                "risk_level": str(b["name"]),
                "min_probability": float(b["min_probability"]),
                "action": self.actions.get(str(b["name"]), "REVIEW"),
            }
            for b in self.bands
        ]


@lru_cache(maxsize=1)
def get_risk_engine() -> RiskEngine:
    """Build the engine from the shared YAML config, falling back to defaults."""
    try:
        cfg = load_config()
        return RiskEngine(
            bands=cfg.get("risk_engine.bands", DEFAULT_BANDS),
            actions=cfg.get("risk_engine.actions", DEFAULT_ACTIONS),
        )
    except Exception as exc:
        # Falling back silently would hide a broken config behind plausible-looking
        # risk bands, so the substitution is logged loudly.
        logger.warning(
            "Could not load risk bands from config (%s: %s); using built-in defaults %s.",
            type(exc).__name__,
            exc,
            [b["name"] for b in DEFAULT_BANDS],
        )
        return RiskEngine()
