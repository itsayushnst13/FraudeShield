"""Tests for the deterministic risk engine."""

from __future__ import annotations

import pytest
from app.core.risk_engine import RiskEngine, get_risk_engine


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "LOW"), (0.29, "LOW"), (0.30, "MEDIUM"), (0.69, "MEDIUM"),
        (0.70, "HIGH"), (0.89, "HIGH"), (0.90, "CRITICAL"), (1.0, "CRITICAL"),
    ],
)
def test_band_boundaries_are_inclusive_at_the_floor(probability, expected):
    assert get_risk_engine().assess(probability).risk_level == expected


def test_each_band_maps_to_its_configured_action():
    engine = get_risk_engine()
    assert engine.assess(0.1).recommended_action == "ALLOW"
    assert engine.assess(0.5).recommended_action == "ALLOW_WITH_MONITORING"
    assert engine.assess(0.8).recommended_action == "HOLD_FOR_MANUAL_REVIEW"
    assert engine.assess(0.95).recommended_action == "BLOCK_AND_ESCALATE"


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
def test_out_of_range_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        get_risk_engine().assess(bad)


def test_assessment_is_deterministic():
    engine = get_risk_engine()
    assert engine.assess(0.712) == engine.assess(0.712)


def test_bands_are_sorted_regardless_of_config_order():
    engine = RiskEngine(
        bands=[
            {"name": "HIGH", "min_probability": 0.7},
            {"name": "LOW", "min_probability": 0.0},
            {"name": "MEDIUM", "min_probability": 0.3},
        ],
        actions={"LOW": "ALLOW", "MEDIUM": "WATCH", "HIGH": "REVIEW"},
    )
    assert engine.levels == ["LOW", "MEDIUM", "HIGH"]
    assert engine.assess(0.5).risk_level == "MEDIUM"


def test_custom_thresholds_shift_the_bands():
    engine = RiskEngine(
        bands=[{"name": "LOW", "min_probability": 0.0}, {"name": "HIGH", "min_probability": 0.5}],
        actions={"LOW": "ALLOW", "HIGH": "BLOCK"},
    )
    assert engine.assess(0.4).risk_level == "LOW"
    assert engine.assess(0.6).risk_level == "HIGH"


def test_band_ceiling_is_none_for_the_top_band():
    assert get_risk_engine().assess(0.99).band_ceiling is None


def test_policy_description_covers_every_band():
    policy = get_risk_engine().policy_description()
    assert len(policy) == 4
    assert {p["risk_level"] for p in policy} == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert all("action" in p and "min_probability" in p for p in policy)
