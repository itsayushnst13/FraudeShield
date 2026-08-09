"""Shared fixtures for ML tests."""

from __future__ import annotations

import numpy as np
import pytest

from fraudshield.config import load_config
from fraudshield.data.synthetic import generate_synthetic_frame


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def frame():
    """Small schema-compatible frame; fast enough to use in every test."""
    return generate_synthetic_frame(n_rows=3000, fraud_rate=0.02, seed=7)


@pytest.fixture(scope="session")
def scores():
    """Deterministic (y_true, y_prob) pair with realistic class separation."""
    rng = np.random.default_rng(11)
    y = np.concatenate([np.zeros(950, dtype=int), np.ones(50, dtype=int)])
    p = np.concatenate([rng.beta(1.2, 12.0, 950), rng.beta(6.0, 2.5, 50)])
    return y, p
