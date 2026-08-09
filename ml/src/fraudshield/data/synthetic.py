"""SYNTHETIC fixture generator - PIPELINE SMOKE TESTING ONLY.

=============================== READ THIS ===============================
Data produced here is NOT the Kaggle dataset and is NOT a substitute for it.
It exists purely so that the training/serving pipeline can be executed and
verified when the real CSV is unavailable.

Any metric computed on this data is MEANINGLESS and must never be reported
as a model result. Every artifact trained on it is tagged
``data_source="SYNTHETIC_SMOKE"`` so it cannot be mistaken for a real run.
=========================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import PCA_COLUMNS

SYNTHETIC_MARKER = "SYNTHETIC_SMOKE"


def generate_synthetic_frame(
    n_rows: int = 20_000,
    fraud_rate: float = 0.004,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a schema-compatible synthetic frame with a learnable fraud signal.

    Legitimate rows are drawn from a standard normal; fraudulent rows are shifted
    on a handful of components so that a model can actually separate the classes
    and the pipeline exercises a non-degenerate path.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(20, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    legit = rng.standard_normal((n_legit, len(PCA_COLUMNS)))
    fraud = rng.standard_normal((n_fraud, len(PCA_COLUMNS)))
    # Shift a few components for the positive class, mirroring the real dataset
    # where V14/V10/V12/V17 carry most of the separating signal.
    for idx, shift in ((13, -3.0), (9, -2.4), (11, -2.2), (16, -2.6), (3, 1.8)):
        fraud[:, idx] += shift
    fraud += rng.standard_normal(fraud.shape) * 0.6

    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    amount = np.concatenate(
        [
            rng.gamma(shape=1.6, scale=45.0, size=n_legit),
            rng.gamma(shape=1.1, scale=130.0, size=n_fraud),
        ]
    )
    time = np.sort(rng.uniform(0, 172_800, size=n_rows))

    df = pd.DataFrame(X, columns=PCA_COLUMNS)
    df.insert(0, "Time", time)
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_synthetic_csv(destination: str | Path, **kwargs) -> Path:
    """Write a synthetic fixture to disk and return the path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_frame(**kwargs).to_csv(path, index=False)
    return path
