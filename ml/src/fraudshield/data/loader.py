"""Loading raw transaction data from disk with schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger
from .schema import ALL_COLUMNS, ValidationResult, validate_dataframe

logger = get_logger(__name__)

SETUP_HINT = """
The Kaggle Credit Card Fraud Detection dataset was not found.

  Expected file: {path}

To obtain it:
  1. Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Download 'creditcard.csv' (about 144 MB, 284,807 rows)
  3. Place it at the path above

Or use the Kaggle CLI (requires ~/.kaggle/kaggle.json credentials):
  python ml/scripts/download_data.py

To smoke-test the pipeline WITHOUT the real dataset, generate a clearly-labelled
synthetic fixture (its metrics are NOT meaningful and must never be reported):
  python ml/scripts/make_synthetic_data.py
"""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset CSV is absent, carrying setup instructions."""


def raw_data_path(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg.path("paths.raw_data")


def dataset_exists(cfg: Config | None = None) -> bool:
    return raw_data_path(cfg).is_file()


def load_raw(
    path: str | Path | None = None,
    *,
    cfg: Config | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, ValidationResult]:
    """Load the raw transactions CSV and validate it against the expected schema.

    Raises :class:`DatasetNotFoundError` with actionable setup instructions when
    the file is missing, rather than silently fabricating data.
    """
    cfg = cfg or load_config()
    resolved = Path(path) if path is not None else raw_data_path(cfg)
    if not resolved.is_file():
        raise DatasetNotFoundError(SETUP_HINT.format(path=resolved))

    logger.info("Loading raw dataset from %s", resolved)
    df = pd.read_csv(resolved)
    keep = [c for c in ALL_COLUMNS if c in df.columns]
    df = df[keep]

    result = validate_dataframe(df)
    for warning in result.warnings:
        logger.warning("Validation warning: %s", warning)
    if validate:
        result.raise_if_invalid()
    logger.info(
        "Loaded %d rows | %d fraud (%.4f%%)",
        result.n_rows,
        result.n_fraud,
        100 * result.fraud_rate,
    )
    return df, result
