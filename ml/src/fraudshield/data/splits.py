"""Leakage-safe stratified train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import Config, load_config
from ..logging_utils import get_logger
from .schema import FEATURE_COLUMNS, TARGET_COLUMN

logger = get_logger(__name__)


@dataclass
class DataSplits:
    """Container for the three disjoint dataset partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, y in (("train", self.y_train), ("val", self.y_val), ("test", self.y_test)):
            rows.append(
                {
                    "split": name,
                    "n_rows": int(len(y)),
                    "n_fraud": int(y.sum()),
                    "fraud_rate": float(y.mean()),
                }
            )
        return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame, cfg: Config | None = None) -> DataSplits:
    """Split into train/val/test with stratification on the fraud label.

    The test set is carved out first and must be touched exactly once, at the
    very end. Threshold selection and model selection both use the validation
    split only - this is the core leakage control of the project.
    """
    cfg = cfg or load_config()
    seed = cfg.seed
    test_size = float(cfg.get("data.test_size", 0.2))
    val_size = float(cfg.get("data.val_size", 0.2))
    stratify_flag = bool(cfg.get("data.stratify", True))

    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[features].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    strat = y if stratify_flag else None
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    strat_rest = y_rest if stratify_flag else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, random_state=seed, stratify=strat_rest
    )

    splits = DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)
    logger.info("Split summary:\n%s", splits.summary().to_string(index=False))
    return splits
