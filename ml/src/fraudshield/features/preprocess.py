"""Feature engineering and preprocessing.

Everything lives inside a scikit-learn ``Pipeline`` that is fitted on the
TRAINING SPLIT ONLY and then applied unchanged to validation, test and live
inference traffic. Fitting the scaler on the full dataset would leak test
distribution statistics into training - the most common silent bug in fraud
projects, so it is structurally prevented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..config import Config, load_config
from ..data.schema import AMOUNT_COLUMN, PCA_COLUMNS, TIME_COLUMN

SECONDS_PER_DAY = 86_400.0


class TransactionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Derive model features from raw transaction columns.

    * ``Time`` (seconds since the dataset's first transaction) is converted into
      a cyclical hour-of-day encoding. The raw value is dropped by default: it
      encodes *when collection started*, which does not generalise to new traffic.
    * ``Amount`` is right-skewed across several orders of magnitude, so it is
      ``log1p`` transformed before scaling.
    * ``V1..V28`` are already PCA outputs and pass through untouched.
    """

    def __init__(
        self,
        derive_hour_of_day: bool = True,
        drop_raw_time: bool = True,
        log_amount: bool = True,
    ) -> None:
        self.derive_hour_of_day = derive_hour_of_day
        self.drop_raw_time = drop_raw_time
        self.log_amount = log_amount

    def fit(self, X: pd.DataFrame, y=None) -> TransactionFeatureBuilder:
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(self._build(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._build(X)
        if hasattr(self, "feature_names_out_"):
            out = out.reindex(columns=self.feature_names_out_, fill_value=0.0)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)
        for col in PCA_COLUMNS:
            if col in X.columns:
                df[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        if AMOUNT_COLUMN in X.columns:
            amount = pd.to_numeric(X[AMOUNT_COLUMN], errors="coerce").astype(float).clip(lower=0.0)
            df["amount_log"] = np.log1p(amount) if self.log_amount else amount

        if TIME_COLUMN in X.columns:
            time = pd.to_numeric(X[TIME_COLUMN], errors="coerce").astype(float)
            if self.derive_hour_of_day:
                hour = (time % SECONDS_PER_DAY) / 3600.0
                radians = 2.0 * np.pi * hour / 24.0
                df["hour_sin"] = np.sin(radians)
                df["hour_cos"] = np.cos(radians)
            if not self.drop_raw_time:
                df["time_raw"] = time

        return df.fillna(0.0)


def build_preprocessor(cfg: Config | None = None) -> Pipeline:
    """Construct the (unfitted) preprocessing pipeline from config."""
    cfg = cfg or load_config()
    scaler_name = str(cfg.get("features.scaler", "standard")).lower()
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    builder = TransactionFeatureBuilder(
        derive_hour_of_day=bool(cfg.get("features.derive_hour_of_day", True)),
        drop_raw_time=bool(cfg.get("features.drop_raw_time", True)),
        log_amount=bool(cfg.get("features.log_amount", True)),
    )
    return Pipeline([("features", builder), ("scaler", scaler)])


def fit_preprocessor(X_train: pd.DataFrame, cfg: Config | None = None) -> Pipeline:
    """Fit the preprocessing pipeline on training data only."""
    pipeline = build_preprocessor(cfg)
    pipeline.fit(X_train)
    return pipeline


def get_output_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the ordered feature names produced by a fitted pipeline."""
    return [str(name) for name in pipeline.named_steps["features"].get_feature_names_out()]
