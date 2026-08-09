"""Expected schema of the Kaggle Credit Card Fraud Detection dataset.

The real dataset (ULB / Worldline, 2013 European cardholders) has 284,807 rows
and 31 columns: ``Time``, ``V1``..``V28`` (PCA components), ``Amount``, ``Class``.
Everything downstream is written against this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

N_PCA_COMPONENTS = 28
PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, N_PCA_COMPONENTS + 1)]
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"

FEATURE_COLUMNS: list[str] = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]
ALL_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET_COLUMN]


@dataclass
class ValidationResult:
    """Outcome of a dataset schema/quality check."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_fraud: int = 0

    @property
    def fraud_rate(self) -> float:
        return (self.n_fraud / self.n_rows) if self.n_rows else 0.0

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Dataset validation failed:\n  - " + "\n  - ".join(self.errors))


def validate_dataframe(df: pd.DataFrame, *, require_target: bool = True) -> ValidationResult:
    """Validate a raw transactions frame against the expected schema.

    Checks column presence, dtypes, nulls, target cardinality and Amount sign.
    Returns a structured result rather than raising, so callers can decide.
    """
    errors: list[str] = []
    warnings: list[str] = []

    required = list(FEATURE_COLUMNS) + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra:
        warnings.append(f"Unexpected extra columns (ignored): {extra}")

    if df.empty:
        errors.append("Dataset is empty.")

    for col in [c for c in FEATURE_COLUMNS if c in df.columns]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}.")
        elif df[col].isna().any():
            errors.append(f"Column '{col}' contains {int(df[col].isna().sum())} null values.")

    if AMOUNT_COLUMN in df.columns and pd.api.types.is_numeric_dtype(df[AMOUNT_COLUMN]):
        if (df[AMOUNT_COLUMN] < 0).any():
            errors.append("Negative values found in 'Amount'.")

    n_fraud = 0
    if require_target and TARGET_COLUMN in df.columns:
        unique = set(pd.unique(df[TARGET_COLUMN].dropna()))
        if not unique.issubset({0, 1}):
            errors.append(f"'Class' must be binary 0/1, found values: {sorted(unique)}")
        n_fraud = int((df[TARGET_COLUMN] == 1).sum())
        if n_fraud == 0:
            errors.append("No positive (fraud) samples present; cannot train or evaluate.")
        elif n_fraud < 50:
            warnings.append(f"Only {n_fraud} fraud samples - metrics will be very high variance.")

    return ValidationResult(
        ok=not errors, errors=errors, warnings=warnings, n_rows=len(df), n_fraud=n_fraud
    )
