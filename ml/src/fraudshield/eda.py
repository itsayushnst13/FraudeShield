"""Exploratory data analysis.

Each function returns the numbers behind a plot as well as writing the figure,
so the notebook can state conclusions in prose backed by values rather than
asking the reader to eyeball a chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data.schema import AMOUNT_COLUMN, PCA_COLUMNS, TARGET_COLUMN, TIME_COLUMN  # noqa: E402
from .logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

FRAUD_COLOR = "#9f1239"
LEGIT_COLOR = "#0e7c66"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def class_distribution(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Quantify the imbalance and show why accuracy is the wrong objective."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    n_legit, n_fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    total = n_legit + n_fraud
    fraud_rate = n_fraud / total if total else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(["Legitimate", "Fraud"], [n_legit, n_fraud], color=[LEGIT_COLOR, FRAUD_COLOR])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Transactions (log scale)")
    axes[0].set_title("Class counts")
    axes[1].bar(["Fraud share"], [fraud_rate * 100], color=FRAUD_COLOR, width=0.4)
    axes[1].set_ylabel("% of all transactions")
    axes[1].set_title(f"Fraud rate: {fraud_rate * 100:.3f}%")
    path = _save(fig, out_dir, "class_distribution.png")

    return {
        "n_legitimate": n_legit,
        "n_fraud": n_fraud,
        "fraud_rate": fraud_rate,
        "imbalance_ratio": (n_legit / n_fraud) if n_fraud else float("inf"),
        "accuracy_of_always_legit": (n_legit / total) if total else 0.0,
        "figure": str(path),
    }


def amount_analysis(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Compare the amount distributions of fraudulent and legitimate spend."""
    fraud = df.loc[df[TARGET_COLUMN] == 1, AMOUNT_COLUMN]
    legit = df.loc[df[TARGET_COLUMN] == 0, AMOUNT_COLUMN]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(
        np.log1p(legit), bins=60, color=LEGIT_COLOR, alpha=0.7, density=True, label="Legitimate"
    )
    axes[0].hist(
        np.log1p(fraud), bins=60, color=FRAUD_COLOR, alpha=0.7, density=True, label="Fraud"
    )
    axes[0].set_xlabel("log1p(Amount)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution (log scale)")
    axes[0].legend()
    axes[1].boxplot([legit, fraud], tick_labels=["Legitimate", "Fraud"], showfliers=False)
    axes[1].set_ylabel("Amount")
    axes[1].set_title("Amount spread (outliers hidden)")
    path = _save(fig, out_dir, "amount_analysis.png")

    return {
        "legit_mean": float(legit.mean()),
        "fraud_mean": float(fraud.mean()),
        "legit_median": float(legit.median()),
        "fraud_median": float(fraud.median()),
        "legit_max": float(legit.max()),
        "fraud_max": float(fraud.max()),
        "legit_skew": float(legit.skew()),
        "zero_amount_rows": int((df[AMOUNT_COLUMN] == 0).sum()),
        "figure": str(path),
    }


def time_analysis(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Look for a time-of-day effect in the fraud rate."""
    hour = ((df[TIME_COLUMN] % 86_400) / 3600).astype(int)
    by_hour = df.assign(hour=hour).groupby("hour")[TARGET_COLUMN].agg(["mean", "sum", "count"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(by_hour.index, by_hour["count"], color=LEGIT_COLOR)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("Transaction volume by hour")
    axes[1].bar(by_hour.index, by_hour["mean"] * 100, color=FRAUD_COLOR)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].set_title("Fraud rate by hour")
    path = _save(fig, out_dir, "time_analysis.png")

    return {
        "peak_fraud_hour": int(by_hour["mean"].idxmax()),
        "peak_fraud_rate": float(by_hour["mean"].max()),
        "lowest_volume_hour": int(by_hour["count"].idxmin()),
        "hourly_fraud_rate": by_hour["mean"].round(6).to_dict(),
        "figure": str(path),
    }


def feature_separation(df: pd.DataFrame, out_dir: Path, top_k: int = 8) -> dict[str, Any]:
    """Rank PCA components by how far apart the two class distributions sit.

    Uses a standardised mean difference (Cohen's d), which is scale-free and so
    comparable across components.
    """
    fraud = df[df[TARGET_COLUMN] == 1]
    legit = df[df[TARGET_COLUMN] == 0]

    rows = []
    for col in PCA_COLUMNS:
        if col not in df.columns:
            continue
        pooled = np.sqrt((fraud[col].var() + legit[col].var()) / 2) or 1e-9
        rows.append(
            {
                "feature": col,
                "cohens_d": float((fraud[col].mean() - legit[col].mean()) / pooled),
                "fraud_mean": float(fraud[col].mean()),
                "legit_mean": float(legit[col].mean()),
            }
        )
    ranked = (
        pd.DataFrame(rows)
        .assign(abs_d=lambda t: t["cohens_d"].abs())
        .sort_values("abs_d", ascending=False)
        .reset_index(drop=True)
    )

    top = ranked.head(top_k)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, feature in zip(axes.ravel(), top["feature"], strict=False):
        ax.hist(legit[feature], bins=60, density=True, alpha=0.65, color=LEGIT_COLOR)
        ax.hist(fraud[feature], bins=60, density=True, alpha=0.65, color=FRAUD_COLOR)
        ax.set_title(feature, fontsize=10)
        ax.set_yticks([])
    fig.suptitle("Most separating PCA components (green = legitimate, red = fraud)")
    path = _save(fig, out_dir, "feature_separation.png")

    return {"ranked": ranked, "top_features": top["feature"].tolist(), "figure": str(path)}


def correlation_analysis(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Confirm the PCA components are near-orthogonal, as PCA output should be."""
    features = [c for c in PCA_COLUMNS if c in df.columns] + [AMOUNT_COLUMN]
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title("Feature correlation")
    fig.colorbar(image, ax=ax, shrink=0.8)
    path = _save(fig, out_dir, "correlation.png")

    off_diagonal = corr.to_numpy()[~np.eye(len(features), dtype=bool)]
    target_corr = (
        df[features + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
    )
    return {
        "max_abs_offdiagonal": float(np.abs(off_diagonal).max()),
        "mean_abs_offdiagonal": float(np.abs(off_diagonal).mean()),
        "most_negative_target_corr": target_corr.head(5).round(4).to_dict(),
        "most_positive_target_corr": target_corr.tail(5).round(4).to_dict(),
        "figure": str(path),
    }


def outlier_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Count extreme values per class using a 3-sigma rule on the PCA block."""
    features = [c for c in PCA_COLUMNS if c in df.columns]
    z = (df[features] - df[features].mean()) / df[features].std().replace(0, 1e-9)
    extreme = (z.abs() > 3).sum(axis=1)
    return {
        "mean_extreme_features_legit": float(extreme[df[TARGET_COLUMN] == 0].mean()),
        "mean_extreme_features_fraud": float(extreme[df[TARGET_COLUMN] == 1].mean()),
        "share_fraud_with_any_extreme": float((extreme[df[TARGET_COLUMN] == 1] > 0).mean()),
        "share_legit_with_any_extreme": float((extreme[df[TARGET_COLUMN] == 0] > 0).mean()),
    }


def run_full_eda(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Run every analysis and return the collected statistics."""
    out_dir = Path(out_dir)
    return {
        "shape": df.shape,
        "class_distribution": class_distribution(df, out_dir),
        "amount": amount_analysis(df, out_dir),
        "time": time_analysis(df, out_dir),
        "separation": feature_separation(df, out_dir),
        "correlation": correlation_analysis(df, out_dir),
        "outliers": outlier_summary(df),
    }
