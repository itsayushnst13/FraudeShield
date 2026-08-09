"""Run exploratory data analysis and write figures plus a statistics summary."""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from fraudshield.config import load_config
from fraudshield.data.loader import DatasetNotFoundError, load_raw
from fraudshield.eda import run_full_eda
from fraudshield.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="Path to the transactions CSV.")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config()
    try:
        df, _ = load_raw(args.data, cfg=cfg)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    figures_dir = cfg.path("paths.figures_dir")
    stats = run_full_eda(df, figures_dir)

    distribution = stats["class_distribution"]
    print(f"\nRows            : {stats['shape'][0]:,}")
    print(
        f"Fraud           : {distribution['n_fraud']:,} ({distribution['fraud_rate'] * 100:.4f}%)"
    )
    print(f"Imbalance       : 1 fraud per {distribution['imbalance_ratio']:.0f} legitimate")
    print(f"Always-legit acc: {distribution['accuracy_of_always_legit']:.4%}  <- the accuracy trap")
    print(f"Top separating  : {', '.join(stats['separation']['top_features'][:6])}")
    print(f"Figures         : {figures_dir}")

    summary_path = cfg.path("paths.reports_dir") / "eda_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in stats.items() if k != "separation"}
    serialisable["top_separating_features"] = stats["separation"]["top_features"]
    summary_path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
    print(f"Summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
