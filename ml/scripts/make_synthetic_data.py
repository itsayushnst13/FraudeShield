"""Generate a SYNTHETIC smoke-test fixture (NOT the real Kaggle dataset).

Metrics produced from this file are meaningless and must never be reported as
model performance. Use it only to verify that the pipeline executes.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fraudshield.config import load_config
from fraudshield.data.synthetic import write_synthetic_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40_000)
    parser.add_argument("--fraud-rate", type=float, default=0.0035)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    destination = args.out or (cfg.root / "data" / "raw" / "creditcard_synthetic.csv")
    path = write_synthetic_csv(
        destination, n_rows=args.rows, fraud_rate=args.fraud_rate, seed=cfg.seed
    )
    print("=" * 72)
    print("WARNING: SYNTHETIC SMOKE DATA - NOT THE KAGGLE DATASET")
    print("Any metric computed from this file is NOT a real model result.")
    print("=" * 72)
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
