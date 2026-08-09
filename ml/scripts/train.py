"""Train, evaluate, select and register the production fraud model."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from fraudshield.config import load_config
from fraudshield.data.loader import DatasetNotFoundError, dataset_exists
from fraudshield.data.synthetic import SYNTHETIC_MARKER
from fraudshield.logging_utils import setup_logging
from fraudshield.training.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="Path to the transactions CSV.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Train on the SYNTHETIC smoke fixture. Metrics will NOT be real.",
    )
    parser.add_argument("--no-deep", action="store_true", help="Skip PyTorch models (faster CI).")
    parser.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        default=None,
        help="Run hyperparameter search (overrides evaluation.hyperparameter_search).",
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip hyperparameter search.",
    )
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--model-name", type=str, default="fraud_detector")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config seed. Changing it re-splits the data, which is how "
        "you obtain a test set no earlier decision has touched.",
    )
    args = parser.parse_args()
    setup_logging()

    cfg = load_config()
    if args.seed is not None:
        cfg._data["project"]["random_seed"] = int(args.seed)
        print(f"Random seed overridden to {args.seed}: data will be re-split.")
    data_path, data_source = args.data, "KAGGLE_CREDITCARD"

    if args.synthetic:
        data_path = data_path or str(cfg.root / "data" / "raw" / "creditcard_synthetic.csv")
        data_source = SYNTHETIC_MARKER
        print("!" * 72)
        print("TRAINING ON SYNTHETIC SMOKE DATA - RESULTING METRICS ARE NOT REAL.")
        print("!" * 72)
    elif data_path is None and not dataset_exists(cfg):
        print(
            "Real dataset not found. Run 'python ml/scripts/download_data.py', or pass "
            "--synthetic to smoke-test the pipeline with clearly-labelled fake data.",
            file=sys.stderr,
        )
        return 1

    try:
        result = run_pipeline(
            cfg=cfg,
            data_path=data_path,
            data_source=data_source,
            include_deep=not args.no_deep,
            model_name=args.model_name,
            version=args.version,
            tune=args.tune,
        )
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\n=== MODEL COMPARISON (validation) ===")
    print(result.comparison.to_string(index=False))
    print(f"\nSelected model : {result.best_name}")
    print(f"Threshold      : {result.best_threshold}")
    print(f"Test PR-AUC    : {result.test_metrics['pr_auc']:.4f}")
    print(f"Test recall    : {result.test_metrics['recall']:.4f}")
    print(f"Test precision : {result.test_metrics['precision']:.4f}")
    print(f"Artifact       : {result.artifact_dir}")
    if data_source == SYNTHETIC_MARKER:
        print("\nREMINDER: these numbers come from synthetic data and are not meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
