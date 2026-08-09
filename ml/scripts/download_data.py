"""Fetch the Kaggle Credit Card Fraud Detection dataset.

Requires Kaggle API credentials at ``~/.kaggle/kaggle.json`` (or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables). If the Kaggle CLI is not
available, manual instructions are printed - no data is ever fabricated.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile

import _bootstrap  # noqa: F401

from fraudshield.config import load_config

DATASET = "mlg-ulb/creditcardfraud"
MANUAL = f"""
Automatic download unavailable. To install the dataset manually:

  1. Open https://www.kaggle.com/datasets/{DATASET}
  2. Sign in and download the archive
  3. Extract 'creditcard.csv' to: {{target}}

Expected: 284,807 rows, 31 columns, 492 fraud cases (0.172%).
"""


def main() -> int:
    cfg = load_config()
    target = cfg.path("paths.raw_data")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        print(f"Dataset already present at {target}")
        return 0

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(target.parent)],
            check=True,
        )
        archive = target.parent / "creditcardfraud.zip"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target.parent)
        archive.unlink(missing_ok=True)
        print(f"Dataset ready at {target}")
        return 0
    except (FileNotFoundError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(MANUAL.format(target=target), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
