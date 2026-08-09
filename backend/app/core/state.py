"""Process-wide mutable service state.

Deliberately tiny and explicit. In a multi-replica deployment these counters
would move to Redis or a metrics backend; keeping them here makes the swap
obvious rather than hiding it behind module globals scattered across the app.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class ServiceCounters:
    """Thread-safe scoring counters for the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_scored = 0
        self.fraud_detected = 0

    def record(self, n_scored: int, n_fraud: int) -> None:
        with self._lock:
            self.total_scored += n_scored
            self.fraud_detected += n_fraud

    @property
    def fraud_rate(self) -> float:
        return (self.fraud_detected / self.total_scored) if self.total_scored else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_transactions_scored": self.total_scored,
            "fraud_detected": self.fraud_detected,
            "fraud_rate": round(self.fraud_rate, 6),
        }


counters = ServiceCounters()


def load_offline_metrics(model_dir: Path) -> dict[str, Any]:
    """Read metrics.json written by the training pipeline, if present."""
    path = Path(model_dir) / "metrics.json"
    if not path.is_file():
        return {"available": False, "detail": "No metrics.json found; run the training pipeline."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"available": False, "detail": f"metrics.json is not valid JSON: {exc}"}
    payload["available"] = True
    return payload
