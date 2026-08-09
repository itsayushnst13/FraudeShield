"""MLflow experiment tracking.

Wrapped behind a small facade so that (a) training still runs if MLflow is
unavailable, and (b) the training code is not littered with tracking calls.
Every model variant becomes a run tagged with its imbalance strategy, which is
what makes the "class_weight vs SMOTE vs undersample" comparison auditable
after the fact rather than a claim in a README.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _absolutise(uri: str, root: Path | None) -> str:
    """Resolve a relative sqlite URI against the repository root.

    Without this, ``sqlite:///mlflow.db`` would create a fresh database in
    whatever directory the process happened to start in.
    """
    prefix = "sqlite:///"
    if root is None or not uri.startswith(prefix):
        return uri
    tail = uri[len(prefix) :]
    if tail.startswith("/"):
        return uri
    return prefix + str((root / tail).resolve())


class ExperimentTracker:
    """Thin MLflow facade that degrades to a no-op when MLflow is missing."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        enabled: bool = True,
        root: Path | None = None,
        artifacts_dir: str | None = None,
    ) -> None:
        self.enabled = enabled
        self._mlflow: Any = None
        if not enabled:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(_absolutise(tracking_uri, root))
            artifact_location = None
            if artifacts_dir and root is not None:
                target = (root / artifacts_dir).resolve()
                target.mkdir(parents=True, exist_ok=True)
                artifact_location = target.as_uri()
            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            logger.info("MLflow tracking at %s (experiment '%s')", tracking_uri, experiment_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("MLflow disabled (%s: %s)", type(exc).__name__, exc)
            self.enabled = False

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        """Context manager for a single tracked run."""
        if not self.enabled or self._mlflow is None:
            yield
            return
        with self._mlflow.start_run(run_name=run_name):
            if tags:
                self._mlflow.set_tags(tags)
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled and self._mlflow is not None:
            safe = {k: str(v)[:250] for k, v in params.items() if v is not None}
            self._mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled and self._mlflow is not None:
            numeric = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v == v  # drop NaN
            }
            self._mlflow.log_metrics(numeric, step=step)

    def log_metric_series(self, name: str, values: list[float]) -> None:
        """Log a per-epoch curve so training dynamics are inspectable in the UI."""
        if self.enabled and self._mlflow is not None:
            for step, value in enumerate(values):
                self._mlflow.log_metric(name, float(value), step=step)

    def log_artifact(self, path: str) -> None:
        if self.enabled and self._mlflow is not None:
            try:
                self._mlflow.log_artifact(path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to log artifact %s: %s", path, exc)
