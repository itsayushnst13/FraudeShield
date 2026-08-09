"""Typed access to configs/config.yaml.

Config is loaded once and cached. Every threshold, path and hyper-parameter in
the project resolves through here so that business logic is never hardcoded in
application code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until the directory containing configs/config.yaml is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate repository root (configs/config.yaml).")


class Config:
    """Thin, dotted-path wrapper over the parsed YAML config."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self._data = data
        self.root = root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value, e.g. ``cfg.get('threshold.cost_false_negative')``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Like :meth:`get` but raises when the key is absent."""
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {dotted_key}")
        return value

    def path(self, dotted_key: str) -> Path:
        """Resolve a config path value against the repository root."""
        raw = self.require(dotted_key)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))

    def as_dict(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=4)
def load_config(config_path: str | None = None) -> Config:
    """Load and cache the project configuration.

    Resolution order: explicit argument -> ``FRAUDSHIELD_CONFIG`` env var ->
    ``configs/config.yaml`` beside the repository root.
    """
    if config_path is None:
        config_path = os.environ.get("FRAUDSHIELD_CONFIG")
    if config_path:
        resolved = Path(config_path).resolve()
        root = find_repo_root(resolved.parent)
    else:
        root = find_repo_root()
        resolved = root / "configs" / "config.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data, root)
