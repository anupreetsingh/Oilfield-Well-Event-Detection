"""YAML config loading with dotted-key overrides (used by the CLIs and Databricks widgets)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """The checkout (or Databricks bundle folder) that relative config paths are resolved against.

    Installed as a wheel on a cluster, the package no longer sits inside the repo, so notebooks
    set OILFIELD_ROOT to the bundle's synced files.
    """
    return Path(os.environ.get("OILFIELD_ROOT", Path(__file__).resolve().parents[2]))


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = repo_root() / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def set_dotted(cfg: dict[str, Any], key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ["tables.catalog=main", "windows.step_seconds=120"] into typed overrides."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def resolve_path(p: str | Path) -> Path:
    """Relative paths in configs are relative to the repo root, so CLIs work from any cwd."""
    p = Path(p)
    if p.is_absolute() or str(p).startswith(("dbfs:", "s3:", "abfss:", "gs:")):
        return p
    return repo_root() / p

