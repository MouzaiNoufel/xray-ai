"""Machine-specific settings (dataset paths) live in configs/local.yaml.

Set the XRAY_LOCAL_CONFIG environment variable to use another file. The smoke test does
this so it never touches your real local.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def local_config_path() -> Path:
    return Path(os.environ.get("XRAY_LOCAL_CONFIG", ROOT / "configs" / "local.yaml"))


def load_local() -> dict:
    p = local_config_path()
    if not p.exists():
        raise SystemExit(
            f"{p} not found. Copy configs/local.example.yaml to configs/local.yaml and set your dataset paths."
        )
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
