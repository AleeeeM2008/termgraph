"""Small configuration loader for reproducible KIRK experiments.

Paths and training parameters live in ``configs/kirk.json``. Environment
variables override the two machine-specific values:

    KIRK_ROOT       path to the external KIRK repository / data
    TERMGRAPH_CONFIG  alternative config file

No absolute path is baked into the library.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "kirk.json"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path or os.environ.get("TERMGRAPH_CONFIG", DEFAULT_CONFIG))
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text())
    root = Path(os.environ.get("KIRK_ROOT", cfg["kirk_root"]))
    if not root.is_absolute():
        root = (cfg_path.parent.parent / root).resolve()
    cfg["kirk_root"] = str(root)
    return cfg


def data_path(cfg: dict) -> Path:
    return Path(cfg["kirk_root"]) / cfg["data_file"]
