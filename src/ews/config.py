"""Load experiment configuration from YAML."""
from __future__ import annotations

from pathlib import Path

import yaml

from .simulator.dataset import SimConfig


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def sim_config(cfg: dict) -> SimConfig:
    s = dict(cfg["simulation"])
    s["split_fracs"] = tuple(s["split_fracs"])
    return SimConfig(**s)
