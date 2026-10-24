"""High-level helpers: build features/labels from a simulated dataset and score models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import WARMUP_STEPS
from .features import build_features_by_episode


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, meta) aligned row-for-row with `df`; meta has labels and bookkeeping."""
    feats = build_features_by_episode(df)
    meta = df[["episode_id", "step", "stage", "ep_type", "split", "coal_temp_c"]].copy()
    meta["valid"] = df["step"].to_numpy() >= WARMUP_STEPS
    return feats, meta


def subset(feats, meta, split: str, stride: int = 1):
    m = (meta["split"].to_numpy() == split) & meta["valid"].to_numpy()
    if stride > 1:
        m &= (meta["step"].to_numpy() % stride == 0)
    return feats[m], meta.loc[m, "stage"].to_numpy().astype(int), m
