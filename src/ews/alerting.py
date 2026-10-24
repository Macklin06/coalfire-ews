"""Turn per-step class probabilities into persistent, level-based alarms."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


def sustained(mask: np.ndarray, m: int) -> np.ndarray:
    """Causal persistence filter: True where `mask` held for the last `m` samples."""
    if m <= 1:
        return mask.astype(bool)
    r = pd.Series(mask.astype(float)).rolling(m, min_periods=m).min()
    return (r.fillna(0.0).to_numpy() > 0)


def prob_at_least(probs: np.ndarray, k: int) -> np.ndarray:
    """P(stage >= k) from an (n, 4) probability matrix."""
    return probs[:, k:].sum(axis=1)


@dataclass
class AlertPolicy:
    thresholds: dict = field(default_factory=lambda: {1: 0.5, 2: 0.5, 3: 0.5})
    persistence: dict = field(default_factory=lambda: {1: 3, 2: 3, 3: 3})   # consecutive steps above threshold

    def alarms(self, probs: np.ndarray, k: int) -> np.ndarray:
        return sustained(prob_at_least(probs, k) > self.thresholds[k], self.persistence[k])

    def level(self, probs: np.ndarray) -> np.ndarray:
        out = np.zeros(len(probs), dtype=np.int8)
        for k in (1, 2, 3):
            out[self.alarms(probs, k)] = k
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["thresholds"] = {str(k): float(v) for k, v in self.thresholds.items()}
        d["persistence"] = {str(k): int(v) for k, v in self.persistence.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AlertPolicy":
        return cls({int(k): float(v) for k, v in d["thresholds"].items()},
                   {int(k): int(v) for k, v in d["persistence"].items()})
