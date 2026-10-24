"""Status-quo fixed-threshold baselines operating on gas indices (CO ppm, Graham's ratio)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BASELINE_THRESHOLDS, MIN_O2_DEF_FOR_GRAHAM, STEPS_PER_HOUR
from .indices import graham_ratio, o2_deficiency


def _stage_from_instant(co, gr, thresholds) -> np.ndarray:
    """Highest alert level whose CO or Graham's-ratio threshold is exceeded (NaN-safe)."""
    stage = np.zeros(len(co), dtype=np.int8)
    for k in (1, 2, 3):
        thr = thresholds[k]
        hit = (co >= thr["co_ppm"]) | (gr >= thr["graham"])
        stage[np.nan_to_num(hit.astype(float)).astype(bool)] = k
    return stage


def _instant_indices(ep: pd.DataFrame):
    co = ep["co_ppm"].to_numpy(float)
    od = o2_deficiency(ep["o2_pct"].to_numpy(float), ep["co2_pct"].to_numpy(float),
                       ep["ch4_pct"].to_numpy(float), co)
    gr = graham_ratio(co, od, MIN_O2_DEF_FOR_GRAHAM)
    return co, gr


def continuous_threshold_stage(ep: pd.DataFrame, thresholds=BASELINE_THRESHOLDS, persistence: int = 2) -> np.ndarray:
    """Real-time telemetry with fixed trigger levels, requiring `persistence` consecutive exceedances."""
    co, gr = _instant_indices(ep)
    inst = _stage_from_instant(co, gr, thresholds)
    out = np.zeros_like(inst)
    for k in (1, 2, 3):
        m = pd.Series((inst >= k).astype(float)).rolling(persistence, min_periods=persistence).min().fillna(0).to_numpy() > 0
        out[m] = k
    return out


def periodic_manual_stage(ep: pd.DataFrame, thresholds=BASELINE_THRESHOLDS, interval_h: float = 8.0,
                          offset_steps: int = 0) -> np.ndarray:
    """Bag / tube-bundle sampling once per shift: the status is refreshed only at sample times."""
    co, gr = _instant_indices(ep)
    inst = _stage_from_instant(co, gr, thresholds)
    interval = int(interval_h * STEPS_PER_HOUR)
    out = np.zeros(len(ep), dtype=np.int8)
    cur = 0
    for i in range(len(ep)):
        if i >= offset_steps and (i - offset_steps) % interval == 0 and not np.isnan(co[i]):
            cur = inst[i]
        out[i] = cur
    return out


def baseline_stages(df: pd.DataFrame, kind: str, seed: int = 0, **kw) -> np.ndarray:
    """Apply a baseline to each episode in a long DataFrame; returns a stage per row."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, ep in df.groupby("episode_id", sort=False, observed=True):
        ep = ep.reset_index(drop=True)
        if kind == "continuous":
            parts.append(continuous_threshold_stage(ep, **kw))
        elif kind == "periodic":
            parts.append(periodic_manual_stage(ep, offset_steps=int(rng.integers(0, 32)), **kw))
        else:
            raise ValueError(kind)
    return np.concatenate(parts)


TUNED_RULE_FEATURES = ["co_med_4h", "make_med_4h", "co_above_floor", "graham_12h_med", "cor_12h", "make_med_12h"]


def fit_single_feature_rule(feats_val: pd.DataFrame, meta_val: dict, k: int, budget: float):
    """Best single-feature threshold rule (tuned on validation data to the same false-alarm budget).

    This is the *strong* non-ML baseline: a fixed threshold per level, but chosen from data with
    persistence, instead of a handbook value. Returns dict(feature, threshold, persistence, result).
    """
    from .metrics import calibrate_alarm_rule, expected_delay_h
    best = None
    for name in TUNED_RULE_FEATURES:
        x = feats_val[name].to_numpy(float)
        grid = np.unique(np.nanquantile(x, np.linspace(0.5, 0.9995, 80)))
        th, m, res = calibrate_alarm_rule(meta_val["ep"], meta_val["ty"], meta_val["st"], x, k, budget, grid=grid)
        cost = expected_delay_h(res) if res["fa_per_100d"] <= budget else np.inf
        if best is None or cost < best["cost"]:
            best = {"feature": name, "threshold": th, "persistence": m, "result": res, "cost": cost}
    return best
