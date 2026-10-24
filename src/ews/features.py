"""Causal rolling-window feature engineering.

The *same* function is used offline (whole episodes) and online (a rolling buffer of the most
recent `BUFFER_STEPS` samples): every feature at time t depends only on samples in
(t - 96 steps, t], so the last row computed on a 192-step buffer equals the offline value.
`tests/test_features.py` asserts this train/serve equivalence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import SENSOR_COLS, STEPS_PER_HOUR
from .indices import (carbon_oxides_ratio, co_make_lpm, graham_ratio, o2_deficiency,
                      young_ratio)

BUFFER_STEPS = 192          # online history kept per panel (48 h at 15 min)
FFILL_LIMIT = 16            # forward-fill gaps up to 4 h; longer gaps stay NaN (and are flagged)
H1, H4, H12, H24 = (STEPS_PER_HOUR * h for h in (1, 4, 12, 24))   # window lengths in steps


def _roll(s: pd.Series, w: int, fn: str, minp: int | None = None) -> pd.Series:
    r = s.rolling(w, min_periods=minp or max(1, w // 2))
    return getattr(r, fn)()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for ONE contiguous series. Input columns: SENSOR_COLS + 'hour'."""
    d = raw[SENSOR_COLS].astype("float64").reset_index(drop=True)
    missing = d[["co_ppm", "co2_pct", "o2_pct", "ch4_pct"]].isna().any(axis=1).astype("float64")
    d = d.ffill(limit=FFILL_LIMIT)
    f: dict[str, pd.Series] = {}

    co, co2, o2, ch4, q, p = (d[c] for c in ("co_ppm", "co2_pct", "o2_pct", "ch4_pct", "airflow_m3s", "baro_hpa"))
    o2def = pd.Series(o2_deficiency(o2, co2, ch4, co), index=d.index)
    make = pd.Series(co_make_lpm(co, q), index=d.index)
    cor = pd.Series(carbon_oxides_ratio(co, co2), index=d.index)

    # --- raw / instantaneous -----------------------------------------------------------------
    f["co_ppm"], f["co2_pct"], f["o2_pct"], f["ch4_pct"] = co, co2, o2, ch4
    f["temp_c"], f["airflow"], f["o2_def"], f["co_make"] = d["temp_c"], q, o2def, make

    # --- CO / CO-make level and trend (spike-robust: medians and minima) -----------------------
    for w, tag in ((H1, "1h"), (H4, "4h"), (H12, "12h"), (H24, "24h")):
        f[f"co_mean_{tag}"] = _roll(co, w, "mean")
        f[f"make_mean_{tag}"] = _roll(make, w, "mean")
    for w, tag in ((H4, "4h"), (H12, "12h")):
        f[f"co_med_{tag}"] = _roll(co, w, "median")
        f[f"make_med_{tag}"] = _roll(make, w, "median")
    for w, tag in ((H4, "4h"), (H12, "12h"), (H24, "24h")):
        f[f"co_min_{tag}"] = _roll(co, w, "min")
        f[f"make_min_{tag}"] = _roll(make, w, "min")
    f["co_std_4h"] = _roll(co, H4, "std", 4)
    f["co_max_4h"] = _roll(co, H4, "max")
    f["co_spikiness"] = f["co_mean_1h"] - f["co_min_4h"]                       # blast signature
    f["co_above_floor"] = f["co_med_4h"] - f["co_min_24h"]
    f["make_slope_4h"] = (f["make_med_4h"] - _roll(f["make_med_4h"].shift(H4), H4, "median", 1)) / 4.0   # L/min per h
    f["make_slope_12h"] = (f["make_med_12h"] - f["make_med_12h"].shift(H12)) / 12.0
    f["co_slope_12h"] = (f["co_med_12h"] - f["co_med_12h"].shift(H12)) / 12.0
    f["make_ratio_24h"] = f["make_med_12h"] / (f["make_min_24h"] + 0.5)         # growth vs floor
    f["co_log_slope_24h"] = (np.log1p(f["co_med_12h"]) - np.log1p(f["co_med_12h"].shift(H12))) / 12.0

    # --- oxygen deficiency, Graham / Young / carbon-oxides ratios -----------------------------------
    f["o2def_mean_4h"] = _roll(o2def, H4, "mean")
    f["o2def_mean_24h"] = _roll(o2def, H24, "mean")
    f["o2def_slope_12h"] = (_roll(o2def, H12, "mean") - _roll(o2def, H12, "mean").shift(H12)) / 12.0
    co_sm, def_sm = _roll(co, H4, "mean"), f["o2def_mean_4h"]
    gr_sm = pd.Series(graham_ratio(co_sm, def_sm, min_def=0.1), index=d.index)
    f["graham_4h"] = gr_sm
    f["graham_12h_med"] = _roll(gr_sm, H12, "median", 4)
    co2_excess = co2 - _roll(co2, H24, "min")
    f["co2_excess"] = co2_excess
    f["young_4h"] = pd.Series(young_ratio(_roll(co2_excess, H4, "mean"), def_sm, min_def=0.1), index=d.index)
    f["cor_4h"] = _roll(cor, H4, "median")
    f["cor_12h"] = _roll(cor, H12, "median")
    f["co2_slope_12h"] = (_roll(co2, H12, "mean") - _roll(co2, H12, "mean").shift(H12)) / 12.0

    # --- context: methane, ventilation, pressure, time of day ---------------------------------------
    f["ch4_mean_4h"] = _roll(ch4, H4, "mean")
    f["ch4_std_24h"] = _roll(ch4, H24, "std", 8)
    f["airflow_rel_24h"] = q / _roll(q, H24, "mean")
    f["airflow_chg_4h"] = q / q.shift(H4) - 1.0
    f["baro_chg_4h"] = p - p.shift(H4)
    f["temp_chg_24h"] = d["temp_c"] - d["temp_c"].shift(H24)
    hr = raw["hour"].astype("float64").reset_index(drop=True)
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    f["missing_frac_24h"] = _roll(missing, H24, "mean", 1)
    f["stale_co"] = (d["co_ppm"].diff().abs() == 0).astype("float64").rolling(H4, min_periods=1).mean()

    out = pd.DataFrame(f)
    return out.astype("float32")


FEATURE_NAMES: list[str] = list(build_features(
    pd.DataFrame({**{c: np.ones(200) for c in SENSOR_COLS}, "hour": np.zeros(200)})).columns)

# Feature groups used in ablations
CONTEXT_FEATURES = ["airflow", "airflow_rel_24h", "airflow_chg_4h", "baro_chg_4h", "ch4_pct",
                    "ch4_mean_4h", "ch4_std_24h", "hour_sin", "hour_cos"]
CLASSIC_INDEX_FEATURES = ["o2_def", "graham_4h", "graham_12h_med", "young_4h", "cor_4h", "cor_12h",
                          "co_make"]


def build_features_by_episode(df: pd.DataFrame, group_col: str = "episode_id") -> pd.DataFrame:
    """Apply `build_features` to every episode of a long DataFrame (rows must be time-ordered)."""
    parts = []
    for _, g in df.groupby(group_col, sort=False, observed=True):
        parts.append(build_features(g))
    out = pd.concat(parts, ignore_index=True)
    out.index = df.index if len(out) == len(df) else out.index
    return out
