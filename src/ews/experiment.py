"""Experiment orchestration shared by `scripts/run_experiments.py` and the notebook."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .alerting import AlertPolicy, prob_at_least
from .baselines import baseline_stages, fit_single_feature_rule
from .constants import STAGES, WARMUP_STEPS
from .features import CLASSIC_INDEX_FEATURES, CONTEXT_FEATURES, FEATURE_NAMES, build_features_by_episode
from .metrics import (alarm_from_score, calibrate_alarm_rule, evaluate_alarm_level, expected_delay_h,
                      lead_vs_baseline, stage_metrics)
from .pipeline import prepare, subset
from .simulator import SimConfig, generate_dataset

LEVELS = (1, 2, 3)
LEVEL_NAMES = {1: "Incubation alert", 2: "Active-heating alert", 3: "Fire alert"}


@dataclass
class SplitData:
    """All rows of one split (complete episodes, including warm-up) plus aligned features."""
    name: str
    ep: np.ndarray
    ty: np.ndarray
    st: np.ndarray
    F: pd.DataFrame
    sub: pd.DataFrame
    P: dict = field(default_factory=dict)     # model name -> (n, 4) probabilities

    @property
    def valid(self) -> np.ndarray:
        return self.sub["step"].to_numpy() >= WARMUP_STEPS


def make_split(df: pd.DataFrame, F: pd.DataFrame, name: str) -> SplitData:
    mk = (df["split"].to_numpy() == name)
    sub = df[mk].reset_index(drop=True)
    return SplitData(name, sub["episode_id"].to_numpy(), sub["ep_type"].astype(str).to_numpy(),
                     sub["stage"].to_numpy().astype(int), F[mk].reset_index(drop=True), sub)


def load_or_build(cfg: SimConfig, cache_dir: str | Path, force: bool = False, log=print):
    """Simulate + featurise, caching to parquet so reruns are fast."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    p_df, p_f = cache / "sensors.parquet", cache / "features.parquet"
    if p_df.exists() and p_f.exists() and not force:
        log("loading cached dataset/features")
        return pd.read_parquet(p_df), pd.read_parquet(p_f)
    log(f"simulating {cfg.n_episodes} episodes ...")
    df = generate_dataset(cfg, cache, verbose=True)
    log("building features ...")
    F, _ = prepare(df)
    F.to_parquet(p_f, index=False)
    return df, F


# ------------------------------------------------------------------------------------------
def detector_scores(split: SplitData, det: dict, k: int) -> np.ndarray:
    if det["kind"] == "model":
        return prob_at_least(split.P[det["name"]], k)
    return split.F[det["feature"][k]].to_numpy(float)          # tuned rule: per-level feature


def calibrate_and_evaluate(det: dict, V: SplitData, T: SplitData, budget: float, levels=LEVELS) -> dict:
    """Calibrate (threshold, persistence) on validation at an FA budget, evaluate on test."""
    out = {}
    for k in levels:
        sv = detector_scores(V, det, k)
        grid = None
        if det["kind"] == "rule":
            grid = np.unique(np.nanquantile(sv, np.linspace(0.5, 0.9995, 80)))
        th, m, rv = calibrate_alarm_rule(V.ep, V.ty, V.st, sv, k, budget, grid=grid)
        st = detector_scores(T, det, k)
        res = evaluate_alarm_level(T.ep, T.ty, T.st, alarm_from_score(T.ep, st, th, m), k)
        res.update(threshold=th, persistence=m, val_fa_per_100d=rv["fa_per_100d"],
                   expected_delay_h=expected_delay_h(res))
        out[k] = res
    return out


def make_rule_detector(V: SplitData, budget: float) -> dict:
    """Best single-feature tuned threshold rule per level (strong non-ML baseline)."""
    feat = {}
    for k in LEVELS:
        best = fit_single_feature_rule(V.F, {"ep": V.ep, "ty": V.ty, "st": V.st}, k, budget)
        feat[k] = best["feature"]
    return {"kind": "rule", "name": "Tuned threshold rule", "feature": feat}


def handbook_results(T: SplitData) -> dict:
    """Evaluate the two handbook fixed-threshold baselines on a split."""
    out = {}
    for kind, label in (("periodic", "Manual 8-hourly sampling"), ("continuous", "Continuous fixed threshold")):
        stg = baseline_stages(T.sub, kind)
        out[label] = {k: evaluate_alarm_level(T.ep, T.ty, T.st, stg >= k, k) for k in LEVELS}
        out[label]["stage"] = stg
    return out


def summarize(res: dict) -> dict:
    """Strip DataFrames / arrays: keep scalar metrics."""
    keep = ("detection_rate", "median_delay_h", "expected_delay_h", "fa_per_100d", "n_events", "false_alarms",
            "days", "threshold", "persistence", "val_fa_per_100d")
    return {k: (float(res[k]) if isinstance(res[k], (int, float, np.floating)) else res[k])
            for k in keep if k in res}


def stage_eval(split: SplitData, probs: np.ndarray) -> dict:
    m = split.valid
    return stage_metrics(split.st[m], probs[m].argmax(1))


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) or (isinstance(o, np.floating) and np.isnan(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2))


# ------------------------------------------------------------------------------------------
# probability cache so that tabular and sequence models can be trained in separate steps
PROB_DIR = Path("reports/cache")


def save_probs(name: str, split: str, P: np.ndarray, root: Path = PROB_DIR):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"probs_{name}_{split}.npy", P.astype(np.float32))


def load_probs(name: str, split: str, root: Path = PROB_DIR):
    p = root / f"probs_{name}_{split}.npy"
    return np.load(p) if p.exists() else None


MODEL_LABELS = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost",
                "cnn": "1D-CNN", "lstm": "LSTM"}
