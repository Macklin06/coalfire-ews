"""Inference engine: features -> stage probabilities -> persistent alert level + explanation."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..alerting import AlertPolicy, prob_at_least
from ..baselines import continuous_threshold_stage
from ..constants import GAS_COLS, SENSOR_COLS, STAGES
from ..features import BUFFER_STEPS, build_features
from ..indices import o2_deficiency
from ..models.tabular import XGBStageModel
from .describe import describe

ALERT_LABELS = ["No alert", "Level 1 - Incubation alert", "Level 2 - Active-heating alert", "Level 3 - Fire alert"]
FAULT_STEPS = 8          # >= 2 h of missing gas telemetry => telemetry fault (alert state is held, never cleared)


class Predictor:
    def __init__(self, model_path="artifacts/xgb_stage.json", policy_path="artifacts/alert_policy.json"):
        self.model = XGBStageModel.load(model_path)
        self.policy = AlertPolicy.from_dict(json.loads(Path(policy_path).read_text()))
        self.max_persist = max(self.policy.persistence.values())

    def predict_window(self, window: pd.DataFrame, top_k: int = 5) -> dict:
        """Prediction for the LAST row of `window` (needs >= BUFFER_STEPS rows; uses the last BUFFER_STEPS)."""
        window = window.tail(BUFFER_STEPS).reset_index(drop=True)
        feats = build_features(window)
        tail = feats.tail(self.max_persist)
        probs = self.model.predict_proba(tail)
        p_now = probs[-1]
        level, p_ge, exceed = 0, {}, {}
        for k in (1, 2, 3):
            m = self.policy.persistence[k]
            sc = prob_at_least(probs, k)
            p_ge[k] = float(sc[-1])
            exceed[k] = bool((sc[-m:] > self.policy.thresholds[k]).all()) if len(sc) >= m else False
            if exceed[k]:
                level = k
        stage = int(p_now.argmax())
        focus = level if level > 0 else stage
        contrib = self.model.contributions(tail.tail(1))[0]          # (4, F+1)
        c = contrib[focus, :-1]
        order = np.argsort(-np.abs(c))[:top_k]
        last = tail.iloc[-1]
        expl = [{"feature": self.model.feature_names[i], "description": describe(self.model.feature_names[i]),
                 "value": None if np.isnan(last.iloc[i]) else float(last.iloc[i]),
                 "contribution": float(c[i]),
                 "direction": "raises risk" if c[i] > 0 else "lowers risk"} for i in order]
        gas_missing = window[GAS_COLS].tail(FAULT_STEPS).isna().any(axis=1).all()
        row = window.iloc[-1]
        od = float(o2_deficiency(row.o2_pct, row.co2_pct, row.ch4_pct, row.co_ppm)) if not np.isnan(row.o2_pct) else None
        return {
            "stage": STAGES[stage], "stage_index": stage,
            "probs": {STAGES[i]: float(p_now[i]) for i in range(4)},
            "p_at_least": {str(k): p_ge[k] for k in p_ge},
            "risk_score": float(100.0 * (p_now * np.arange(4)).sum() / 3.0),
            "alert_level": level, "alert_label": ALERT_LABELS[level],
            "telemetry_fault": bool(gas_missing),
            "message": ("Gas telemetry lost for >= 2 h: alert state is held; verify with manual sampling."
                        if gas_missing else None),
            "indices": {"o2_deficiency_pct": od, "graham_4h": _f(last.get("graham_4h")),
                        "co_make_lpm": _f(last.get("co_make")), "co_med_4h_ppm": _f(last.get("co_med_4h"))},
            "explanation": expl, "explained_class": STAGES[focus]}


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


class Monitor:
    """Stateful per-panel monitor: feed one reading per sampling interval."""

    def __init__(self, predictor: Predictor):
        self.pred = predictor
        self.buf: deque = deque(maxlen=BUFFER_STEPS)
        self.held_level = 0
        self._base_hist: deque = deque(maxlen=2)

    def prime(self, rows: list[dict]):
        for r in rows:
            self.buf.append(r)

    def push(self, reading: dict) -> dict:
        self.buf.append(reading)
        if len(self.buf) < BUFFER_STEPS:
            return {"ready": False, "n_buffered": len(self.buf), "needed": BUFFER_STEPS}
        out = self.pred.predict_window(pd.DataFrame(list(self.buf)))
        if out["telemetry_fault"]:
            out["alert_level"] = max(out["alert_level"], self.held_level)      # fail-safe: never clear on data loss
            out["alert_label"] = ALERT_LABELS[out["alert_level"]]
        self.held_level = out["alert_level"]
        w = pd.DataFrame(list(self.buf)).tail(4)
        out["baseline_level"] = int(continuous_threshold_stage(w)[-1])
        out["ready"] = True
        return out


def reading_from_row(row) -> dict:
    d = {c: (None if pd.isna(row[c]) else float(row[c])) for c in SENSOR_COLS}
    d["hour"] = float(row["hour"])
    return {k: (np.nan if v is None else v) for k, v in d.items()}
