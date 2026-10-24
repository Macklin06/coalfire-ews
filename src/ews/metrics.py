"""Row-level classification metrics and event-level early-warning metrics (lead time, false alarms)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from .constants import STAGES, STEP_MIN, WARMUP_STEPS

STEP_H = STEP_MIN / 60.0


def stage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(STAGES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "qw_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_abs_stage_error": float(np.abs(y_true.astype(int) - y_pred.astype(int)).mean()),
        "per_class": {STAGES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                  "support": int(s[i])} for i in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _episode_slices(ep_ids: np.ndarray):
    change = np.flatnonzero(np.r_[True, ep_ids[1:] != ep_ids[:-1], True])
    return [(change[i], change[i + 1]) for i in range(len(change) - 1)]


def evaluate_alarm_level(ep_ids: np.ndarray, ep_types: np.ndarray, stage: np.ndarray, alarm: np.ndarray,
                         k: int, grace_h: float = 6.0, refractory_h: float = 12.0,
                         warmup: int = WARMUP_STEPS) -> dict:
    """Event-level evaluation of one alert level k (alarm = boolean state per row).

    * event: first step where the latent stage >= k.
    * detected: an alarm is active at some step in [onset - grace, end of episode];
      delay_h = (first such step - onset) * dt  (negative = warned *before* the stage was reached).
    * false alarm: a new alarm onset (merged within `refractory_h`) after which the latent stage stays
      below k for the following `grace_h` hours.
    """
    grace, refr = int(grace_h / STEP_H), int(refractory_h / STEP_H)
    events, fa, days = [], 0, 0.0
    for a, b in _episode_slices(ep_ids):
        s, al = stage[a:b][warmup:], alarm[a:b][warmup:].astype(bool)
        days += len(s) * STEP_H / 24.0
        edges = np.flatnonzero(al & ~np.r_[False, al[:-1]])
        kept, last = [], -10**9
        for e in edges:
            if e - last >= refr:
                kept.append(e)
                last = e
        for e in kept:
            if s[e:e + grace + 1].max() < k:
                fa += 1
        hit = np.flatnonzero(s >= k)
        if len(hit):
            onset = int(hit[0])
            start = max(onset - grace, 0)
            idx = np.flatnonzero(al[start:])
            det = int(start + idx[0]) if len(idx) else None
            events.append({"episode_id": int(ep_ids[a]), "ep_type": str(ep_types[a]), "onset": onset,
                           "detected": det is not None,
                           "delay_h": (det - onset) * STEP_H if det is not None else np.nan})
    ev = pd.DataFrame(events)
    return {"events": ev, "false_alarms": int(fa), "days": float(days),
            "fa_per_100d": 100.0 * fa / max(days, 1e-9),
            "detection_rate": float(ev["detected"].mean()) if len(ev) else np.nan,
            "median_delay_h": float(ev["delay_h"].median()) if len(ev) else np.nan,
            "n_events": int(len(ev))}


def lead_vs_baseline(model_events: pd.DataFrame, base_events: pd.DataFrame) -> dict:
    """Paired lead time (hours) of the model over a baseline on the events both can see."""
    m = model_events.merge(base_events, on=["episode_id", "onset"], suffixes=("_m", "_b"))
    both = m[m.detected_m & m.detected_b]
    lead = (both.delay_h_b - both.delay_h_m)
    return {"n_events": int(len(m)), "n_paired": int(len(both)),
            "model_only_detected": int((m.detected_m & ~m.detected_b).sum()),
            "baseline_only_detected": int((~m.detected_m & m.detected_b).sum()),
            "median_lead_h": float(lead.median()) if len(lead) else np.nan,
            "mean_lead_h": float(lead.mean()) if len(lead) else np.nan,
            "q25_lead_h": float(lead.quantile(0.25)) if len(lead) else np.nan,
            "q75_lead_h": float(lead.quantile(0.75)) if len(lead) else np.nan}


MISS_PENALTY_H = 168.0   # a missed event is scored as a 7-day delay when ranking operating points


def _alarm_from_score(ep_ids, score, th, persistence):
    from .alerting import sustained
    alarm = np.zeros(len(score), dtype=bool)
    for a, b in _episode_slices(ep_ids):
        alarm[a:b] = sustained(np.nan_to_num(score[a:b], nan=-np.inf) > th, persistence)
    return alarm


def expected_delay_h(res: dict) -> float:
    ev = res["events"]
    if not len(ev):
        return np.nan
    d = ev["delay_h"].where(ev["detected"], MISS_PENALTY_H)
    return float(d.clip(upper=MISS_PENALTY_H).mean())


def calibrate_alarm_rule(ep_ids, ep_types, stage, score, k: int, fa_budget_per_100d: float,
                         persistences=(2, 4, 8, 16), grid: np.ndarray | None = None,
                         warmup: int = WARMUP_STEPS):
    """Pick (threshold, persistence) minimising mean detection delay subject to a false-alarm budget.

    `score` is any monotone risk score (class probability P(stage>=k) or a raw feature). Returns
    (threshold, persistence, result_dict_on_the_calibration_set).
    """
    grid = np.linspace(0.02, 0.995, 60) if grid is None else np.asarray(grid, float)
    best = None
    for m in persistences:
        for th in grid:                     # ascending threshold -> descending false alarms
            res = evaluate_alarm_level(ep_ids, ep_types, stage, _alarm_from_score(ep_ids, score, th, m), k,
                                       warmup=warmup)
            if res["fa_per_100d"] <= fa_budget_per_100d:
                cost = expected_delay_h(res)
                if best is None or cost < best[0]:
                    best = (cost, float(th), int(m), res)
                break
    if best is None:                        # budget unreachable: fall back to the strictest setting
        m, th = persistences[-1], float(grid[-1])
        res = evaluate_alarm_level(ep_ids, ep_types, stage, _alarm_from_score(ep_ids, score, th, m), k, warmup=warmup)
        return th, m, res
    return best[1], best[2], best[3]


def alarm_from_score(ep_ids, score, th, persistence):
    return _alarm_from_score(ep_ids, score, th, persistence)
