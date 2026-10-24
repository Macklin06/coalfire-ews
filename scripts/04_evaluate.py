"""Step 4: full evaluation -> reports/results.json (+ reports/cache/eval.pkl for the figures).

Covers: stage classification, event-level early-warning metrics at a matched false-alarm budget,
operating curves, feature ablations, simulator-sensitivity stress tests, named edge-case scenarios,
TreeSHAP explanations, and the serving artifacts (alert policy + model card metadata).
"""
import argparse
import datetime as dt
import json
import pickle
import time

import numpy as np
import pandas as pd

from ews import __version__
from ews.alerting import AlertPolicy, prob_at_least
from ews.baselines import baseline_stages, continuous_threshold_stage
from ews.config import load_config, sim_config
from ews.constants import STAGES
from ews.experiment import (LEVEL_NAMES, LEVELS, MODEL_LABELS, calibrate_and_evaluate, detector_scores,
                            handbook_results, load_or_build, load_probs, make_rule_detector, make_split,
                            save_json, stage_eval, summarize)
from ews.features import CLASSIC_INDEX_FEATURES, CONTEXT_FEATURES, FEATURE_NAMES, build_features
from ews.metrics import alarm_from_score, evaluate_alarm_level, lead_vs_baseline, stage_metrics
from ews.models.tabular import XGBStageModel
from ews.pipeline import prepare, subset
from ews.simulator import SimConfig, generate_dataset
from ews.simulator.scenarios import SCENARIOS, build_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--data", default="data/processed")
ap.add_argument("--skip-ablation", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config)
BUDGET = cfg["alerting"]["fa_budget_per_100_panel_days"]
t00 = time.time()
log = lambda *s: print(f"[{time.time() - t00:6.0f}s]", *s, flush=True)

df, F = load_or_build(sim_config(cfg), a.data, log=log)
V, T = make_split(df, F, "val"), make_split(df, F, "test")
models = [m for m in ("logreg", "rf", "xgb", "cnn", "lstm") if load_probs(m, "test") is not None]
for m in models:
    V.P[m], T.P[m] = load_probs(m, "val"), load_probs(m, "test")
log("models available:", models)
res: dict = {"config": cfg, "version": __version__, "models": models, "created": dt.datetime.now(dt.UTC).isoformat()}
pk: dict = {}

# --- dataset summary ----------------------------------------------------------------------------
valid = df["step"] >= 192
res["dataset"] = {
    "rows": int(len(df)), "episodes": int(df.episode_id.nunique()), "days_per_episode": cfg["simulation"]["n_days"],
    "episodes_by_split": df.groupby("split", observed=True).episode_id.nunique().to_dict(),
    "episodes_by_type": df.groupby("ep_type", observed=True).episode_id.nunique().to_dict(),
    "stage_share_rows": {STAGES[k]: float(v) for k, v in df[valid].stage.value_counts(normalize=True).sort_index().items()},
    "missing_share": {c: float(df[c].isna().mean()) for c in ("co_ppm", "co2_pct", "o2_pct", "ch4_pct")},
    "n_features": len(FEATURE_NAMES)}

# --- E1: stage classification -------------------------------------------------------------------
hb = handbook_results(T)
res["stage"] = {m: stage_eval(T, T.P[m]) for m in models}
for label, d in hb.items():
    m = T.valid
    res["stage"][label] = stage_metrics(T.st[m], d["stage"][m])
log("stage metrics:", {k: round(v["macro_f1"], 3) for k, v in res["stage"].items()})

# --- E2: event-level early-warning at matched FA budget ----------------------------------------
rule = make_rule_detector(V, BUDGET)
dets = [{"kind": "model", "name": m} for m in models] + [rule]
ev = {}
for d in dets:
    ev[d["name"]] = calibrate_and_evaluate(d, V, T, BUDGET)
    log("events", d["name"], {k: (round(r["detection_rate"], 2), round(r["median_delay_h"], 1), round(r["fa_per_100d"], 2))
                               for k, r in ev[d["name"]].items()})
res["rule_features"] = rule["feature"]
res["events"] = {n: {k: summarize(r) for k, r in v.items()} for n, v in ev.items()}
res["events"].update({n: {k: summarize(d[k]) for k in LEVELS} for n, d in hb.items()})
res["handbook_thresholds_note"] = "CO>=15/50/400 ppm or Graham>=0.5/1/3 (see constants.py)"

paired = {}
for n, per in ev.items():
    paired[n] = {}
    for base_name, base in list(((b, hb[b]) for b in hb)) + [("Tuned threshold rule", ev["Tuned threshold rule"])]:
        if base_name == n:
            continue
        paired[n][base_name] = {k: lead_vs_baseline(per[k]["events"], base[k]["events"]) for k in LEVELS}
res["paired_lead"] = paired
pk["events"] = {n: {k: r["events"] for k, r in v.items()} for n, v in ev.items()}
pk["events"].update({n: {k: d[k]["events"] for k in LEVELS} for n, d in hb.items()})

# --- E3: operating curves (delay vs. false-alarm budget) ---------------------------------------
curves = {}
cands = [d for d in dets if d["name"] in ("xgb", "Tuned threshold rule") or d["name"] in ("cnn", "lstm")]
for d in cands:
    curves[d["name"]] = {}
    for b in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        out = calibrate_and_evaluate(d, V, T, b, levels=(1, 2))
        curves[d["name"]][b] = {k: summarize(r) for k, r in out.items()}
    log("curve", d["name"])
res["operating_curves"] = curves

# --- serving artifacts: alert policy from the XGBoost calibration -----------------------------
pol = AlertPolicy({k: ev["xgb"][k]["threshold"] for k in LEVELS}, {k: ev["xgb"][k]["persistence"] for k in LEVELS})
json.dump(pol.to_dict(), open("artifacts/alert_policy.json", "w"), indent=2)

# --- E4: feature ablations (XGBoost) ----------------------------------------------------------
if not a.skip_ablation:
    meta = df[["episode_id", "step", "stage", "ep_type", "split"]].copy()
    meta["valid"] = meta["step"] >= 192
    INST = ["co_ppm", "co2_pct", "o2_pct", "ch4_pct", "temp_c", "airflow", "o2_def", "co_make"]
    sets = {
        "All features": FEATURE_NAMES,
        "No context (airflow / pressure / CH4 / time)": [f for f in FEATURE_NAMES if f not in CONTEXT_FEATURES],
        "No classical index features": [f for f in FEATURE_NAMES if f not in CLASSIC_INDEX_FEATURES],
        "No spike-robust statistics (min / median)": [f for f in FEATURE_NAMES if not any(s in f for s in ("_min_", "_med_", "spikiness", "above_floor"))],
        "Instantaneous values only (no rolling windows)": INST,
    }
    abl = {}
    for name, cols in sets.items():
        Xtr, ytr, _ = subset(F[cols], meta, "train", cfg["training"]["tabular_stride"])
        Xva, yva, _ = subset(F[cols], meta, "val", cfg["training"]["tabular_stride"])
        mdl = XGBStageModel().fit(Xtr, ytr, Xva, yva)
        V2, T2 = make_split(df, F[cols], "val"), make_split(df, F[cols], "test")
        V2.P["x"], T2.P["x"] = mdl.predict_proba(V2.F), mdl.predict_proba(T2.F)
        out = calibrate_and_evaluate({"kind": "model", "name": "x"}, V2, T2, BUDGET, levels=(1, 2))
        abl[name] = {"n_features": len(cols), "macro_f1": stage_eval(T2, T2.P["x"])["macro_f1"],
                     **{f"L{k}": summarize(r) for k, r in out.items()}}
        log("ablation", name, round(abl[name]["macro_f1"], 3), {k: round(out[k]["expected_delay_h"], 1) for k in out})
    res["ablation"] = abl

# --- E5: sensitivity / stress tests of the simulator assumptions -----------------------------
xgbm = XGBStageModel.load("artifacts/xgb_stage.json")
rule_nom = {k: (ev["Tuned threshold rule"][k]["threshold"], ev["Tuned threshold rule"][k]["persistence"]) for k in LEVELS}
conds = {"Nominal (new seeds)": {}, "Sensor noise x2": {"noise_mult": 2.0}, "Sensor noise x3": {"noise_mult": 3.0},
         "Heating CO make halved": {"co_gen_mult": 0.5}, "Halved CO make + noise x2": {"co_gen_mult": 0.5, "noise_mult": 2.0},
         "Frequent telemetry dropouts (80% of panels)": {"dropout_prob": 0.8}}
sens = {}
for i, (cname, kw) in enumerate(conds.items()):
    sc = SimConfig(n_episodes=150, seed=9000 + i, **kw)
    d2 = generate_dataset(sc)
    d2["split"] = "test"
    F2, _ = prepare(d2)
    S = make_split(d2, F2, "test")
    S.P["xgb"] = xgbm.predict_proba(S.F)
    row = {}
    cont = baseline_stages(S.sub, "continuous")
    for k in (1, 2):
        mo = evaluate_alarm_level(S.ep, S.ty, S.st, alarm_from_score(S.ep, prob_at_least(S.P["xgb"], k), pol.thresholds[k], pol.persistence[k]), k)
        th, m = rule_nom[k]
        ru = evaluate_alarm_level(S.ep, S.ty, S.st, alarm_from_score(S.ep, S.F[rule["feature"][k]].to_numpy(float), th, m), k)
        cb = evaluate_alarm_level(S.ep, S.ty, S.st, cont >= k, k)
        row[f"L{k}"] = {"xgb": summarize(mo), "rule": summarize(ru), "continuous": summarize(cb)}
    sens[cname] = row
    log("sensitivity", cname, {k: (round(v["xgb"]["detection_rate"], 2), round(v["xgb"]["median_delay_h"], 1), round(v["xgb"]["fa_per_100d"], 2)) for k, v in row.items()})
res["sensitivity"] = sens

# --- E6: named edge-case scenarios ------------------------------------------------------------
scen, pk["scenarios"] = {}, {}
for name, desc in SCENARIOS.items():
    sen, lat, spec = build_scenario(name)
    Fs = build_features(sen)
    Ps = xgbm.predict_proba(Fs)
    ep = np.zeros(len(sen), int)
    ty = np.array([name] * len(sen))
    st = lat["stage"].to_numpy().astype(int)
    cont = continuous_threshold_stage(sen)
    row = {"description": desc, "max_stage": int(st.max())}
    alarms = {}
    for k in LEVELS:
        al = alarm_from_score(ep, prob_at_least(Ps, k), pol.thresholds[k], pol.persistence[k])
        r_m = evaluate_alarm_level(ep, ty, st, al, k)
        r_b = evaluate_alarm_level(ep, ty, st, cont >= k, k)
        first = lambda arr: (float(np.argmax(arr) / 96.0) if arr[192:].any() else None)
        alarms[k] = al
        row[f"L{k}"] = {"stage_onset_day": (float(np.argmax(st >= k) / 96.0) if (st >= k).any() else None),
                        "model_first_alarm_day": first(np.where(np.arange(len(al)) >= 192, al, False)),
                        "baseline_first_alarm_day": first(np.where(np.arange(len(al)) >= 192, cont >= k, False)),
                        "model_false_alarms": r_m["false_alarms"], "baseline_false_alarms": r_b["false_alarms"]}
    scen[name] = row
    pk["scenarios"][name] = {"sensors": sen, "latent": lat, "P": Ps, "alarms": alarms, "baseline": cont}
    log("scenario", name, {k: row[k]["model_first_alarm_day"] for k in ("L1", "L2", "L3")})
res["scenarios"] = scen

# --- E7: TreeSHAP ------------------------------------------------------------------------------
rng = np.random.default_rng(0)
vm = np.where(T.valid)[0]
per_class = []
for c in range(4):
    idx = vm[T.st[vm] == c]
    per_class.append(rng.choice(idx, min(1200, len(idx)), replace=False))
sel = np.concatenate(per_class)
Xs = T.F.iloc[sel]
contrib = xgbm.contributions(Xs)                         # (n, 4, F+1)
imp = {STAGES[c]: dict(sorted(zip(xgbm.feature_names, np.abs(contrib[:, c, :-1]).mean(0).tolist()),
                              key=lambda kv: -kv[1])[:15]) for c in range(4)}
res["shap_top_features"] = imp
pk["shap"] = {"X": Xs, "contrib": contrib.astype(np.float32), "y": T.st[sel]}
log("shap done")

# --- persist -------------------------------------------------------------------------------------
res["model_meta"] = {"feature_names": xgbm.feature_names, "stages": STAGES, "alert_policy": pol.to_dict(),
                     "fa_budget_per_100_panel_days": BUDGET}
json.dump(res["model_meta"], open("artifacts/model_meta.json", "w"), indent=2)
save_json(res, "reports/results.json")
pk["T"] = {"ep": T.ep, "ty": T.ty, "st": T.st, "P": T.P, "sub": T.sub[["episode_id", "step", "t_days", "co_ppm", "co2_pct", "o2_pct", "ch4_pct", "coal_temp_c"]],
           "F": T.F[["co_med_4h", "make_med_4h", "graham_12h_med", "o2_def", "co_make", "co_above_floor"]]}
pk["handbook_stage"] = {n: d["stage"] for n, d in hb.items()}
pk["policy"] = pol
with open("reports/cache/eval.pkl", "wb") as fh:
    pickle.dump(pk, fh)
log("done")
