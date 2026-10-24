"""Build and execute notebooks/01_end_to_end.ipynb (narrative + live code)."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

M("""# Spontaneous-combustion early-warning system — end-to-end notebook
**MI 302 – Mine Hazards, Rescue and Recovery** · NITK Surathkal

Goal: detect the stage of coal self-heating (Normal → Incubation → Active heating → Fire) from gas-sensor streams
*earlier* than fixed gas-ratio thresholds, without more false alarms.

Heavy steps (simulation, model training, full evaluation) are run by `scripts/01…05` and cached; this notebook
re-runs the key steps live and reads the cached results. **All data is simulated — see `docs/ASSUMPTIONS.md`.**""")
C("""
import json, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25})
R = json.load(open("../reports/results.json"))
df = pd.read_parquet("../data/processed/sensors.parquet")
print(df.shape, "rows;", df.episode_id.nunique(), "episodes")
""")
M("""## 1. Background: what the gases tell us
* Coal oxidises in air; heat builds up where ventilation is too weak to remove it but oxygen still reaches the coal.
* Oxidation consumes O₂ and releases CO and CO₂. CO grows much faster than O₂ consumption as temperature rises.
* **Oxygen deficiency** = 0.265·N₂ − O₂ · **Graham's ratio** = 100·CO% / O₂ deficiency% · **Young's ratio** = excess CO₂ / O₂ deficiency · **CO make** = CO × airflow.""")
M("## 2. Data: a physics-informed simulator\nA hidden coal temperature (Semenov heat balance) drives gas generation; shot-firing, diesel, ventilation changes, barometric pumping and sensor faults are added on top.")
C("""
from ews.simulator.episode import sample_spec, simulate_episode
spec = sample_spec("runaway", 11)
sen, lat = simulate_episode(spec)
fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
ax[0].plot(sen.t_days, lat.coal_temp_c); ax[0].set_yscale("log"); ax[0].set_ylabel("hidden coal T (°C)")
ax[1].plot(sen.t_days, sen.co_ppm, lw=.6); ax[1].set_ylabel("CO (ppm)")
ax[2].plot(sen.t_days, sen.airflow_m3s, lw=.6, color="g"); ax[2].set_ylabel("airflow (m³/s)"); ax[2].set_xlabel("day")
plt.show()
""")
C("""
print(pd.DataFrame(R["dataset"]["episodes_by_type"], index=["episodes"]).T.T)
print("share of rows per stage:", {k: round(v, 3) for k, v in R["dataset"]["stage_share_rows"].items()})
display(Image("../reports/figures/fig01_example_episode.png")); display(Image("../reports/figures/fig02_confounders.png"))
""")
M("""## 3. Features
55 *causal* rolling-window features per time step: oxygen deficiency, Graham / Young / CO-CO₂ ratios, CO make, 1–24 h means, medians, minima ("floor"), slopes, ventilation and pressure context, missing-data share.
Medians and minima ignore 1–3 h blasting spikes but still rise during real heating.""")
C("""
from ews.features import build_features, FEATURE_NAMES
F = build_features(sen)
print(len(FEATURE_NAMES), "features, e.g.:", FEATURE_NAMES[:12])
F[["co_ppm","co_med_4h","co_min_24h","co_above_floor","graham_12h_med"]].iloc[::96].round(2).head(12)
""")
M("## 4. Models and stage classification (unseen test panels)")
C("""
rows = []
for n, v in R["stage"].items():
    rows.append({"model": n, "macro-F1": v["macro_f1"], "quad. kappa": v["qw_kappa"], "mean |stage error|": v["mean_abs_stage_error"],
                 **{f"recall {s}": v["per_class"][s]["recall"] for s in v["per_class"]}})
pd.DataFrame(rows).set_index("model").round(3)
""")
C("""display(Image("../reports/figures/fig03_stage_models.png")); display(Image("../reports/figures/fig04_confusion_xgb.png"))""")
M("""## 5. The metric that matters: lead time at a matched false-alarm rate
Stage accuracy is not the goal — *warning earlier* is. For each alert level k (stage ≥ k) an alarm fires when P(stage ≥ k) stays above a threshold for m consecutive samples. We pick threshold and m on the **validation** panels so that false alarms ≤ 1 per 100 panel-days, then measure delay on the **test** panels. Every method (including the tuned threshold rule) gets the same budget.""")
C("""
from ews.experiment import make_split, load_probs, calibrate_and_evaluate, summarize
import pandas as pd
F_all = pd.read_parquet("../data/processed/features.parquet")
V, T = make_split(df, F_all, "val"), make_split(df, F_all, "test")
for s in (V, T): s.P["xgb"] = load_probs("xgb", s.name, root=__import__("pathlib").Path("../reports/cache"))
live = calibrate_and_evaluate({"kind": "model", "name": "xgb"}, V, T, 1.0, levels=(2,))
print("LIVE: XGBoost level-2 alert ->", {k: round(v, 2) for k, v in summarize(live[2]).items()})
""")
C("""
ev = R["events"]; rows = []
for name, per in ev.items():
    for k in ("1", "2", "3"):
        r = per[k]; rows.append({"method": name, "level": k, "detection rate": r["detection_rate"], "median delay (h)": r["median_delay_h"],
                                 "false alarms /100 d": r["fa_per_100d"], "events": r["n_events"]})
tab = pd.DataFrame(rows); tab.pivot(index="method", columns="level", values=["median delay (h)", "false alarms /100 d", "detection rate"]).round(2)
""")
C("""display(Image("../reports/figures/fig05_detection_delay.png")); display(Image("../reports/figures/fig06_operating_curve.png"))""")
M("**Paired lead time** — hours by which XGBoost's alarm precedes each baseline's alarm on the *same* events:")
C("""
pl = R["paired_lead"]["xgb"]; rows = []
for b, per in pl.items():
    for k, r in per.items(): rows.append({"vs baseline": b, "level": k, "median lead (h)": r["median_lead_h"], "paired events": r["n_paired"],
                                          "events only ML caught": r["model_only_detected"], "events only baseline caught": r["baseline_only_detected"]})
pd.DataFrame(rows).round(1)
""")
M("## 6. Explainability (TreeSHAP)")
C("""display(Image("../reports/figures/fig07_shap_importance.png"))
import os
if os.path.exists("../reports/figures/fig07b_shap_beeswarm.png"): display(Image("../reports/figures/fig07b_shap_beeswarm.png"))""")
C("""
from ews.models.tabular import XGBStageModel
from ews.serving.describe import describe
m = XGBStageModel.load("../artifacts/xgb_stage.json")
i = int(np.where((T.st == 2) & T.valid)[0][50])                       # an Active-heating row
contrib = m.contributions(T.F.iloc[[i]])[0][2, :-1]
top = np.argsort(-np.abs(contrib))[:6]
pd.DataFrame({"feature": [describe(m.feature_names[j]) for j in top], "value": T.F.iloc[i, top].values.round(2), "SHAP (log-odds)": contrib[top].round(2)})
""")
M("## 7. Ablation: which parts of the feature set matter?")
C("""display(Image("../reports/figures/fig08_ablation.png"))""")
M("## 8. Stress tests and edge cases\nWe re-simulate with more sensor noise, weaker heating CO, and heavy telemetry dropout **without retraining**, and run named scenarios (fast runaway, slow incubation, dropout during heating, ventilation shock, blasting storm, self-quenching).")
C("""display(Image("../reports/figures/fig09_sensitivity.png")); display(Image("../reports/figures/fig10_scenarios.png"))
pd.DataFrame({k: {"L2 onset (d)": v["L2"]["stage_onset_day"], "ML alarm (d)": v["L2"]["model_first_alarm_day"], "fixed-threshold alarm (d)": v["L2"]["baseline_first_alarm_day"],
                  "ML false alarms (L1-3)": sum(v[f"L{j}"]["model_false_alarms"] for j in (1,2,3)),
                  "fixed false alarms (L1-3)": sum(v[f"L{j}"]["baseline_false_alarms"] for j in (1,2,3))} for k, v in R["scenarios"].items()}).T""")
M("""## 9. Conclusions and limitations
See the report for the narrative. Key caveats: simulated data; Level-1 (incubation) detection is slow; performance on real mines is unknown until the simulator is calibrated or the model is validated on real gas logs.""")
nbf.write(nb, "notebooks/01_end_to_end.ipynb")
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
nbf.write(nb, "notebooks/01_end_to_end.ipynb")
print("notebook executed")
