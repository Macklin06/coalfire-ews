"""Step 5: draw all report figures from reports/results.json and reports/cache/eval.pkl."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ews.constants import STAGES
from ews.experiment import MODEL_LABELS
from ews.indices import graham_ratio, o2_deficiency

OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
R = json.load(open("reports/results.json")); PK = pickle.load(open("reports/cache/eval.pkl", "rb"))
df = pd.read_parquet("data/processed/sensors.parquet")
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
SC = ["#2ea043", "#d4a72c", "#f0883e", "#f85149"]
def save(name): plt.tight_layout(); plt.savefig(OUT / name, bbox_inches="tight"); plt.close()
LAB = {**MODEL_LABELS, "Tuned threshold rule": "Tuned threshold rule"}

# 1. example runaway episode with stage shading
cand = df[(df.split == "test") & (df.ep_type == "runaway")].groupby("episode_id").stage.max()
tid = cand[cand == 3].index[1]
e = df[df.episode_id == tid].reset_index(drop=True)
od = o2_deficiency(e.o2_pct, e.co2_pct, e.ch4_pct, e.co_ppm); gr = graham_ratio(e.co_ppm, od)
fig, ax = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
for a, (y, l) in zip(ax, [(e.coal_temp_c, "Hidden coal temp (°C)"), (e.co_ppm, "CO (ppm)"), (od, "O2 deficiency (%)"),
                          (pd.Series(gr).clip(upper=6), "Graham's ratio (%)"), (e.co2_pct, "CO2 (%)")]):
    a.plot(e.t_days, y, lw=.8); a.set_ylabel(l)
    for k in range(4):
        m = e.stage.values == k
        if m.any(): a.fill_between(e.t_days, 0, 1, where=m, color=SC[k], alpha=.12, transform=a.get_xaxis_transform())
ax[0].set_yscale("log"); ax[1].set_yscale("symlog", linthresh=10); ax[-1].set_xlabel("day"); ax[0].set_title("Simulated runaway episode (shading = true stage: green Normal → red Fire)")
save("fig01_example_episode.png")

# 2. confounders
tid = df[(df.ep_type == "normal") & (df.split == "test")].episode_id.iloc[3]
e = df[df.episode_id == tid].reset_index(drop=True)
fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
ax[0].plot(e.t_days, e.co_ppm, lw=.7); ax[0].axhline(15, color="r", ls="--", lw=.8, label="fixed Level-1 CO trigger (15 ppm)")
ax[0].set_ylabel("CO (ppm)"); ax[0].legend(); ax[0].set_title("A panel with NO heating: shot-firing spikes and ventilation changes")
ax[1].plot(e.t_days, e.airflow_m3s, color="g", lw=.7); ax[1].set_ylabel("airflow (m³/s)"); ax[1].set_xlabel("day")
save("fig02_confounders.png")

# 3. stage metrics
names = [n for n in R["stage"]]
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(names))
ax.bar(x - .2, [R["stage"][n]["macro_f1"] for n in names], .4, label="macro-F1")
ax.bar(x + .2, [R["stage"][n]["qw_kappa"] for n in names], .4, label="quadratic κ")
ax.set_xticks(x); ax.set_xticklabels([LAB.get(n, n) for n in names], rotation=20, ha="right"); ax.legend(); ax.set_ylim(0, 1)
ax.set_title("Stage classification on unseen test panels"); save("fig03_stage_models.png")

# 4. confusion matrix XGB
cm = np.array(R["stage"]["xgb"]["confusion"], float); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.6, 4)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4): ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="w" if cmn[i, j] > .5 else "k")
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(STAGES, rotation=25, ha="right"); ax.set_yticklabels(STAGES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("XGBoost confusion (row-normalised)"); ax.grid(False); save("fig04_confusion_xgb.png")

# 5. detection delay distributions
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for a, k in zip(axs, (1, 2)):
    meths = ["Manual 8-hourly sampling", "Continuous fixed threshold", "Tuned threshold rule", "xgb", "cnn", "lstm"]
    data = [PK["events"][m][k].query("detected").delay_h.values for m in meths]
    a.boxplot(data, vert=False, showfliers=False); a.set_yticks(range(1, len(meths) + 1)); a.set_yticklabels([LAB.get(m, m) for m in meths])
    a.axvline(0, color="k", lw=.8); a.set_xlabel("detection delay vs. true stage onset (h; negative = early)")
    a.set_title(["", "Level 1 (Incubation)", "Level 2 (Active heating)"][k])
fig.suptitle("Detection delay of detected events (ML/tuned rules calibrated to ≤1 false alarm per 100 panel-days)", fontsize=9)
save("fig05_detection_delay.png")

# 6. operating curves
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for a, k in zip(axs, ("1", "2")):
    for n, c in R["operating_curves"].items():
        b = sorted(c, key=float); a.plot([float(x) for x in b], [c[x][k]["expected_delay_h"] for x in b], "o-", label=LAB.get(n, n))
    a.set_xscale("log"); a.set_xlabel("false-alarm budget (per 100 panel-days)"); a.set_ylabel("mean delay, misses = 168 h (h)")
    a.set_title(["", "Level 1", "Level 2"][int(k)])
axs[0].legend(); save("fig06_operating_curve.png")

# 7. SHAP
S = PK["shap"]; names = list(S["X"].columns); c2 = S["contrib"][:, 2, :-1]
imp = np.abs(c2).mean(0); o = np.argsort(-imp)[:12][::-1]
fig, ax = plt.subplots(figsize=(6, 4)); ax.barh([names[i] for i in o], imp[o], color="#f0883e"); ax.set_xlabel("mean |SHAP| (log-odds)")
ax.set_title("What drives the 'Active heating' score"); save("fig07_shap_importance.png")
try:
    import shap
    shap.summary_plot(c2, S["X"], max_display=12, show=False); plt.title("SHAP summary: Active-heating class"); save("fig07b_shap_beeswarm.png")
except Exception as ex:
    print("beeswarm skipped:", ex)

# 8. ablation
if "ablation" in R:
    ab = R["ablation"]; n = list(ab)
    fig, ax = plt.subplots(figsize=(8, 3.6)); y = np.arange(len(n))
    ax.barh(y - .2, [ab[k]["L1"]["expected_delay_h"] for k in n], .4, label="Level 1"); ax.barh(y + .2, [ab[k]["L2"]["expected_delay_h"] for k in n], .4, label="Level 2")
    ax.set_yticks(y); ax.set_yticklabels(n); ax.invert_yaxis(); ax.set_xlabel("mean detection delay incl. misses (h) — lower is better"); ax.legend()
    ax.set_title("Feature ablation (XGBoost, 1 FA / 100 panel-days)"); save("fig08_ablation.png")

# 9. sensitivity
sens = R["sensitivity"]; cn = list(sens); fig, axs = plt.subplots(1, 3, figsize=(12, 3.6)); w = .27
for a, (met, lab) in zip(axs, [("detection_rate", "L2 detection rate"), ("median_delay_h", "L2 median delay (h)"), ("fa_per_100d", "L2 false alarms / 100 d")]):
    for j, (m, nm) in enumerate([("xgb", "XGBoost"), ("rule", "Tuned rule"), ("continuous", "Fixed threshold")]):
        a.bar(np.arange(len(cn)) + (j - 1) * w, [sens[c]["L2"][m][met] for c in cn], w, label=nm)
    a.set_xticks(range(len(cn))); a.set_xticklabels([c.replace(" (new seeds)", "") for c in cn], rotation=35, ha="right", fontsize=7); a.set_title(lab)
axs[0].legend(fontsize=7); save("fig09_sensitivity.png")

# 10. edge-case scenarios
sc = PK["scenarios"]; fig, axs = plt.subplots(len(sc), 1, figsize=(9, 2.1 * len(sc)), sharex=True)
for a, (n, d) in zip(axs, sc.items()):
    s, l, P = d["sensors"], d["latent"], d["P"]; a.plot(s.t_days, s.co_ppm.clip(upper=300), lw=.6, color="#58a6ff"); a.set_ylabel(n.replace("_", "\n"), fontsize=7)
    a2 = a.twinx(); a2.plot(s.t_days, P[:, 2:].sum(1), color="#f0883e", lw=.9); a2.set_ylim(0, 1.05); a2.grid(False)
    al = d["alarms"][2]
    if al.any(): a.axvline(s.t_days[np.argmax(al)], color="#f0883e", ls="--")
    bl = d["baseline"] >= 2
    if bl[192:].any(): a.axvline(s.t_days[192 + np.argmax(bl[192:])], color="r", ls=":")
    on = np.argmax(l.stage.values >= 2) if (l.stage.values >= 2).any() else None
    if on: a.axvline(s.t_days[on], color="k", lw=.8)
axs[-1].set_xlabel("day"); axs[0].set_title("Edge cases: blue CO (ppm), orange P(≥Active); lines: ML L2 alarm (dashed), fixed-threshold L2 (dotted), true onset (black)", fontsize=8)
save("fig10_scenarios.png")
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
