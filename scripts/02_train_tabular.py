"""Step 2: train logistic regression, random forest and XGBoost on engineered features."""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from ews.config import load_config, sim_config
from ews.experiment import load_or_build, save_probs
from ews.models.tabular import XGBStageModel, make_logreg, make_random_forest
from ews.pipeline import prepare, subset

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--data", default="data/processed")
a = ap.parse_args()
cfg = load_config(a.config)
df, F = load_or_build(sim_config(cfg), a.data)
meta = df[["episode_id", "step", "stage", "ep_type", "split"]].copy()
meta["valid"] = meta["step"] >= 192
tr = cfg["training"]

timings = {}
models = {}
Xtr, ytr, _ = subset(F, meta, "train", tr["tabular_stride"])
Xva, yva, _ = subset(F, meta, "val", tr["tabular_stride"])
t = time.time()
xgbm = XGBStageModel().fit(Xtr, ytr, Xva, yva)
timings["xgb"] = time.time() - t
print(f"XGBoost: {timings['xgb']:.0f}s, best iteration {xgbm.booster.best_iteration}")
xgbm.save("artifacts/xgb_stage.json")
models["xgb"] = xgbm

Xr, yr, _ = subset(F, meta, "train", tr["rf_stride"])
t = time.time()
rf = make_random_forest().fit(Xr, yr)
timings["rf"] = time.time() - t
print(f"Random forest: {timings['rf']:.0f}s")
models["rf"] = rf

Xl, yl, _ = subset(F, meta, "train", tr["logreg_stride"])
t = time.time()
lr = make_logreg().fit(Xl, yl)
timings["logreg"] = time.time() - t
print(f"Logistic regression: {timings['logreg']:.0f}s")
models["logreg"] = lr

for name, m in models.items():
    for split in ("val", "test"):
        mk = (df["split"].to_numpy() == split)
        P = m.predict_proba(F[mk])
        save_probs(name, split, P)
json.dump(timings, open("reports/cache/train_seconds_tabular.json", "w"))
joblib.dump({"rf_n_trees": 150}, "reports/cache/rf_meta.joblib")
print("probabilities cached.")
