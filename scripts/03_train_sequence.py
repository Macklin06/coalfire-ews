"""Step 3: train the 1D-CNN and LSTM on raw sensor windows (requires PyTorch)."""
import argparse
import json
import time

import numpy as np

from ews.config import load_config, sim_config
from ews.experiment import load_or_build, save_probs
from ews.models.sequence import SeqData, SequenceStageModel
from ews.models.tabular import sqrt_balanced_weights

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--data", default="data/processed")
ap.add_argument("--archs", default="cnn,lstm")
a = ap.parse_args()
cfg = load_config(a.config)
df, _ = load_or_build(sim_config(cfg), a.data)
t = time.time()
data = SeqData(df)
print(f"sequence tensor {data.X.shape} built in {time.time() - t:.0f}s")
y_tr = data.y[data.splits == "train"][:, 192:].ravel()
w = np.bincount(y_tr, minlength=4).astype(float)
cw = ((len(y_tr) / (4 * np.maximum(w, 1))) ** 0.5)
import os
hist = json.load(open("reports/cache/train_history_sequence.json")) if os.path.exists("reports/cache/train_history_sequence.json") else {}
for arch in a.archs.split(","):
    t = time.time()
    m = SequenceStageModel(arch, epochs=cfg["training"]["seq_epochs"],
                           windows_per_epoch=cfg["training"]["seq_windows_per_epoch"]).fit(data, cw)
    hist[arch] = {"history": m.history, "train_seconds": time.time() - t}
    m.save(f"artifacts/{arch}_stage.pt")
    for split in ("val", "test"):
        idx = np.where(data.splits == split)[0]
        P = m.predict_full(data, idx)
        save_probs(arch, split, P)
    print(f"{arch} done in {time.time() - t:.0f}s")
json.dump(hist, open("reports/cache/train_history_sequence.json", "w"), indent=1)
