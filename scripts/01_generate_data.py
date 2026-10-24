"""Step 1: simulate the dataset and build rolling-window features (cached to data/processed)."""
import argparse
import time

from ews.config import load_config, sim_config
from ews.experiment import load_or_build

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
t = time.time()
cfg = load_config(a.config)
df, F = load_or_build(sim_config(cfg), a.out, force=a.force)
print(f"rows={len(df):,} episodes={df.episode_id.nunique()} features={F.shape[1]} ({time.time() - t:.0f}s)")
print(df.groupby("split", observed=True).episode_id.nunique().to_dict())
print((df.stage.value_counts(normalize=True).sort_index().round(3)).to_dict())
