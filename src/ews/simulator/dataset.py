"""Generate a labelled multi-episode dataset (many simulated panels) and split it by episode."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import EPISODE_TYPES, sample_spec, simulate_episode

DEFAULT_MIX = {"normal": 0.30, "benign": 0.15, "stable_warm": 0.10, "slow_runaway": 0.13,
               "runaway": 0.17, "fast_runaway": 0.05, "quench": 0.10}


@dataclass
class SimConfig:
    n_episodes: int = 600
    n_days: float = 20.0
    step_min: float = 15.0
    seed: int = 2026
    type_mix: dict = field(default_factory=lambda: dict(DEFAULT_MIX))
    noise_mult: float = 1.0
    dropout_prob: float = 0.3
    blast_prob: float = 0.8
    co_gen_mult: float = 1.0
    split_fracs: tuple = (0.60, 0.15, 0.25)   # train / val / test, stratified by episode type


def _assign_types(cfg: SimConfig) -> list[str]:
    counts = {t: int(round(cfg.n_episodes * cfg.type_mix.get(t, 0.0))) for t in EPISODE_TYPES}
    types = [t for t, c in counts.items() for _ in range(c)]
    while len(types) < cfg.n_episodes:
        types.append("normal")
    return types[: cfg.n_episodes]


def _assign_splits(types: list[str], fracs, rng) -> list[str]:
    splits = np.empty(len(types), dtype=object)
    for t in set(types):
        idx = np.where(np.array(types) == t)[0]
        rng.shuffle(idx)
        n_tr, n_va = int(round(fracs[0] * len(idx))), int(round(fracs[1] * len(idx)))
        splits[idx[:n_tr]] = "train"
        splits[idx[n_tr:n_tr + n_va]] = "val"
        splits[idx[n_tr + n_va:]] = "test"
    return list(splits)


def generate_dataset(cfg: SimConfig, out_dir: str | Path | None = None, verbose: bool = False) -> pd.DataFrame:
    """Simulate `cfg.n_episodes` panels. Returns one long DataFrame (sensors + latent truth)."""
    rng = np.random.default_rng(cfg.seed)
    types = _assign_types(cfg)
    splits = _assign_splits(types, cfg.split_fracs, rng)
    seeds = rng.integers(0, 2**31 - 1, size=cfg.n_episodes)
    frames = []
    for i, (ty, sp, sd) in enumerate(zip(types, splits, seeds)):
        spec = sample_spec(ty, int(sd), cfg.n_days, cfg.step_min, cfg.noise_mult,
                           cfg.dropout_prob, cfg.blast_prob, cfg.co_gen_mult)
        sensors, latent = simulate_episode(spec)
        df = pd.concat([sensors, latent], axis=1)
        df.insert(0, "episode_id", np.int32(i))
        df["ep_type"] = ty
        df["split"] = sp
        frames.append(df)
        if verbose and (i + 1) % 100 == 0:
            print(f"  simulated {i + 1}/{cfg.n_episodes}")
    data = pd.concat(frames, ignore_index=True)
    data["ep_type"] = data["ep_type"].astype("category")
    data["split"] = data["split"].astype("category")
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out / "sensors.parquet", index=False)
        pd.Series(asdict(cfg)).to_json(out / "sim_config.json")
    return data


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path) / "sensors.parquet")
