"""Deep sequence models (1D-CNN, LSTM) trained directly on raw sensor windows.

Requires PyTorch (`pip install -e .[dl]`). Inputs are the last `WINDOW` samples (48 h) of
transformed sensor channels; the output is a distribution over the four self-heating stages.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..constants import WARMUP_STEPS
from ..indices import co_make_lpm, o2_deficiency

WINDOW = WARMUP_STEPS
SEQ_CHANNELS = ["log_co", "co2", "o2", "ch4", "temp", "log_q", "baro", "o2def", "log_make",
                "hour_sin", "hour_cos", "gap"]


def sequence_channels(ep: pd.DataFrame) -> np.ndarray:
    """(T, C) float32 array of transformed channels for one contiguous series (NaN-safe)."""
    g = ep[["co_ppm", "co2_pct", "o2_pct", "ch4_pct", "temp_c", "airflow_m3s", "baro_hpa"]].astype(float)
    gap = g[["co_ppm", "co2_pct", "o2_pct", "ch4_pct"]].isna().any(axis=1).to_numpy().astype(float)
    g = g.ffill(limit=16)
    co, co2, o2, ch4, tc, q, p = (g[c].to_numpy() for c in g.columns)
    out = np.stack([
        np.log1p(np.maximum(co, 0)), co2, o2, ch4, tc, np.log(np.maximum(q, 0.1)), p - 1000.0,
        o2_deficiency(o2, co2, ch4, co), np.log1p(np.maximum(co_make_lpm(co, q), 0)),
        np.sin(2 * np.pi * ep["hour"].to_numpy() / 24.0), np.cos(2 * np.pi * ep["hour"].to_numpy() / 24.0), gap,
    ], axis=1)
    return out.astype(np.float32)


class SeqData:
    """Episodes stacked as (n_ep, T, C) with per-channel normalisation fitted on the training split."""

    def __init__(self, df: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.ep_ids, arrays, labels, splits = [], [], [], []
        for eid, ep in df.groupby("episode_id", sort=False, observed=True):
            arrays.append(sequence_channels(ep.reset_index(drop=True)))
            labels.append(ep["stage"].to_numpy().astype(np.int64))
            splits.append(str(ep["split"].iloc[0]))
            self.ep_ids.append(int(eid))
        self.X = np.stack(arrays)
        self.y = np.stack(labels)
        self.splits = np.array(splits)
        if mean is None:
            tr = self.X[self.splits == "train"].reshape(-1, self.X.shape[-1])
            mean, std = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0) + 1e-6
            mean[-1], std[-1] = 0.0, 1.0
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.X = np.nan_to_num((self.X - self.mean) / self.std, nan=0.0).astype(np.float32)


def _build_nets():
    import torch
    from torch import nn

    class CNN1D(nn.Module):
        def __init__(self, c_in: int, n_classes: int = 4):
            super().__init__()
            def block(i, o, k, s):
                return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
            self.net = nn.Sequential(block(c_in, 32, 7, 1), block(32, 48, 5, 2), block(48, 64, 5, 2), block(64, 64, 3, 2))
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))

        def forward(self, x):                      # x: (B, T, C)
            h = self.net(x.transpose(1, 2))
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))

    class LSTMNet(nn.Module):
        def __init__(self, c_in: int, n_classes: int = 4, hidden: int = 64):
            super().__init__()
            self.pool = nn.AvgPool1d(4)             # 15-min -> hourly
            self.lstm = nn.LSTM(c_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))

        def forward(self, x):
            z = self.pool(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.lstm(z)
            return self.head(out[:, -1])

    return CNN1D, LSTMNet


class SequenceStageModel:
    def __init__(self, arch: str = "cnn", epochs: int = 6, windows_per_epoch: int = 60_000, batch: int = 256,
                 lr: float = 2e-3, seed: int = 0):
        self.arch, self.epochs, self.wpe, self.batch, self.lr, self.seed = arch, epochs, windows_per_epoch, batch, lr, seed
        self.net = None
        self.history: list[dict] = []
        self.mean = self.std = None

    def _new_net(self, c_in: int):
        CNN1D, LSTMNet = _build_nets()
        return CNN1D(c_in) if self.arch == "cnn" else LSTMNet(c_in)

    @staticmethod
    def _gather(X: np.ndarray, ep_idx: np.ndarray, ends: np.ndarray):
        import torch
        offs = np.arange(-WINDOW + 1, 1)
        return torch.from_numpy(X[ep_idx[:, None], ends[:, None] + offs[None, :]])

    def fit(self, data: SeqData, class_weights: np.ndarray, log=print):
        import torch
        from sklearn.metrics import f1_score
        from torch import nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        rng = np.random.default_rng(self.seed)
        self.mean, self.std = data.mean, data.std
        self.net = self._new_net(data.X.shape[-1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * (self.wpe // self.batch + 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        tr_eps, va_eps = np.where(data.splits == "train")[0], np.where(data.splits == "val")[0]
        T = data.X.shape[1]
        best, best_state = -1.0, None
        for ep in range(self.epochs):
            t0 = time.time()
            self.net.train()
            e_idx = rng.choice(tr_eps, self.wpe)
            ends = rng.integers(WINDOW, T, self.wpe)           # steps >= WARMUP are the evaluated region
            perm = rng.permutation(self.wpe)
            tot = 0.0
            for s in range(0, self.wpe, self.batch):
                b = perm[s:s + self.batch]
                xb = self._gather(data.X, e_idx[b], ends[b])
                yb = torch.from_numpy(data.y[e_idx[b], ends[b]])
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(b)
            ve = va_eps[:: max(1, len(va_eps) // 30)]
            P = self.predict_proba_episodes(data, ve, stride=8)
            yv = np.concatenate([data.y[i, WINDOW::8] for i in ve])
            f1 = f1_score(yv, P.argmax(1), average="macro")
            self.history.append({"epoch": ep + 1, "train_loss": tot / self.wpe, "val_macro_f1": float(f1),
                                 "seconds": time.time() - t0})
            log(f"  [{self.arch}] epoch {ep + 1}/{self.epochs} loss={tot / self.wpe:.4f} val_macroF1={f1:.4f} ({time.time() - t0:.0f}s)")
            if f1 > best:
                best, best_state = f1, {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        return self

    def predict_proba_episodes(self, data: SeqData, ep_idx: np.ndarray, stride: int = 1) -> np.ndarray:
        """Probabilities for steps WINDOW, WINDOW+stride, ... of each requested episode, concatenated."""
        import torch
        self.net.eval()
        T = data.X.shape[1]
        outs = []
        with torch.no_grad():
            for i in ep_idx:
                ends = np.arange(WINDOW, T, stride)
                for s in range(0, len(ends), 512):
                    e = ends[s:s + 512]
                    xb = self._gather(data.X, np.full(len(e), i), e)
                    outs.append(torch.softmax(self.net(xb), dim=1).numpy())
        return np.concatenate(outs)

    def predict_full(self, data: SeqData, ep_idx: np.ndarray) -> np.ndarray:
        """(n_ep * T, 4) probabilities, with pre-warm-up rows set to 'Normal'."""
        T = data.X.shape[1]
        P = self.predict_proba_episodes(data, ep_idx, 1).reshape(len(ep_idx), T - WINDOW, 4)
        full = np.zeros((len(ep_idx), T, 4), dtype=np.float32)
        full[:, :, 0] = 1.0
        full[:, WINDOW:, :] = P
        return full.reshape(-1, 4)

    def save(self, path: str):
        import torch
        torch.save({"arch": self.arch, "state": self.net.state_dict(), "mean": self.mean, "std": self.std}, path)
