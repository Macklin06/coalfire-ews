"""Named edge-case scenarios used for validation and for the live dashboard demo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import STEPS_PER_DAY
from .episode import EpisodeSpec, sample_spec, simulate_episode

SCENARIOS: dict[str, str] = {
    "runaway": "Typical runaway: slow incubation for days, then accelerating heating and fire.",
    "fast_runaway": "Fast runaway: short incubation (2-3 days), rapid transition to fire.",
    "slow_incubation": "Slow incubation: stays near the critical point for many days (hard to detect).",
    "dropout_during_heating": "Gas telemetry lost for ~2 days spanning the incubation -> active-heating transition.",
    "ventilation_shock": "No heating, but airflow is halved for 3 days (gas concentrations double).",
    "blasting_storm": "No heating, but heavy shot-firing with large CO spikes every shift.",
    "self_quenching": "Heating that is sealed/cooled after reaching ~100 C (should alarm, then recover).",
}

_SEEDS = {"runaway": 11, "fast_runaway": 5, "slow_incubation": 21, "dropout_during_heating": 11,
          "ventilation_shock": 7, "blasting_storm": 9, "self_quenching": 41}


def _first_step(stage: np.ndarray, k: int) -> int | None:
    idx = np.flatnonzero(stage >= k)
    return int(idx[0]) if len(idx) else None


def build_scenario(name: str, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, EpisodeSpec]:
    """Return (sensors, latent, spec) for a named scenario (deterministic for a given seed)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    sd = _SEEDS[name] if seed is None else seed
    if name == "runaway":
        spec = sample_spec("runaway", sd, dropout_prob=0.0)
    elif name == "fast_runaway":
        spec = sample_spec("fast_runaway", sd, dropout_prob=0.0)
    elif name == "slow_incubation":
        spec = sample_spec("slow_runaway", sd, dropout_prob=0.0)
    elif name == "self_quenching":
        spec = sample_spec("quench", sd, dropout_prob=0.0)
    elif name == "dropout_during_heating":
        spec = sample_spec("runaway", sd, dropout_prob=0.0)
        _, lat = simulate_episode(spec)
        on = _first_step(lat["stage"].to_numpy(), 2) or int(12 * STEPS_PER_DAY)
        spec.dropouts = [(max(on - STEPS_PER_DAY, 0), on + STEPS_PER_DAY, ["co_ppm", "co2_pct", "o2_pct", "ch4_pct"])]
    elif name == "ventilation_shock":
        spec = sample_spec("normal", sd, dropout_prob=0.0)
        spec.vent_events = [(8.0, 3.0, 0.5)]
        spec.vent_event_rate = 0.0
    elif name == "blasting_storm":
        spec = sample_spec("normal", sd, dropout_prob=0.0, blast_prob=1.0)
        spec.blast_prob = 1.0
        spec.blast_hours = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0]
    sensors, latent = simulate_episode(spec)
    return sensors, latent, spec
