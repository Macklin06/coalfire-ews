"""One simulated return-airway monitoring episode: latent heating -> gases -> noisy sensors.

Pipeline
--------
1. latent coal temperature from the thermal balance model (`thermal.py`)
2. heating gas make from semi-empirical temperature curves (Graham's ratio, O2 deficiency,
   Young's ratio) -> diluted by return airflow, delayed and low-pass filtered (transport)
3. confounders common in real mines: ventilation changes, barometric pumping, shot-firing
   fumes, diesel equipment, strata CO2/CH4 emission
4. sensor effects: noise, drift, quantisation, spikes, dropouts and frozen readings

The curves are calibrated to *typical published magnitudes* (see docs/ASSUMPTIONS.md); they are
NOT fitted to a particular mine. That is the main limitation of the study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..constants import FRESH_AIR_CO2, FRESH_AIR_O2, STAGE_T_EDGES
from .thermal import ThermalParams, critical_r, simulate_excess_temperature

EPISODE_TYPES = ["normal", "benign", "stable_warm", "slow_runaway", "runaway", "fast_runaway", "quench"]

# Temperature -> Graham's ratio (%) and Young's ratio curves (typical literature magnitudes).
_GR_T = np.array([30, 50, 70, 90, 120, 160, 220, 300, 600], float)
_GR_V = np.array([0.15, 0.30, 0.60, 1.00, 1.80, 3.50, 8.0, 20.0, 30.0])
_Y_T = np.array([30, 100, 200, 300, 600], float)
_Y_V = np.array([0.25, 0.40, 0.70, 0.85, 0.90])


@dataclass
class EpisodeSpec:
    ep_type: str
    seed: int
    n_days: float
    step_min: float
    t_amb: float
    thermal: ThermalParams
    q0: float                      # nominal return airflow (m3/s)
    ch4_base: float                # % in return
    co2_strata: float              # % strata CO2 in return
    co_bg: float                   # background CO (ppm) from diesel / strata
    do2_base: float                # % O2 deficiency generated at ambient (oxidation baseline)
    k_o2: float                    # 1/degC growth of O2 consumption with temperature
    tau_mix_min: float             # transport mixing time constant
    delay_steps: int               # transport delay
    start_hour: float
    noise: dict                    # per-channel noise sigma
    drift: dict                    # per-channel drift (random-walk sigma per sqrt(day))
    blast_hours: list              # nominal shot-firing hours of day
    blast_prob: float
    vent_event_rate: float         # events per day
    dropouts: list = field(default_factory=list)   # (i0, i1, [channels])
    frozen: list = field(default_factory=list)     # (i0, i1, channel)
    co_gen_mult: float = 1.0       # sensitivity knob: scales heating-CO make
    vent_events: list = field(default_factory=list)  # forced events (start_day, dur_day, factor)


def _ou(n: int, dt_days: float, tau_days: float, sigma: float, rng) -> np.ndarray:
    a = math.exp(-dt_days / tau_days)
    e = rng.normal(0.0, sigma * math.sqrt(1 - a * a), n)
    return lfilter([1.0], [1.0, -a], e)


def sample_spec(ep_type: str, seed: int, n_days: float = 20.0, step_min: float = 15.0,
                noise_mult: float = 1.0, dropout_prob: float = 0.3, blast_prob: float = 0.8,
                co_gen_mult: float = 1.0) -> EpisodeSpec:
    """Draw a random episode specification for a given episode type."""
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.045, 0.058)
    r_c = critical_r(b)
    r_lo = 0.2 * r_c
    t_start = rng.uniform(1.0, 2.5)
    tau = rng.uniform(2.5, 5.0)
    quench_peak = None

    def crossing(t_c, cap_mult):
        r_cap = cap_mult * r_c
        return r_cap, (t_c - t_start) * (r_cap - r_lo) / (r_c - r_lo)

    if ep_type == "normal":
        r_cap, t_ramp = r_lo * rng.uniform(0.9, 1.8), rng.uniform(3, 10)
    elif ep_type == "benign":
        r_cap, t_ramp = rng.uniform(0.3, 0.65) * r_c, rng.uniform(4, 12)
    elif ep_type == "stable_warm":
        r_cap, t_ramp = rng.uniform(0.95, 0.995) * r_c, rng.uniform(4, 12)
    elif ep_type == "slow_runaway":
        r_cap, t_ramp = crossing(rng.uniform(3.5, 8), rng.uniform(1.1, 1.4))
    elif ep_type == "runaway":
        r_cap, t_ramp = crossing(rng.uniform(3, 7), rng.uniform(1.4, 2.5))
    elif ep_type == "fast_runaway":
        tau = rng.uniform(1.5, 2.5)
        r_cap, t_ramp = crossing(rng.uniform(2.5, 5), rng.uniform(2.5, 4.0))
    elif ep_type == "quench":
        r_cap, t_ramp = crossing(rng.uniform(3, 7), rng.uniform(1.3, 2.0))
        quench_peak = rng.uniform(30.0, 130.0)
    else:
        raise ValueError(f"unknown episode type {ep_type!r}")

    thermal = ThermalParams(b=b, tau_days=tau, r_lo=r_lo, r_cap=r_cap, t_ramp_days=t_ramp,
                            t_start_days=t_start, quench_peak=quench_peak)
    nm = noise_mult
    noise = {"co_ppm": rng.uniform(0.4, 0.9) * nm, "co2_pct": rng.uniform(0.006, 0.012) * nm,
             "o2_pct": rng.uniform(0.02, 0.045) * nm, "ch4_pct": rng.uniform(0.01, 0.03) * nm,
             "temp_c": 0.15 * nm, "airflow_m3s": 0.015 * nm, "baro_hpa": 0.15}
    drift = {"co_ppm": 0.3 * nm, "co2_pct": 0.003 * nm, "o2_pct": 0.01 * nm, "ch4_pct": 0.005 * nm}

    n = int(round(n_days * 24 * 60 / step_min))
    dropouts, frozen = [], []
    if rng.random() < dropout_prob:
        for _ in range(rng.integers(1, 3)):
            dur = int(rng.uniform(0.2, 2.0) * 24 * 60 / step_min)
            i0 = int(rng.integers(int(3 * 24 * 60 / step_min), max(n - dur, int(3 * 24 * 60 / step_min) + 1)))
            chans = (["co_ppm", "co2_pct", "o2_pct", "ch4_pct"] if rng.random() < 0.7
                     else [str(rng.choice(["co_ppm", "co2_pct", "o2_pct", "ch4_pct"]))])
            dropouts.append((i0, min(i0 + dur, n), chans))
    if rng.random() < 0.08:
        dur = int(rng.uniform(0.3, 2.0) * 24 * 60 / step_min)
        i0 = int(rng.integers(int(3 * 24 * 60 / step_min), max(n - dur, int(3 * 24 * 60 / step_min) + 1)))
        frozen.append((i0, min(i0 + dur, n), str(rng.choice(["co_ppm", "o2_pct"]))))

    return EpisodeSpec(
        ep_type=ep_type, seed=seed, n_days=n_days, step_min=step_min,
        t_amb=rng.uniform(24.0, 32.0), thermal=thermal, q0=rng.uniform(15.0, 60.0),
        ch4_base=rng.uniform(0.05, 0.6), co2_strata=rng.uniform(0.03, 0.2),
        co_bg=rng.uniform(0.3, 2.0), do2_base=rng.uniform(0.03, 0.08), k_o2=0.03,
        tau_mix_min=rng.uniform(30, 75), delay_steps=int(rng.integers(1, 4)),
        start_hour=rng.uniform(0, 24), noise=noise, drift=drift,
        blast_hours=[6.0, 14.0, 22.0], blast_prob=blast_prob, vent_event_rate=0.15,
        dropouts=dropouts, frozen=frozen, co_gen_mult=co_gen_mult)


def stage_from_temperature(t_coal: np.ndarray) -> np.ndarray:
    return np.digitize(t_coal, STAGE_T_EDGES).astype(np.int8)


def simulate_episode(spec: EpisodeSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sensors, latent) DataFrames, one row per sampling step."""
    rng = np.random.default_rng(spec.seed + 10_000)
    n = int(round(spec.n_days * 24 * 60 / spec.step_min))
    dt_days = spec.step_min / (60 * 24)
    t_days = np.arange(n) * dt_days
    hour = (spec.start_hour + t_days * 24.0) % 24.0

    # 1. latent coal temperature ---------------------------------------------------------
    x, quenched = simulate_excess_temperature(spec.thermal, n, spec.step_min, rng)
    t_coal = spec.t_amb + x
    stage = stage_from_temperature(t_coal)

    # 2. ventilation, barometric pressure ------------------------------------------------
    log_q = np.zeros(n)
    events = list(spec.vent_events)
    for _ in range(rng.poisson(spec.vent_event_rate * spec.n_days)):
        events.append((rng.uniform(0, spec.n_days), rng.uniform(0.3, 3.0), rng.uniform(0.7, 1.3)))
    ramp = 0.5 / 24.0  # 30 min changeover
    for s, d, f in events:
        up = np.clip((t_days - s) / ramp, 0, 1)
        down = np.clip((t_days - (s + d)) / ramp, 0, 1)
        log_q += math.log(f) * (up - down)
    log_q += 0.04 * np.sin(2 * math.pi * (hour - 9) / 24.0) + _ou(n, dt_days, 0.5, 0.01, rng)
    q = spec.q0 * np.exp(log_q)
    dil = spec.q0 / q                                      # dilution factor (>1 when airflow drops)

    p = 1000.0 + _ou(n, dt_days, 1.5, 5.0, rng) + 0.8 * np.sin(2 * math.pi * t_days / 0.5)
    dpdt_h = np.gradient(p, dt_days * 24.0)
    dpdt_h = np.convolve(dpdt_h, np.ones(4) / 4, mode="same")
    pump = np.clip(np.exp(-0.12 * dpdt_h), 0.7, 1.5)

    # 3. heating gas make ---------------------------------------------------------------
    d_t = np.maximum(t_coal - spec.t_amb, 0.0)
    do2_raw = spec.do2_base * np.exp(spec.k_o2 * d_t)
    do2_heat = 9.0 * np.tanh(do2_raw / 9.0)                          # saturating at fire
    gr = np.exp(np.interp(t_coal, _GR_T, np.log(_GR_V))) * spec.co_gen_mult   # Graham's ratio (%)
    co_heat = gr * do2_heat * 100.0                                  # ppm (before dilution)
    yr = np.interp(t_coal, _Y_T, _Y_V)
    co2_heat = yr * do2_heat                                         # % excess CO2

    def transport(sig):
        sig = np.concatenate([np.full(spec.delay_steps, sig[0]), sig])[:n]
        a = math.exp(-spec.step_min / spec.tau_mix_min)
        return lfilter([1 - a], [1.0, -a], sig, zi=[sig[0] * a])[0]

    flush = pump ** 0.5
    co_h = transport(co_heat) * dil * flush
    do2_h = transport(do2_heat) * dil * flush
    co2_h = transport(co2_heat) * dil * flush

    # 4. confounders -----------------------------------------------------------------------
    co_blast = np.zeros(n)
    co2_blast = np.zeros(n)
    n_days_int = int(math.ceil(spec.n_days)) + 1
    for day in range(-1, n_days_int):
        for bh in spec.blast_hours:
            prob = spec.blast_prob if bh == 14.0 else spec.blast_prob * 0.35
            if rng.random() > prob:
                continue
            t0_hour = day * 24.0 + bh + rng.normal(0, 0.4)
            t0 = (t0_hour - spec.start_hour) / 24.0
            if t0 < -0.5 or t0 > spec.n_days:
                continue
            peak = rng.uniform(15, 120)
            tau = rng.uniform(1.0, 3.0) / 24.0
            dt = t_days - t0
            shape = np.where(dt >= 0, np.exp(-np.maximum(dt, 0) / tau), 0.0) * (dt > -dt_days * 0.5)
            co_blast += peak * shape
            co2_blast += peak * 1.2e-3 * shape
    co_blast *= dil
    co2_blast *= dil

    co_diesel = np.zeros(n)
    for _ in range(rng.poisson(1.5 * spec.n_days)):
        t0, dur, amp = rng.uniform(0, spec.n_days), rng.uniform(1, 3) / 24, rng.uniform(1, 6)
        co_diesel += amp * ((t_days >= t0) & (t_days < t0 + dur))
    co_diesel = np.convolve(co_diesel, np.ones(3) / 3, mode="same") * dil

    co_bg = spec.co_bg * dil * (1 + _ou(n, dt_days, 2.0, 0.08, rng))
    ch4 = spec.ch4_base * pump * dil * (1 + _ou(n, dt_days, 3.0, 0.06, rng))
    co2_str = spec.co2_strata * dil * (1 + _ou(n, dt_days, 3.0, 0.05, rng)) * pump ** 0.3

    # 5. gas composition ---------------------------------------------------------------------
    co_true = co_h + co_blast + co_diesel + co_bg
    m = ch4 / 100.0
    o2_true = FRESH_AIR_O2 * (1 - m) - do2_h - 0.005 * (co_blast / 50.0)
    co2_true = FRESH_AIR_CO2 * (1 - m) + co2_str + co2_h + co2_blast + 0.0004 * co_diesel
    air_t_true = 24.0 + 0.012 * d_t * dil + 0.4 * np.sin(2 * math.pi * (hour - 15) / 24) \
        + _ou(n, dt_days, 5.0, 0.3, rng)

    # 6. sensor effects --------------------------------------------------------------------
    def drift_walk(sigma):
        w = np.cumsum(rng.normal(0, sigma * math.sqrt(dt_days), n))
        return w

    nz = spec.noise
    co_m = co_true + np.clip(drift_walk(spec.drift["co_ppm"]), -3, 3) \
        + rng.normal(0, nz["co_ppm"], n) + co_true * rng.normal(0, 0.02, n)
    co2_m = co2_true + np.clip(drift_walk(spec.drift["co2_pct"]), -0.03, 0.03) + rng.normal(0, nz["co2_pct"], n)
    o2_m = o2_true + np.clip(drift_walk(spec.drift["o2_pct"]), -0.08, 0.08) + rng.normal(0, nz["o2_pct"], n)
    ch4_m = ch4 + np.clip(drift_walk(spec.drift["ch4_pct"]), -0.05, 0.05) + rng.normal(0, nz["ch4_pct"], n)
    temp_m = air_t_true + rng.normal(0, nz["temp_c"], n)
    q_m = q * (1 + rng.normal(0, nz["airflow_m3s"], n))
    p_m = p + rng.normal(0, nz["baro_hpa"], n)

    for arr, amp, scale in ((co_m, 20.0, 1.0), (ch4_m, 0.05, 1.0), (o2_m, 0.08, 1.0)):   # electrical spikes
        idx = np.where(rng.random(n) < 0.0006)[0]
        arr[idx] += rng.choice([-1, 1], len(idx)) * rng.uniform(0.3, 1.0, len(idx)) * amp * scale

    co_m = np.round(np.maximum(co_m, 0.0))                 # 1 ppm resolution, no negatives
    co2_m = np.round(np.maximum(co2_m, 0.0), 2)
    o2_m = np.round(o2_m, 2)
    ch4_m = np.round(np.maximum(ch4_m, 0.0), 2)

    sensors = pd.DataFrame({
        "co_ppm": co_m, "co2_pct": co2_m, "o2_pct": o2_m, "ch4_pct": ch4_m,
        "temp_c": np.round(temp_m, 2), "airflow_m3s": np.round(q_m, 2), "baro_hpa": np.round(p_m, 2)})

    for i0, i1, chans in spec.frozen:
        for c in chans if isinstance(chans, list) else [chans]:
            sensors.loc[i0:i1 - 1, c] = sensors.loc[i0, c]
    for i0, i1, chans in spec.dropouts:
        sensors.loc[i0:i1 - 1, chans] = np.nan

    sensors.insert(0, "hour", np.round(hour, 4))
    sensors.insert(0, "t_days", np.round(t_days, 5))
    sensors.insert(0, "step", np.arange(n, dtype=np.int32))

    latent = pd.DataFrame({
        "coal_temp_c": t_coal.astype(np.float32), "stage": stage,
        "graham_true": gr.astype(np.float32), "quenched": quenched,
        "co_true_ppm": co_true.astype(np.float32), "o2_def_true": do2_h.astype(np.float32)})
    return sensors, latent
