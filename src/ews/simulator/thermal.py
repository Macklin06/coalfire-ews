"""Latent coal-temperature dynamics (Semenov-type thermal balance).

Dimensionless excess temperature x = T_coal - T_ambient obeys

    dx/dtau = G(r(t) * exp(b x)) - k_loss * x ,       G(R) = R / (1 + R / G_max)

where  r = Q0/h  is the (scaled) heat-generation potential of the oxidising coal pocket,
b the Arrhenius-linearised temperature sensitivity, and k_loss the heat-loss multiplier
(1 normally, larger after sealing / water / inertisation).  For r below the critical value

    r_c = 1 / (b e)

a stable warm equilibrium exists (benign warming); once r(t) rises past r_c the equilibrium
vanishes through a saddle-node bifurcation and the pocket "runs away" — after a long,
slow bottleneck (incubation).  This reproduces the days-to-weeks incubation followed by
rapid acceleration described for coal self-heating.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def critical_r(b: float) -> float:
    return 1.0 / (b * math.e)


@dataclass
class ThermalParams:
    b: float                 # 1/degC
    tau_days: float          # thermal time-constant of the pocket
    r_lo: float              # heat-generation potential at start (degC)
    r_cap: float             # final heat-generation potential (degC)
    t_ramp_days: float       # duration of the ramp from r_lo to r_cap
    t_start_days: float      # when the ramp starts
    g_max: float = 600.0     # saturation of generation (fuel / oxygen limitation)
    quench_peak: float | None = None   # if set: sealing starts when x reaches this value
    quench_loss_mult: float = 6.0
    quench_r_factor: float = 0.15
    r_noise_sigma: float = 0.03


def simulate_excess_temperature(p: ThermalParams, n_steps: int, step_min: float,
                                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the thermal balance with explicit Euler. Returns (x, quenched_flag)."""
    dt_days = step_min / (60.0 * 24.0)
    dtau = dt_days / p.tau_days
    t_days = np.arange(n_steps) * dt_days
    ramp = np.clip((t_days - p.t_start_days) / max(p.t_ramp_days, 1e-6), 0.0, 1.0)
    r_t = p.r_lo + (p.r_cap - p.r_lo) * ramp

    # slow multiplicative OU noise on r (ventilation / packing variations of the pocket)
    eps = np.zeros(n_steps)
    a = math.exp(-dt_days / 1.5)
    shocks = rng.normal(0.0, p.r_noise_sigma * math.sqrt(1 - a * a), n_steps)
    for i in range(1, n_steps):
        eps[i] = a * eps[i - 1] + shocks[i]
    r_t = r_t * (1.0 + eps)

    x = np.zeros(n_steps)
    quenched = np.zeros(n_steps, dtype=bool)
    xi = float(p.r_lo)  # start near the warm equilibrium
    b, gmax = p.b, p.g_max
    loss = 1.0
    r_factor = 1.0
    q_on = False
    for i in range(n_steps):
        if (not q_on) and p.quench_peak is not None and xi >= p.quench_peak:
            q_on = True
            loss = p.quench_loss_mult
            r_factor = p.quench_r_factor
        R = r_t[i] * r_factor * math.exp(min(b * xi, 30.0))
        gen = R / (1.0 + R / gmax)
        xi = xi + dtau * (gen - loss * xi)
        xi = min(max(xi, 0.0), 650.0)
        x[i] = xi
        quenched[i] = q_on
    return x, quenched
