"""Classical mine-gas indices used for spontaneous-combustion interpretation."""
from __future__ import annotations

import numpy as np

from .constants import MIN_O2_DEF_FOR_GRAHAM


def nitrogen_pct(o2, co2, ch4, co_ppm):
    """Nitrogen by difference (%), ignoring minor gases."""
    return 100.0 - o2 - co2 - ch4 - co_ppm * 1e-4


def o2_deficiency(o2, co2, ch4, co_ppm):
    """Oxygen deficiency (%) = 0.265 * N2 - O2 (fresh-air N2/O2 ratio is 3.77).

    Using N2 by difference makes the figure insensitive to dilution by methane.
    """
    return 0.265 * nitrogen_pct(o2, co2, ch4, co_ppm) - o2


def graham_ratio(co_ppm, o2_def, min_def: float = MIN_O2_DEF_FOR_GRAHAM):
    """Graham's ratio (%) = 100 * CO(%) / O2 deficiency(%). NaN when deficiency is tiny."""
    co_pct = np.asarray(co_ppm, dtype=float) * 1e-4
    d = np.asarray(o2_def, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gr = 100.0 * co_pct / d
    return np.where(d >= min_def, gr, np.nan)


def young_ratio(co2_excess, o2_def, min_def: float = MIN_O2_DEF_FOR_GRAHAM):
    """Young's ratio = excess CO2 / O2 deficiency. NaN when deficiency is tiny."""
    with np.errstate(divide="ignore", invalid="ignore"):
        yr = np.asarray(co2_excess, dtype=float) / np.asarray(o2_def, dtype=float)
    return np.where(np.asarray(o2_def) >= min_def, yr, np.nan)


def carbon_oxides_ratio(co_ppm, co2_pct):
    """CO / CO2 (both as %)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(co_ppm, dtype=float) * 1e-4) / np.maximum(np.asarray(co2_pct, dtype=float), 1e-3)


def co_make_lpm(co_ppm, airflow_m3s):
    """CO make rate in litres/minute = CO(ppm) * 1e-6 * Q(m3/s) * 1000 L/m3 * 60 s/min."""
    return np.asarray(co_ppm, dtype=float) * np.asarray(airflow_m3s, dtype=float) * 0.06
