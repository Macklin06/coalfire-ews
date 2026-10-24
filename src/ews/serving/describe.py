"""Human-readable descriptions of engineered features for explained alerts."""
DESCRIPTIONS = {
    "co_ppm": "current CO concentration", "co_make": "CO make rate (CO x airflow)",
    "co_med_4h": "4-h median CO", "co_med_12h": "12-h median CO", "co_mean_1h": "1-h mean CO",
    "co_mean_4h": "4-h mean CO", "co_mean_12h": "12-h mean CO", "co_mean_24h": "24-h mean CO",
    "make_med_4h": "4-h median CO make rate", "make_med_12h": "12-h median CO make rate",
    "make_mean_24h": "24-h mean CO make rate", "make_min_24h": "24-h minimum CO make (baseline floor)",
    "co_above_floor": "CO above its 24-h floor", "co_spikiness": "short CO spike (blast-like signature)",
    "make_slope_4h": "short-term CO-make trend", "make_slope_12h": "12-h CO-make trend",
    "co_slope_12h": "12-h CO trend", "make_ratio_24h": "CO make relative to its 24-h floor",
    "co_log_slope_24h": "exponential CO growth rate", "o2_def": "oxygen deficiency",
    "o2def_mean_4h": "4-h mean oxygen deficiency", "o2def_mean_24h": "24-h mean oxygen deficiency",
    "o2def_slope_12h": "oxygen-deficiency trend", "graham_4h": "Graham's ratio (4-h)",
    "graham_12h_med": "Graham's ratio (12-h median)", "young_4h": "Young's ratio (4-h)",
    "cor_4h": "CO/CO2 ratio (4-h)", "cor_12h": "CO/CO2 ratio (12-h)", "co2_excess": "CO2 above its 24-h floor",
    "co2_slope_12h": "CO2 trend", "airflow": "return airflow", "airflow_rel_24h": "airflow vs 24-h mean",
    "airflow_chg_4h": "airflow change (4 h)", "baro_chg_4h": "barometric pressure change (4 h)",
    "ch4_pct": "methane", "ch4_mean_4h": "4-h mean methane", "temp_c": "air temperature",
    "temp_chg_24h": "air-temperature change (24 h)", "missing_frac_24h": "share of missing gas data (24 h)",
}


def describe(name: str) -> str:
    return DESCRIPTIONS.get(name, name.replace("_", " "))
