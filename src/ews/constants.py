"""Domain constants shared by simulator, features, baselines and serving.

All gas-index thresholds are *indicative literature values* and are configurable.
Replace them with the trigger levels of your mine's TARP / DGMS-approved scheme
before any real use (see docs/ASSUMPTIONS.md).
"""
from __future__ import annotations

STEP_MIN = 15                      # sampling interval of the monitoring system (minutes)
STEPS_PER_HOUR = 60 // STEP_MIN
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
WARMUP_STEPS = 192                 # 48 h of history needed before the models give output

STAGES = ["Normal", "Incubation", "Active Heating", "Fire"]
# Latent coal temperature (deg C) edges separating the four stages.
STAGE_T_EDGES = (40.0, 80.0, 200.0)

SENSOR_COLS = ["co_ppm", "co2_pct", "o2_pct", "ch4_pct", "temp_c", "airflow_m3s", "baro_hpa"]
GAS_COLS = ["co_ppm", "co2_pct", "o2_pct", "ch4_pct"]

# Fixed-threshold rules used as the *status-quo baseline* (CO ppm, Graham's ratio).
BASELINE_THRESHOLDS = {
    1: {"co_ppm": 15.0, "graham": 0.5},    # Incubation alert
    2: {"co_ppm": 50.0, "graham": 1.0},    # Active-heating alert
    3: {"co_ppm": 400.0, "graham": 3.0},   # Fire alert
}
MIN_O2_DEF_FOR_GRAHAM = 0.2   # % — below this, Graham's ratio is numerically meaningless

FRESH_AIR_O2 = 20.93
FRESH_AIR_CO2 = 0.04
