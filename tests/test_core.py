import numpy as np
import pandas as pd

from ews.constants import SENSOR_COLS
from ews.features import BUFFER_STEPS, build_features
from ews.indices import graham_ratio, o2_deficiency
from ews.metrics import evaluate_alarm_level
from ews.simulator.episode import sample_spec, simulate_episode


def test_o2_deficiency_fresh_air_is_near_zero():
    assert abs(o2_deficiency(20.93, 0.04, 0.0, 0.0)) < 0.05


def test_graham_ratio_formula():
    # CO = 100 ppm = 0.01 %, O2 deficiency = 1 % -> Graham = 100*0.01/1 = 1 %
    assert np.isclose(graham_ratio(100.0, 1.0), 1.0)
    assert np.isnan(graham_ratio(100.0, 0.05))        # deficiency too small to be meaningful


def test_runaway_reaches_fire_and_normal_does_not():
    _, lat_r = simulate_episode(sample_spec("fast_runaway", 1))
    _, lat_n = simulate_episode(sample_spec("normal", 1))
    assert lat_r.stage.max() == 3 and lat_n.stage.max() == 0


def test_simulator_is_deterministic():
    a, _ = simulate_episode(sample_spec("runaway", 5))
    b, _ = simulate_episode(sample_spec("runaway", 5))
    pd.testing.assert_frame_equal(a, b)


def test_online_features_equal_offline_features():
    sen, _ = simulate_episode(sample_spec("runaway", 3))
    off = build_features(sen)
    for i in (400, 1000, 1500):
        on = build_features(sen.iloc[i - BUFFER_STEPS + 1:i + 1].reset_index(drop=True)).iloc[-1]
        assert np.allclose(on.fillna(0), off.iloc[i].fillna(0), atol=1e-4)


def test_alarm_metrics_toy_case():
    n = 400
    stage = np.zeros(n, int); stage[300:] = 2
    alarm = np.zeros(n, bool); alarm[290:] = True          # alarm 10 steps (2.5 h) early
    r = evaluate_alarm_level(np.zeros(n, int), np.array(["x"] * n), stage, alarm, 2, warmup=0)
    assert r["detection_rate"] == 1.0 and np.isclose(r["median_delay_h"], -2.5) and r["false_alarms"] == 0
    alarm2 = np.zeros(n, bool); alarm2[50:60] = True        # spurious alarm, no event follows
    r2 = evaluate_alarm_level(np.zeros(n, int), np.array(["x"] * n), stage, alarm2, 2, warmup=0)
    assert r2["false_alarms"] == 1 and r2["detection_rate"] == 0.0
