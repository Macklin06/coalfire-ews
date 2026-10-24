import os

import pytest
from fastapi.testclient import TestClient

from ews.simulator.scenarios import build_scenario

pytestmark = pytest.mark.skipif(not os.path.exists("artifacts/xgb_stage.json"), reason="train the model first")


def test_health_and_predict():
    from ews.serving.app import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.post("/predict", json={"readings": []}).status_code == 422
    sen, _, _ = build_scenario("runaway")
    rows = sen.iloc[:300][["co_ppm", "co2_pct", "o2_pct", "ch4_pct", "temp_c", "airflow_m3s", "baro_hpa", "hour"]]
    r = c.post("/predict", json={"readings": rows.to_dict("records")})
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] in ["Normal", "Incubation", "Active Heating", "Fire"] and len(body["explanation"]) == 5
