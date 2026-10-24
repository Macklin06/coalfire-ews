"""FastAPI service: /predict for a window of readings, /stream/{scenario} for the live dashboard."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..constants import SENSOR_COLS, WARMUP_STEPS
from ..simulator.scenarios import SCENARIOS, build_scenario
from .engine import Monitor, Predictor, reading_from_row

app = FastAPI(title="Coal self-heating early-warning system", version="1.0")
_pred: Predictor | None = None


def predictor() -> Predictor:
    global _pred
    if _pred is None:
        _pred = Predictor()
    return _pred


class Reading(BaseModel):
    co_ppm: float | None = None
    co2_pct: float | None = None
    o2_pct: float | None = None
    ch4_pct: float | None = None
    temp_c: float
    airflow_m3s: float
    baro_hpa: float
    hour: float = 0.0          # hour of day 0-24 (used for the shot-firing time pattern)


class Window(BaseModel):
    readings: list[Reading]    # oldest first, one per 15 minutes, at least 192 (48 h)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scenarios")
def scenarios():
    return SCENARIOS


@app.post("/predict")
def predict(w: Window):
    if len(w.readings) < WARMUP_STEPS:
        raise HTTPException(422, f"need at least {WARMUP_STEPS} readings (48 h), got {len(w.readings)}")
    df = pd.DataFrame([r.model_dump() for r in w.readings]).astype(float)
    return predictor().predict_window(df)


@app.get("/stream/{scenario}")
async def stream(scenario: str, speed: float = 15.0, seed: int | None = None):
    if scenario not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    sen, lat, _ = build_scenario(scenario, seed)
    mon = Monitor(predictor())
    mon.prime([reading_from_row(sen.iloc[i]) for i in range(WARMUP_STEPS - 1)])

    async def gen():
        for i in range(WARMUP_STEPS - 1, len(sen)):
            row = sen.iloc[i]
            pred = mon.push(reading_from_row(row))
            payload = {"i": i, "day": float(row.t_days),
                       "reading": {c: (None if pd.isna(row[c]) else float(row[c])) for c in SENSOR_COLS},
                       "pred": pred,
                       "truth": {"coal_temp_c": float(lat.coal_temp_c.iloc[i]), "stage": int(lat.stage.iloc[i])}}
            yield f"data: {json.dumps(payload, default=float)}\n\n"
            await asyncio.sleep(1.0 / max(speed, 0.1))
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "static" / "index.html").read_text()
