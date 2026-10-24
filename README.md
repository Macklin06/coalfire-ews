# Coal Self-Heating Early-Warning System (MI 302 course project)

Machine-learning early warning for **spontaneous combustion in underground coal mines**: classify the stage of self-heating
(*Normal → Incubation → Active Heating → Fire*) from gas-sensor streams (CO, CO₂, O₂, CH₄ + airflow, pressure, temperature)
and warn **earlier than fixed gas-ratio thresholds, without drowning operators in false alarms**.

> **Important:** no public real-mine dataset exists, so all data are produced by a physics-informed **simulator**.
> Results show the method works *under our assumptions*; they are not proof of real-mine performance.
> Read [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) first.

## Headline results (150 unseen simulated test panels, ≈2,700 panel-days)

| Level-2 (active heating, coal ≥ 80 °C) alert | Detected | Median delay vs true onset | False alarms / 100 panel-days |
|---|---|---|---|
| Handbook: manual 8-hourly sampling | 95% | 6.5 h | 8.7 |
| Handbook: continuous fixed threshold | 97% | 3.6 h | 59.3 |
| Data-tuned threshold rule | 92% | 14.4 h | 0.33 |
| **XGBoost** | **100%** | **-4.5 h** | **1.7** |
| 1D-CNN / LSTM | 100% / 100% | -4.0 / -0.8 h | 1.3 / 1.3 |

(negative delay = warned before the coal reached the stage). Median lead of XGBoost over handbook sampling / continuous thresholds:
**8.8 h / 6.8 h**, and over the tuned rule **17.6 h**.
Stage macro-F1: XGBoost 0.835 vs fixed thresholds 0.564.

**Honest negatives:** early *incubation* (40–80 °C) is hard to detect from gases alone (median ≈ 2.8 days after onset);
Level-1 false alarms still occur in extreme ventilation/blasting scenarios; airflow/pressure context features added nothing in the simulator.
Full details: `report/Final_Project_Report.docx`, `reports/results.json`, `notebooks/01_end_to_end.ipynb`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dl,dev]"          # drop ,dl if you do not want PyTorch (CNN/LSTM steps are then skipped)

make all                             # = scripts 01-03 (simulate, train)  (~10 min on a laptop CPU)
python scripts/04_evaluate.py        # events, ablations, stress tests, SHAP (~15 min)
python scripts/05_make_figures.py
pytest -q                            # unit tests

uvicorn ews.serving.app:app --port 8000   # open http://localhost:8000  -> live dashboard
```
Trained XGBoost model and alert policy are included in `artifacts/`, so the API and dashboard work immediately after `pip install -e .`.
The notebook needs `data/processed/` and `reports/cache/`, which are created by the scripts above (they are git-ignored because they are large).

## How it works (one paragraph per step)
1. **Simulator** (`src/ews/simulator`): a hidden coal temperature follows a Semenov-type heat balance. The stage label comes from this temperature (not from gas thresholds). Gases are generated from temperature, diluted by airflow, and disturbed by blasting fumes, diesel, ventilation changes, pressure pumping and sensor faults.
2. **Features** (`features.py`): 55 causal rolling-window features: oxygen deficiency, Graham / Young / CO-CO₂ ratios, CO make, spike-robust medians & minima, slopes. The same code runs offline and in streaming (tested).
3. **Models** (`models/`): logistic regression, random forest, XGBoost on features; 1D-CNN and LSTM on raw 48-h windows.
4. **Alert logic** (`alerting.py`, `metrics.py`): alarm when P(stage ≥ k) stays above a threshold for m samples; threshold and m are tuned on validation panels to a false-alarm budget (1 per 100 panel-days) for *every* method, then tested.
5. **Explanations**: exact TreeSHAP from XGBoost, shown in plain words on every alert.
6. **Service** (`serving/`): FastAPI `/predict`, `/stream/{scenario}` and a Chart.js dashboard; fail-safe hold of the alert level when telemetry is lost.

## Repository layout
```
configs/default.yaml     experiment settings          scripts/01..05_*.py   pipeline steps
src/ews/                 package (simulator, features, models, metrics, serving)
artifacts/               trained XGBoost + alert policy (small)
reports/                 results.json + figures       report/                 final report (docx) + generator
notebooks/               executed end-to-end notebook docs/                   assumptions, viva cheat-sheet
tests/                   unit + API tests
```

## Using real data (next step)
Provide a table with columns `co_ppm, co2_pct, o2_pct, ch4_pct, temp_c, airflow_m3s, baro_hpa, hour` at 15-minute intervals
and call `Predictor().predict_window(df)` (≥192 rows). Retrain/calibrate on real incidents before trusting any alert, and replace the
handbook trigger levels in `constants.py` with your mine's TARP values.

## Authors
Macklin Chriss Miranda · Ganesh M. · Mounesh Gowda — NITK Surathkal, Mining Engineering, MI 302 (2026-27).
