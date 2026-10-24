# How to explain this project (viva / interview cheat-sheet)

**One sentence.** We simulate how gases in a return airway change as a coal pocket heats up, then train
ML models to tell the heating stage from noisy gas readings — and show they warn earlier, with fewer
false alarms, than fixed gas-ratio thresholds.

## Likely questions
**Why simulate?** No public real data; mines don't share it. We made the simulator physics-based and
state the limitation openly (see `ASSUMPTIONS.md`).

**How do you know the label is right?** In the simulator we know the hidden coal temperature, so the stage label is
ground truth. In a real mine you never know it — that is why real validation needs lab or mine data.

**What is Graham's ratio?** 100 × CO(%) / oxygen deficiency(%). It measures CO produced per oxygen consumed;
it rises with temperature and is (roughly) independent of dilution by airflow — that's why it is used.

**Why not just use thresholds?** (1) Shot-firing and diesel produce CO spikes → false alarms; (2) thresholds
see only the current value, not the trend; (3) at low temperature the signals are tiny.

**Why features like "24-h minimum" and "median"?** Blast spikes last 1-3 h; a median or minimum ignores them
but still rises when the coal is really heating. That is how the model separates blasting from heating.

**Why split by episode?** Rows within a panel are strongly correlated. Splitting rows would leak the same panel
into train and test and inflate accuracy.

**What is "lead time"?** Hours between the alarm and the moment the true stage is reached (negative = alarm
before). We compare methods at the *same false-alarm rate*, otherwise a method that alarms constantly looks "earlier".

**Why XGBoost over LSTM?** Similar performance, much cheaper, and exact SHAP explanations. The sequence models are a
check that deep learning does not give a big gain here (small data, strong engineered features).

**What does SHAP show?** Which features push the model towards "Active heating": e.g. CO make above its 24-h floor, rising
CO trend, Graham's ratio. These agree with mining practice, which supports the model.

**What would you do with real data?** Calibrate the simulator curves to lab data, fine-tune on historical gas logs,
validate on past heating incidents, set thresholds with the mine's TARP.

## Resume bullets (edit numbers from `reports/results.json`)
* Built a physics-informed simulator of coal self-heating (Semenov heat balance → gas make → noisy sensors) generating 600 panel-episodes (1.15 M readings).
* Engineered 55 causal rolling-window features (Graham/Young ratios, CO make rate, spike-robust trends); XGBoost, RF, 1D-CNN and LSTM compared with episode-level splits.
* Introduced a lead-time-at-matched-false-alarm-rate metric; ML raised active-heating alerts **X h** earlier than fixed thresholds at the same false-alarm rate (fill from results).
* Explained alerts with TreeSHAP; deployed with FastAPI and a live Chart.js dashboard; feature code verified identical offline vs streaming by unit tests.
