# Assumptions and limitations (read this before presenting)

## 1. Why the data is synthetic
Public datasets of continuous multi-gas telemetry from a mine panel that went through self-heating
do not exist, and mine monitoring records are confidential. We therefore built a **physics-informed
simulator** (`src/ews/simulator/`). The honest consequence: *all reported numbers are results on
simulated mines*; they show that the method works **if** real gas behaviour looks like our
assumptions. They are not evidence of performance in a real mine.

## 2. What the simulator models
1. **Hidden coal temperature** from a Semenov-type heat balance (heat generation grows
   exponentially with temperature, heat loss is linear). When the heat-generation potential passes a critical
   value the pocket runs away (saddle-node bifurcation) after a long slow "bottleneck", which is
   our incubation period.
2. **Stage labels come from the hidden temperature** (Normal <40 °C, Incubation 40-80 °C,
   Active heating 80-200 °C, Fire >=200 °C), **not** from gas thresholds. This avoids the circular
   argument "the ML model learned the same threshold we used to label".
3. **Gas make** from temperature: O2 deficiency, Graham's ratio and Young's ratio curves with
   typical literature magnitudes, diluted by return airflow, with transport delay and mixing lag.
4. **Confounders**: shot-firing CO spikes (the main cause of false alarms for fixed thresholds),
   diesel CO, ventilation changes, barometric pumping, background CH4 / CO2.
5. **Sensor effects**: noise, drift, 1-ppm resolution, spikes, telemetry dropouts, frozen sensors.

## 3. Things the group MUST verify with real sources before claiming more
| Item | Where in code | Action |
|---|---|---|
| Alert thresholds (CO 15/50/400 ppm, Graham 0.5/1/3) | `constants.py` | Replace with the values from the DGMS circular / mine TARP you cite in the report |
| Stage temperature edges (40/80/200 °C) | `constants.py` | Compare with your textbook / lecture notes |
| Graham / Young curves vs temperature | `simulator/episode.py` (`_GR_*`, `_Y_*`) | Compare with published laboratory data (programmed-temperature oxidation tests) |
| Sampling interval (15 min) | `constants.py` | Real tube-bundle systems may be slower (10-30 min) |

## 4. Known weaknesses
* **Incubation (40-80 °C) is hard to detect early**: at these temperatures CO from the coal is of the same size as
  CO from diesel and blasting. Detection delay at Level 1 is days, not hours.
* The fixed-threshold baseline uses *handbook* levels. A smarter, data-tuned single threshold is included
  as a second baseline; the ML advantage over it is smaller than over the handbook rule.
* Test set = 150 simulated panels; event counts per level are 40-75, so differences of a few hours are within noise.
* No real mine data, no multi-point (spatial) information, no H2 / ethylene / ethane channels.
