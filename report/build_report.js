// Builds the final project report (.docx) from reports/results.json and reports/figures/*.png
// Usage: node report/build_report.js   (run from the repository root)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel, AlignmentType,
  BorderStyle, WidthType, ShadingType, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const R = JSON.parse(fs.readFileSync("reports/results.json", "utf8"));
const FIG = "reports/figures/";
const FONT = "Calibri";
const W = 9026; // A4 content width (DXA) with 1-inch margins

// ---------- helpers ----------
const f1 = (x, d = 1) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "–" : (100 * x).toFixed(0) + "%");
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const P = (t, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o,
  children: Array.isArray(t) ? t : [run(t)] });
const B = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(t) ? t : [run(t)] });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: t, font: FONT, bold: true, size: 30, color: "1F3864" })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 90 }, children: [new TextRun({ text: t, font: FONT, bold: true, size: 25, color: "2E5597" })] });
const CAP = (t) => new Paragraph({ spacing: { before: 60, after: 160 }, alignment: AlignmentType.CENTER, children: [run(t, { italics: true, size: 19 })] });
const mono = (t) => new Paragraph({ spacing: { after: 40 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2" }, children: [new TextRun({ text: t, font: "Consolas", size: 18 })] });
const bold = (t) => run(t, { bold: true });

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function IMG(name, maxW = 600, maxH = 640) {
  const { w, h } = pngSize(FIG + name);
  let s = Math.min(maxW / w, maxH / h);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + name), transformation: { width: Math.round(w * s), height: Math.round(h * s) },
      altText: { title: name, description: name, name } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function TBL(headers, rows, widths, opts = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, hdr, i) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: hdr ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : (opts.hi && opts.hi(i) ? { type: ShadingType.CLEAR, fill: "FFF2CC" } : undefined),
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: hdr || t === rows.__first ? AlignmentType.LEFT : AlignmentType.LEFT,
      children: [new TextRun({ text: String(t), font: FONT, size: 18, bold: hdr })] })],
  });
  return new Table({ width: { size: tot, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, j) => cell(h, widths[j], true)) }),
      ...rows.map((r, i) => new TableRow({ cantSplit: true, children: r.map((c, j) => cell(c, widths[j], false, i)) }))] });
}

const LAB = { logreg: "Logistic regression", rf: "Random forest", xgb: "XGBoost", cnn: "1D-CNN", lstm: "LSTM",
  "Tuned threshold rule": "Tuned threshold rule (data-fitted)", "Manual 8-hourly sampling": "Handbook: manual 8-hourly sampling",
  "Continuous fixed threshold": "Handbook: continuous fixed threshold" };
const lab = (n) => LAB[n] || n;

// ---------- content ----------
const ds = R.dataset, st = R.stage, ev = R.events, pl = R.paired_lead.xgb;
const nTest = ds.episodes_by_split.test, nTrain = ds.episodes_by_split.train, nVal = ds.episodes_by_split.val;
const x2 = ev.xgb["2"], x1 = ev.xgb["1"], r2 = ev["Tuned threshold rule"]["2"], r1 = ev["Tuned threshold rule"]["1"];
const m2 = ev["Manual 8-hourly sampling"]["2"], c2 = ev["Continuous fixed threshold"]["2"];
const m1 = ev["Manual 8-hourly sampling"]["1"], c1 = ev["Continuous fixed threshold"]["1"];
const oc = R.operating_curves;

const kids = [];

// Title block (follows the proposal sheet)
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 80 }, children: [run("NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL", { bold: true, size: 24 })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [run("Department of Mining Engineering", { size: 22 })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run("FINAL GROUP PROJECT REPORT", { bold: true, size: 26, color: "1F3864" })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [run("Spontaneous Combustion Early-Warning System: A Gas-Ratio-Index-Based Machine Learning Approach for Early Detection of Self-Heating in Underground Coal Mines", { bold: true, size: 32, color: "1F3864" })] }));
kids.push(TBL(["Item", "Details"], [
  ["Course name", "Mine Hazards, Rescue and Recovery"], ["Course code", "MI 302"], ["Semester / Year", "Fifth Semester / 3rd Year"],
  ["Academic year", "2026-27"],
  ["Group members", "1. Macklin Chriss Miranda (241MN027)\n2. Ganesh M. (241MN013)\n3. Mounesh Gowda (241MN031)"],
  ["Code repository", "coalfire-ews (GitHub link to be added by the group)"],
], [2400, 6626]));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// Abstract
kids.push(H1("Abstract"));
kids.push(P(`Spontaneous combustion of coal is a leading cause of underground coal-mine fires. It develops over days to weeks, yet current practice relies on periodic manual gas sampling and fixed trigger levels (CO, Graham's ratio). We built and evaluated a machine-learning early-warning system that classifies the stage of self-heating (Normal, Incubation, Active Heating, Fire) from multi-gas telemetry (CO, CO₂, O₂, CH₄), air temperature, airflow and barometric pressure. Because no public real-mine dataset exists, we developed a physics-informed simulator in which a hidden coal temperature (Semenov-type heat balance) drives gas make, which is then diluted by ventilation, disturbed by shot-firing, diesel, barometric pumping and sensor faults. We generated ${ds.episodes} simulated panels (${(ds.rows / 1e6).toFixed(2)} million readings), split by panel into ${nTrain}/${nVal}/${nTest} train/validation/test panels, engineered ${ds.n_features} causal rolling-window features, and compared logistic regression, random forest, XGBoost, a 1D-CNN and an LSTM against two handbook fixed-threshold baselines and a data-tuned threshold rule.`));
kids.push(P(`The central metric is detection lead time at a matched false-alarm budget. On unseen test panels the stage classifiers reach a macro-F1 of ${f1(st.xgb.macro_f1, 2)} (XGBoost) versus ${f1(st["Continuous fixed threshold"].macro_f1, 2)} for fixed thresholds. For the Level-2 (active-heating) alert, XGBoost detected ${pct(x2.detection_rate)} of events with a median delay of ${f1(x2.median_delay_h)} h relative to the true onset (negative = before), at ${f1(x2.fa_per_100d)} false alarms per 100 panel-days, whereas handbook thresholds produced ${f1(m2.fa_per_100d)} (manual sampling) and ${f1(c2.fa_per_100d)} (continuous) false alarms per 100 panel-days and alerted a median of ${f1(Math.min(pl["Manual 8-hourly sampling"]["2"].median_lead_h, pl["Continuous fixed threshold"]["2"].median_lead_h))}–${f1(Math.max(pl["Manual 8-hourly sampling"]["2"].median_lead_h, pl["Continuous fixed threshold"]["2"].median_lead_h))} h later. At Level 1 (incubation), fixed thresholds react quickly but with prohibitively many false alarms, and ML detects incubation only after about ${f1(x1.median_delay_h / 24)} days. TreeSHAP shows that the model relies on spike-robust CO-floor and oxygen-deficiency trend features, consistent with mining practice. All results are on simulated data; we describe the assumptions and the validation needed before real use.`));

// 1 Introduction
kids.push(H1("1. Introduction"));
kids.push(P("Coal is an oxidising material. In a goaf or a crushed-coal pocket where airflow is enough to supply oxygen but not enough to remove the heat produced, the coal temperature rises, oxidation accelerates, and the process can run away to open fire. The sequence is slow at first (incubation, days to weeks) and fast at the end, which makes early detection the key safety lever: the earlier the heating is found, the more options exist (sealing, inertisation, water, withdrawal)."));
kids.push(P("In practice, gas from return airways or tube-bundle points is sampled periodically and interpreted with indices such as Graham's ratio, Young's ratio, the CO/CO₂ ratio and CO make rate against fixed trigger levels. Three weaknesses motivate this project: (i) between sampling rounds early heating can go unnoticed; (ii) CO from shot-firing and diesel equipment mimics heating and causes false alarms, which in turn cause alarm fatigue; (iii) a threshold sees the current value and not the trend."));
kids.push(P("Objectives. (1) Build a realistic, reproducible synthetic data source for self-heating gas trajectories; (2) engineer features from classical gas indices and their trends; (3) compare classical ML and deep sequence models against fixed thresholds using a domain metric (lead time at a matched false-alarm rate); (4) explain alerts with SHAP; (5) deliver a working API and live dashboard; (6) test edge cases and state limitations honestly."));

// 2 Proposal review
kids.push(H1("2. Review of the original proposal and changes made"));
kids.push(P("We re-examined the proposal critically before building anything. The table lists what we changed and why. The most important change is that stage labels are no longer defined from gas thresholds: if they were, the ML model would simply re-learn the thresholds and any 'lead over thresholds' would be circular."));
kids.push(TBL(["Original idea", "Problem", "What we did"], [
  ["Generate synthetic series 'using gas-ratio indices and DGMS thresholds'", "Labels defined by the same thresholds that form the baseline → circular evaluation", "Simulate a hidden coal temperature; stage = temperature band (40/80/200 °C). Gases are consequences, thresholds are only the baseline"],
  ["Baseline = fixed thresholds only", "A weak straw man; ignores that a threshold can also be tuned", "Three baselines: manual 8-hourly sampling, continuous fixed threshold, and a data-tuned single-feature rule given the same false-alarm budget"],
  ["Metric = lead time over thresholds", "Lead time alone rewards noisy detectors", "Lead time reported together with false alarms per 100 panel-days; all methods calibrated to the same budget; operating curves"],
  ["Sensor streams: CO, CO₂, O₂, CH₄, temperature", "Real data also contain blasting fumes, ventilation changes, pressure effects, dropouts", "Added shot-firing, diesel, airflow changes, barometric pumping, drift, spikes, dropouts, frozen sensors; added airflow and pressure channels"],
  ["Random train/test split implied", "Rows of one panel are correlated → leakage", "Split by panel (episode), stratified by episode type"],
  ["LSTM / 1D-CNN assumed to be better", "Not tested on a fair footing", "Same split, same alarm logic, same budget; ablations show where signal comes from"],
  ["'Validate on edge cases'", "Not specified", "Seven named scenarios plus six simulator stress tests (noise ×2/×3, weaker CO, dropouts)"],
  ["Dashboard / API", "Scope creep risk", "Kept minimal: /predict, /stream and one HTML dashboard with explained alerts"],
], [2500, 3000, 3526]));

// 3 Background
kids.push(H1("3. Background: gas indices used"));
kids.push(P("Let CO, CO₂, O₂ and CH₄ be measured concentrations (CO in ppm, the others in %) and N₂ = 100 − O₂ − CO₂ − CH₄ − CO."));
kids.push(B([bold("Oxygen deficiency (%) "), run("= 0.265·N₂ − O₂. Using nitrogen by difference makes it insensitive to dilution by methane.")]));
kids.push(B([bold("Graham's ratio (%) "), run("= 100 × CO(%) / O₂ deficiency(%). Rises with coal temperature; relatively insensitive to dilution by airflow.")]));
kids.push(B([bold("Young's ratio "), run("= excess CO₂ / O₂ deficiency.")]));
kids.push(B([bold("CO/CO₂ (carbon-oxides) ratio "), run("= CO(%)/CO₂(%).")]));
kids.push(B([bold("CO make (L/min) "), run("= CO(ppm) × 10⁻⁶ × airflow(m³/s) × 1000 × 60. Corrects the concentration for ventilation.")]));
kids.push(P("Practical note. At low temperatures the oxygen deficiency is a few hundredths of a percent, comparable to the accuracy of a paramagnetic O₂ analyser, so Graham's ratio is unreliable below roughly 0.2 % deficiency. We therefore evaluate it only above that level in the baselines and use smoothed values in the ML features."));

// 4 Data
kids.push(H1("4. Data"));
kids.push(H2("4.1 Real or synthetic?"));
kids.push(P("We searched for a public dataset of continuous multi-gas telemetry from a coal panel progressing through self-heating and did not find one; mine monitoring data are confidential and heating incidents are rare. Laboratory adiabatic and programmed-temperature oxidation tests exist in the literature, but they provide gas-versus-temperature curves, not noisy underground time series. We therefore created a physics-informed synthetic data set and made the limitation explicit. Section 9 describes how the study can be tied to real data."));
kids.push(H2("4.2 Simulator design"));
kids.push(P("Each simulated episode is one return-airway monitoring record of 20 days at 15-minute sampling (1,920 readings)."));
kids.push(B([bold("Hidden heating dynamics. "), run("The excess temperature x of a coal pocket follows dx/dτ = G(r·exp(b·x)) − x, a Semenov-type balance between exponentially temperature-dependent heat generation and linear heat loss. The heat-generation potential r(t) ramps up slowly. Below the critical value r_c = 1/(b·e) the pocket settles at a warm equilibrium; above it, the equilibrium disappears and the pocket escapes through a long bottleneck (incubation) into runaway. Sealing is modelled as a sudden increase in heat loss.")]));
kids.push(B([bold("Stage labels. "), run("From the hidden temperature: Normal < 40 °C, Incubation 40–80 °C, Active Heating 80–200 °C, Fire ≥ 200 °C.")]));
kids.push(B([bold("Gas make. "), run("Oxygen deficiency grows exponentially with temperature (saturating at fire); Graham's and Young's ratios follow temperature curves of typical published magnitude; CO and CO₂ follow from them. Make is divided by relative airflow, delayed by 15–45 min and low-pass filtered (transport and mixing).")]));
kids.push(B([bold("Confounders. "), run("Shot-firing CO/CO₂ spikes (peaks 15–120 ppm decaying over 1–3 h, up to three times per day), diesel CO bumps, airflow changes (±30 % for hours to days), barometric pumping, strata CH₄ and CO₂.")]));
kids.push(B([bold("Sensor model. "), run("Noise, slow drift, 1-ppm resolution, electrical spikes, telemetry dropouts (30 % of panels), frozen readings.")]));
kids.push(P(`Seven episode types were generated: normal, benign warming, stable warm pocket, slow runaway, runaway, fast runaway and sealed (quenched) heating. Figure 1 shows a runaway episode; note how little the gases move until the late stage, and Figure 2 shows a panel with no heating at all where blasting spikes cross the fixed CO trigger repeatedly.`));
kids.push(IMG("fig01_example_episode.png", 520, 560)); kids.push(CAP("Figure 1. Simulated runaway episode. The stage shading comes from the hidden coal temperature (top panel)."));
kids.push(IMG("fig02_confounders.png", 560, 340)); kids.push(CAP("Figure 2. A panel without heating: shot-firing spikes repeatedly exceed a 15 ppm fixed trigger."));
kids.push(H2("4.3 Data set summary"));
const typeRows = Object.entries(ds.episodes_by_type).map(([k, v]) => [k, v]);
kids.push(TBL(["Quantity", "Value"], [
  ["Episodes (panels)", `${ds.episodes} (train ${nTrain} / validation ${nVal} / test ${nTest}), stratified by type`],
  ["Rows (readings)", ds.rows.toLocaleString("en-US")],
  ["Episodes per type", typeRows.map(([k, v]) => `${k} ${v}`).join(", ")],
  ["Row share per stage (after 48 h warm-up)", Object.entries(ds.stage_share_rows).map(([k, v]) => `${k} ${(100 * v).toFixed(1)}%`).join(", ")],
  ["Missing gas readings", `about ${(100 * ds.missing_share.co_ppm).toFixed(1)} % (telemetry dropouts)`],
  ["Engineered features", ds.n_features],
], [3000, 6026]));
kids.push(P(""));

// 5 Methods
kids.push(H1("5. Methodology"));
kids.push(H2("5.1 Pre-processing and features"));
kids.push(P(`Gas gaps up to 4 h are forward-filled; longer gaps stay missing and are flagged by a missing-share feature (XGBoost and random forest handle missing values natively). The first 48 h of each episode are used only as history. We engineered ${ds.n_features} causal features (each uses only past data): current values; oxygen deficiency; CO make rate; Graham, Young and CO/CO₂ ratios computed on smoothed values; means over 1, 4, 12 and 24 h; spike-robust medians and minima ("floor") of CO and CO make; CO above its 24-h floor; short-term and 12-h slopes and exponential growth rate; a blast-signature feature (1-h mean minus 4-h minimum); airflow, pressure and methane context; time of day; share of missing data. Medians and minima ignore 1–3 h blasting spikes but still rise when the coal is genuinely heating. Unit tests verify that the same function gives identical features offline and on a 48-h streaming buffer.`));
kids.push(H2("5.2 Models"));
kids.push(B("Logistic regression (standardised features, class-balanced) – simple reference."));
kids.push(B("Random forest (150 trees, depth 16) and XGBoost (multi-class softmax, depth 6, learning rate 0.08, early stopping on validation log-loss) on the engineered features, with square-root class weights so rarer, more dangerous stages weigh more."));
kids.push(B("1D-CNN and LSTM (PyTorch) on raw 48-hour windows of 12 transformed channels (the LSTM sees hourly-averaged data). Same class weights, AdamW, 6 epochs, best validation macro-F1 kept."));
kids.push(H2("5.3 Alert logic and false-alarm control"));
kids.push(P("For alert level k ∈ {1, 2, 3} (stage ≥ Incubation, Active Heating, Fire) the model score is P(stage ≥ k). An alarm is active when the score exceeds a threshold for m consecutive samples (m ∈ {2, 4, 8, 16}). For every detector, threshold and m are selected on the validation panels to minimise mean detection delay (misses count as 7 days) subject to a false-alarm budget of 1 per 100 panel-days; they are then frozen and evaluated on the test panels. The tuned threshold rule receives exactly the same treatment, choosing the best single feature per level from six candidates. The handbook baselines use fixed levels (CO ≥ 15/50/400 ppm or Graham's ratio ≥ 0.5/1/3), with two practices: manual sampling once per 8-hour shift, and continuous telemetry requiring two consecutive exceedances."));
kids.push(H2("5.4 Evaluation metrics"));
kids.push(B([bold("Stage classification: "), run("macro-F1, quadratic-weighted κ, mean absolute stage error and per-stage recall (after the 48-h warm-up).")]));
kids.push(B([bold("Event level (per alert level k): "), run("the event is the first time the true stage reaches k. Detection delay = alarm time − event time (negative: warned in advance, allowed up to 6 h before). A false alarm is a new alarm (merged within 12 h) after which the true stage stays below k for 6 h. Rates are per 100 panel-days; the test set contains 2,700 panel-days.")]));
kids.push(B([bold("Lead time "), run("of a model over a baseline = baseline delay − model delay on events both detect.")]));

// 6 Results
kids.push(H1("6. Results"));
kids.push(H2("6.1 Stage classification"));
const stRows = ["logreg", "rf", "xgb", "cnn", "lstm", "Manual 8-hourly sampling", "Continuous fixed threshold"].map((n) => {
  const s = st[n]; return [lab(n), f1(s.macro_f1, 3), f1(s.qw_kappa, 3), f1(s.mean_abs_stage_error, 3),
    pct(s.per_class.Incubation.recall), pct(s.per_class["Active Heating"].recall), pct(s.per_class.Fire.recall)];
});
kids.push(TBL(["Method", "Macro-F1", "Quad. κ", "Mean |stage error|", "Recall Incub.", "Recall Active", "Recall Fire"], stRows, [2626, 900, 900, 1200, 1100, 1150, 1150], { hi: (i) => i === 2 }));
kids.push(CAP("Table 1. Stage classification on the unseen test panels (XGBoost highlighted)."));
kids.push(P(`All learned models clearly outperform fixed thresholds (macro-F1 ${f1(st.logreg.macro_f1, 2)}–${f1(st.xgb.macro_f1, 2)} vs ${f1(st["Continuous fixed threshold"].macro_f1, 2)}). XGBoost is best, but the random forest, CNN and LSTM are within about 0.02 macro-F1, so we cannot claim a meaningful difference among them with ${nTest} test panels. The difficult stage is Incubation (recall ${pct(st.xgb.per_class.Incubation.recall)}): between 40 and 60 °C the extra gas is smaller than the background and noise, so the model confuses it with Normal (Figure 4). Active-heating and Fire recall are ${pct(st.xgb.per_class["Active Heating"].recall)} and ${pct(st.xgb.per_class.Fire.recall)}.`));
kids.push(IMG("fig03_stage_models.png", 520, 260)); kids.push(CAP("Figure 3. Stage-classification quality of all methods."));
kids.push(IMG("fig04_confusion_xgb.png", 330, 300)); kids.push(CAP("Figure 4. XGBoost confusion matrix (row-normalised)."));

kids.push(H2("6.2 Early-warning performance (lead time at a matched false-alarm budget)"));
function evTable(k, title) {
  const names = ["Manual 8-hourly sampling", "Continuous fixed threshold", "Tuned threshold rule", "logreg", "rf", "xgb", "cnn", "lstm"];
  const rows = names.map((n) => { const r = ev[n][k]; return [lab(n), pct(r.detection_rate), f1(r.median_delay_h), f1(r.fa_per_100d, 2)]; });
  return [TBL(["Method", "Events detected", "Median delay (h)", "False alarms / 100 panel-days"], rows, [3426, 1600, 1700, 2300], { hi: (i) => i === 5 }), CAP(title)];
}
kids.push(...evTable("2", `Table 2. Level-2 (active-heating) alert on test panels; ${x2.n_events} events. Delay is relative to the true onset; negative = early. ML and tuned rule calibrated to ≤1 false alarm per 100 panel-days on validation panels.`));
kids.push(P(`Level 2 is where ML helps most. XGBoost detects ${pct(x2.detection_rate)} of the ${x2.n_events} events, typically ${f1(Math.abs(x2.median_delay_h))} h before the coal reaches 80 °C, with ${f1(x2.fa_per_100d)} false alarms per 100 panel-days. Handbook thresholds raise ${f1(m2.fa_per_100d)} (manual) and ${f1(c2.fa_per_100d)} (continuous) false alarms per 100 panel-days, almost all caused by shot-firing CO, and detect only after the onset (median ${f1(m2.median_delay_h)} h and ${f1(c2.median_delay_h)} h). On paired events the XGBoost alarm precedes the manual-sampling alarm by a median of ${f1(pl["Manual 8-hourly sampling"]["2"].median_lead_h)} h and the continuous-threshold alarm by ${f1(pl["Continuous fixed threshold"]["2"].median_lead_h)} h, while producing roughly ${f1(m2.fa_per_100d / x2.fa_per_100d, 0)}–${f1(c2.fa_per_100d / x2.fa_per_100d, 0)} times fewer false alarms. Compared with the stronger data-tuned rule, which keeps a low false-alarm rate (${f1(r2.fa_per_100d, 2)}) but detects only ${pct(r2.detection_rate)} of events at a median delay of ${f1(r2.median_delay_h)} h, the median lead is ${f1(pl["Tuned threshold rule"]["2"].median_lead_h)} h. Because the tuned rule operated at a lower false-alarm rate than XGBoost on the test set, we also compare them at similar rates: with a budget of 2 per 100 panel-days the rule reaches ${f1(oc["Tuned threshold rule"]["2.0"]["2"].fa_per_100d, 2)} false alarms with a mean delay (misses counted as 7 days) of ${f1(oc["Tuned threshold rule"]["2.0"]["2"].expected_delay_h)} h, whereas XGBoost at ${f1(oc.xgb["1.0"]["2"].fa_per_100d, 2)} false alarms has ${f1(oc.xgb["1.0"]["2"].expected_delay_h)} h (Figure 6).`));
kids.push(...evTable("1", `Table 3. Level-1 (incubation) alert on test panels; ${x1.n_events} events.`));
kids.push(P(`Level 1 is a different story and we report it as a mixed result. Handbook thresholds fire within hours of the onset (median ${f1(m1.median_delay_h)} h manual, ${f1(c1.median_delay_h)} h continuous) but at ${f1(m1.fa_per_100d, 0)}–${f1(c1.fa_per_100d, 0)} false alarms per 100 panel-days, i.e. roughly one false alarm every 1.5–3.5 days per panel, which would be ignored in practice. At about 2 false alarms per 100 panel-days, XGBoost detects ${pct(x1.detection_rate)} of incubation events but only after a median of ${f1(x1.median_delay_h / 24, 1)} days; the data-tuned rule detects ${pct(r1.detection_rate)} after ${f1(r1.median_delay_h / 24, 1)} days. The physical reason is that the gas signal from coal below 60 °C is of the order of the background CO and the sensor noise. Honest conclusion: gas telemetry alone does not allow a low-false-alarm Level-1 alert in our simulation within hours; ML shifts the usable detection of incubation from roughly ${f1(r1.median_delay_h / 24, 1)} days (tuned rule) to ${f1(x1.median_delay_h / 24, 1)} days and with better coverage.`));
kids.push(...evTable("3", `Table 4. Level-3 (fire) alert; ${ev.xgb["3"].n_events} events. Fire is obvious in the gases, so there is little to gain.`));
kids.push(P(""));
kids.push(TBL(["XGBoost lead over …", "Level 1 (h)", "Level 2 (h)", "Level 3 (h)"], ["Manual 8-hourly sampling", "Continuous fixed threshold", "Tuned threshold rule"].map((b) =>
  [lab(b), ...["1", "2", "3"].map((k) => `${f1(pl[b][k].median_lead_h)}  (n=${pl[b][k].n_paired})`)]), [3626, 1800, 1800, 1800]));
kids.push(CAP("Table 5. Median paired lead time of XGBoost (positive = XGBoost earlier) on events detected by both; n = number of paired events. Negative values at Level 1 versus handbook thresholds reflect their (false-alarm-prone) quick triggering."));
kids.push(IMG("fig05_detection_delay.png", 600, 270)); kids.push(CAP("Figure 5. Detection delay distributions for detected events."));
kids.push(IMG("fig06_operating_curve.png", 540, 230)); kids.push(CAP("Figure 6. Mean detection delay (misses counted as 7 days) against the false-alarm budget used for calibration; lower is better."));
kids.push(P(`Deep sequence models versus engineered features. The CNN and LSTM reach a similar Level-2 performance (delay ${f1(ev.cnn["2"].median_delay_h)} h and ${f1(ev.lstm["2"].median_delay_h)} h) and slightly worse stage macro-F1 (${f1(st.cnn.macro_f1, 3)} and ${f1(st.lstm.macro_f1, 3)}). With ${nTrain} training panels and strong domain features, deep learning gives no advantage in our experiments, but it also does not need hand-made features, which is useful as a cross-check.`));

kids.push(H2("6.3 Explainability"));
const shapA = Object.keys(R.shap_top_features["Active Heating"]).slice(0, 5).join(", ");
kids.push(P(`TreeSHAP values were computed exactly from the XGBoost trees on a class-balanced sample of 4,800 test rows. For the Active-Heating class the five most influential features are ${shapA}: the model looks at the floor of CO (which blasting spikes cannot lift), the Graham's ratio and the trend in oxygen deficiency — the same quantities a mining engineer would examine. In the live dashboard every alert shows its top five contributing features in words. A caution: some features rank high for Fire (air temperature, CO₂) because air temperature rises in our simulator when fire is present; they are legitimate in the simulator but should be re-checked on real data.`));
kids.push(IMG("fig07_shap_importance.png", 400, 280)); kids.push(CAP("Figure 7. Mean |SHAP| of the top features for the Active-Heating class."));
kids.push(IMG("fig07b_shap_beeswarm.png", 440, 330)); kids.push(CAP("Figure 8. SHAP summary (beeswarm) for the Active-Heating class."));

kids.push(H2("6.4 Ablation study"));
const ab = R.ablation;
kids.push(TBL(["Feature set (XGBoost)", "Features", "Macro-F1", "L1 mean delay (h)", "L2 mean delay (h)"], Object.entries(ab).map(([k, v]) => [k, v.n_features, f1(v.macro_f1, 3), f1(v.L1.expected_delay_h), f1(v.L2.expected_delay_h)]), [3926, 900, 1000, 1600, 1600]));
kids.push(CAP("Table 6. Ablations. Mean delay counts missed events as 7 days; lower is better. Differences below about 3 h are within noise."));
kids.push(P("Findings. Rolling-window features are essential: using only instantaneous values drops macro-F1 from 0.835 to 0.785 and makes Level-1 detection about 21 h slower. Removing the spike-robust min/median features hurts Level 1 most (+5.7 h), supporting the design intent. Removing airflow, pressure, methane and time-of-day context or the classical index features did not change results in our simulator — so we cannot claim these help; in the simulator, CO make already normalises for airflow. In real mines the context features may matter more."));
kids.push(IMG("fig08_ablation.png", 520, 240)); kids.push(CAP("Figure 9. Ablation of feature groups."));

kids.push(H2("6.5 Robustness to simulator assumptions (no retraining)"));
const se = R.sensitivity;
kids.push(TBL(["Condition (150 new panels)", "XGBoost: det. / delay h / FA", "Tuned rule: det. / delay h / FA", "Fixed threshold: det. / delay h / FA"],
  Object.entries(se).map(([k, v]) => [k, ...["xgb", "rule", "continuous"].map((m) => `${pct(v.L2[m].detection_rate)} / ${f1(v.L2[m].median_delay_h)} / ${f1(v.L2[m].fa_per_100d)}`)]), [3026, 2000, 2000, 2000]));
kids.push(CAP("Table 7. Level-2 alert under changed simulator settings. Thresholds are those calibrated on nominal validation panels. FA = false alarms per 100 panel-days."));
kids.push(P("Doubling or tripling sensor noise, halving the heating CO make, or introducing telemetry dropouts in 80 % of panels did not reduce Level-2 detection of XGBoost below 97 %. False alarms rise with noise (to about 3.3 per 100 panel-days at ×3 noise) because the thresholds were not re-tuned. The tuned rule detects only 86–90 % of events in five of the six settings. Fixed thresholds remain at about 56–61 false alarms per 100 panel-days in every setting. These tests probe sensitivity to our assumptions but cannot replace real-data validation."));
kids.push(IMG("fig09_sensitivity.png", 600, 190)); kids.push(CAP("Figure 10. Level-2 detection rate, median delay and false alarms under stress conditions."));

kids.push(H2("6.6 Edge-case scenarios"));
const sc = R.scenarios;
kids.push(TBL(["Scenario", "True L2 onset (day)", "ML L2 alarm (day)", "Fixed L2 alarm (day)", "ML false alarms (L1–L3)", "Fixed false alarms (L1–L3)"],
  Object.entries(sc).map(([k, v]) => [k, f1(v.L2.stage_onset_day, 1), f1(v.L2.model_first_alarm_day, 1), f1(v.L2.baseline_first_alarm_day, 1),
    v.L1.model_false_alarms + v.L2.model_false_alarms + v.L3.model_false_alarms, v.L1.baseline_false_alarms + v.L2.baseline_false_alarms + v.L3.baseline_false_alarms]),
  [2200, 1200, 1250, 1350, 1500, 1526]));
kids.push(CAP("Table 8. Named scenarios. '–' = no onset or no alarm. The fixed-threshold alarm day is often the first false alarm caused by blasting (day ≈ 2)."));
kids.push(P("Observations. (i) Fast runaway: ML alarms at the moment of transition (≈ 1 h before true onset), after a short incubation. (ii) Runaway with a 2-day telemetry outage straddling the transition: the Level-2 alert was raised at the very start of the outage (about 1 day before the true onset) and held through it by the fail-safe latch. This scenario therefore does not show what would happen if the outage had begun earlier; that case is untested. (iii) Ventilation shock (airflow halved, no heating) and blasting storm (six blasts a day): no Level-2 false alarm, but a Level-1 false alarm appeared in each (day 10.5 and day 4.0) – a real weakness at Level 1 in extreme confounding. (iv) Slow incubation: the pocket reaches 40 °C on day 10.3 and never leaves the 40–80 °C band; the ML system raised no Level-1 alert in the remaining 10 days – a miss. (v) Self-quenching: alarm raised near the peak and cleared after sealing. Fixed thresholds raised false alarms in every scenario (Table 8)."));
kids.push(IMG("fig10_scenarios.png", 520, 620)); kids.push(CAP("Figure 11. Edge-case scenarios. Blue: CO; orange: model probability of at least active heating; dashed: ML alert; dotted: fixed-threshold alert; black: true onset."));

// 7 Deployment
kids.push(H1("7. Prototype system"));
kids.push(P("The XGBoost model and alert policy are wrapped in a FastAPI service. POST /predict accepts the last 48 h of readings and returns the stage, class probabilities, risk score (0–100), alert level, key indices and a ranked list of explanatory features. GET /stream/{scenario} streams a simulated panel in real time as server-sent events to a browser dashboard that shows a risk gauge, stage, alert level, CO, oxygen deficiency, model probabilities, the hidden coal temperature for judging purposes, a comparison of ML versus fixed-threshold alert levels with live lead-time, and a log of explained alerts. If gas telemetry is missing for 2 h or more, the system reports a telemetry fault and holds the current alert level instead of clearing it (fail-safe)."));
kids.push(mono("uvicorn ews.serving.app:app --port 8000    # then open http://localhost:8000"));
kids.push(P(""));

// 8 Discussion
kids.push(H1("8. Discussion"));
kids.push(B([bold("What worked. "), run(`Trend features with spike-robust statistics let the model separate blasting from heating; at Level 2 this gives earlier warnings than handbook thresholds with 5–35 times fewer false alarms, and better coverage than a tuned single-feature rule.`)]));
kids.push(B([bold("What did not. "), run("Early incubation is not reliably detectable from gases alone; extreme ventilation or blasting patterns still cause Level-1 false alarms; context features did not add value in our simulator; the false-alarm rate on the test panels (1.7 per 100 panel-days at Level 2) was above the 1.0 calibrated on validation, so a safety margin should be applied when setting thresholds.")]));
kids.push(B([bold("Interpretation of 'lead time'. "), run("Lead time is only meaningful together with the false-alarm rate; a detector that alarms constantly is 'early' but useless. All comparisons in this report are therefore tied to the same budget.")]));
kids.push(B([bold("Deep learning versus engineered features. "), run("Equal in performance here; the simpler XGBoost model is cheaper, explainable and easier to audit.")]));

// 9 Limitations
kids.push(H1("9. Limitations and future work"));
kids.push(B("All data are simulated. Gas–temperature curves are of typical published magnitude, not fitted to a mine; absolute numbers (hours of lead) are properties of the simulator and must not be quoted as real-world performance."));
kids.push(B("Handbook trigger levels (CO 15/50/400 ppm; Graham 0.5/1/3) are indicative; they should be replaced by the DGMS / mine TARP values. The group must verify and cite them."));
kids.push(B("The test set has 150 panels; event counts are 28–75 per level, so small differences are not significant. Confidence intervals by bootstrap over panels are a natural extension."));
kids.push(B("Single sampling point; no spatial information, no H₂, C₂H₄ or C₂H₆ channels (ethylene appears at higher temperatures and is a valuable indicator)."));
kids.push(B("Future work: (1) calibrate the simulator curves to laboratory oxidation data and tune the model on any available mine gas logs (e.g. from a mine partner or department laboratory); (2) add hydrocarbon indicators; (3) multi-point spatial modelling; (4) prognosis: predict time-to-active-heating; (5) human-in-the-loop trial with a TARP-based response plan."));

// 10 Conclusion
kids.push(H1("10. Conclusion"));
kids.push(P(`We built an end-to-end early-warning pipeline for coal self-heating: a physics-informed simulator, causal gas-index features, five classifiers, a lead-time-at-matched-false-alarm evaluation, SHAP explanations, an API and a live dashboard. In simulated panels, ML raised active-heating alerts before the coal reached 80 °C in ${pct(x2.detection_rate)} of events at ${f1(x2.fa_per_100d)} false alarms per 100 panel-days, compared with ${f1(m2.fa_per_100d)}–${f1(c2.fa_per_100d)} for fixed thresholds that alerted later. Early incubation remains hard to detect, and all conclusions depend on the simulator until validated on real mine data. The project demonstrates a rigorous way to compare ML with threshold-based gas monitoring and provides a base for real-data validation.`));

// References
kids.push(H1("References"));
["Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
 "Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30.",
 "Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
 "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
 "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
 "Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32.",
 "Frank-Kamenetskii, D. A. (1969). Diffusion and Heat Transfer in Chemical Kinetics. Plenum Press, New York."].forEach((r, i) => kids.push(new Paragraph({ spacing: { after: 70 }, indent: { left: 360, hanging: 360 }, children: [run(`[${i + 1}] ${r}`, { size: 20 })] })));
kids.push(P([run("Domain references to be added by the group: ", { bold: true, size: 20 }), run("DGMS circulars / TARP provisions on spontaneous heating and gas monitoring; the source of Graham's, Young's and CO/CO₂ ratio interpretation used in the MI 302 course notes; adiabatic / programmed-temperature oxidation studies of Indian coals. These were not inserted here because they must be verified against the original documents.", { size: 20 })]));

// Appendix
kids.push(H1("Appendix A. Reproducing the results"));
["pip install -e \".[dl,dev]\"", "python scripts/01_generate_data.py", "python scripts/02_train_tabular.py", "python scripts/03_train_sequence.py", "python scripts/04_evaluate.py", "python scripts/05_make_figures.py", "python scripts/build_notebook.py", "node report/build_report.js", "pytest -q", "uvicorn ews.serving.app:app --port 8000"].forEach((c) => kids.push(mono(c)));
kids.push(P(""));
kids.push(P(`Everything is seeded (simulation seed 2026); the configuration is in configs/default.yaml. Training on one CPU core took about 20 s (XGBoost), 60 s (random forest) and about 3–4 min each for the CNN and LSTM (including inference).`));

const doc = new Document({
  creator: "Group MI302", title: "Spontaneous Combustion Early-Warning System",
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, font: FONT }, paragraph: { spacing: { before: 200, after: 90 }, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("MI 302 – Spontaneous Combustion Early-Warning System – page ", { size: 16 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
    children: kids }],
});
Packer.toBuffer(doc).then((buf) => { fs.mkdirSync("report", { recursive: true }); fs.writeFileSync("report/Final_Project_Report.docx", buf); console.log("report written"); });
