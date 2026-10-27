/* SYNTH Benchmark — interactive GitHub Pages dashboard */

const GENERATORS = [
  "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
  "WGAN_GP", "CTABGAN", "TabDDPM", "ForestDiffusion",
];

/** Maps dashboard filter labels → generators from each source folder. */
const GENERATOR_FAMILIES = {
  "All families": GENERATORS,
  "SDV models": ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"],
  "Other GANS": ["WGAN_GP", "CTABGAN"],
  "Diffusion GANs": ["TabDDPM", "ForestDiffusion"],
};

const GENERATOR_FAMILY_SOURCE = {
  "SDV models": "Generators/SDV models",
  "Other GANS": "Generators/Other GANS",
  "Diffusion GANs": "Generators/Diffusion GANs",
};

const GENERATOR_COLORS = {
  CTGAN: "#1f77b4",
  CopulaGAN: "#ff7f0e",
  TVAE: "#2ca02c",
  GaussianCopula: "#d62728",
  WGAN_GP: "#9467bd",
  CTABGAN: "#8c564b",
  TabDDPM: "#e377c2",
  ForestDiffusion: "#7f7f7f",
};

const UNIT_INTERVAL_METRICS = new Set([
  "Accuracy", "F1", "Precision", "Recall",
  "OverallScore", "WeightedScore", "Utility", "Privacy", "Fidelity",
  "Quality_Score", "NormalizedScore",
]);

const LOWER_BETTER_METRICS = new Set(["RMSE", "MAE", "Mean_Distance"]);

const PRIVACY_LABELS = {
  NNDR: "NNDR (nearest-neighbour distance ratio)",
  Mahalanobis_Distance: "Mahalanobis distance",
  Hungarian_Cosine_Similarity: "Hungarian cosine similarity",
  Cosine_Similarity: "Cosine similarity",
  MIA_AUC: "MIA AUC (membership inference)",
  Mean_Distance: "Mean matching distance",
  Median_Distance: "Median matching distance",
  Min_Distance: "Min matching distance",
  Max_Distance: "Max matching distance",
  Std_Distance: "Std matching distance",
  Num_Matches: "Matching subsample size",
};

const PRIVACY_PREFERRED_ORDER = [
  "Mahalanobis_Distance",
  "NNDR",
  "Hungarian_Cosine_Similarity",
  "Cosine_Similarity",
  "MIA_AUC",
  "Mean_Distance",
  "Median_Distance",
  "Min_Distance",
  "Max_Distance",
  "Std_Distance",
  "Num_Matches",
];

const FIDELITY_LABELS = {
  Quality_Score: "Quality score (SDMetrics)",
  Quality: "Quality score",
  KS_Complement: "KS complement (1 − KS statistic)",
  JS_Divergence: "Jensen–Shannon divergence",
  Gower_Distance: "Gower distance",
  Cosine_Similarity: "Cosine similarity",
  MMD: "Maximum mean discrepancy (MMD)",
  MMD_Multivariate: "MMD (multivariate)",
  Wasserstein_Distance: "Wasserstein distance",
  PCA_Correlation_Diff: "PCA correlation difference",
  PCA_Mean_Error: "PCA mean error",
};

const FIDELITY_PREFERRED_ORDER = [
  "Quality_Score",
  "KS_Complement",
  "JS_Divergence",
  "Gower_Distance",
  "Cosine_Similarity",
  "MMD",
  "MMD_Multivariate",
  "Wasserstein_Distance",
  "PCA_Correlation_Diff",
  "PCA_Mean_Error",
];

const DASH_FONT = '"Abadi MT Condensed Light", "Abadi MT", Abadi, Cabin, "Segoe UI", "Helvetica Neue", Arial, sans-serif';

const PLOTLY_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    color: "#eef2f7",
    family: DASH_FONT,
    size: 12,
  },
  margin: { l: 64, r: 28, t: 40, b: 92 },
  xaxis: {
    gridcolor: "rgba(154, 168, 188, 0.16)",
    zerolinecolor: "rgba(154, 168, 188, 0.28)",
    linecolor: "rgba(154, 168, 188, 0.35)",
    tickfont: { size: 11, family: DASH_FONT },
    title: { font: { size: 12, family: DASH_FONT } },
  },
  yaxis: {
    gridcolor: "rgba(154, 168, 188, 0.16)",
    zerolinecolor: "rgba(154, 168, 188, 0.28)",
    linecolor: "rgba(154, 168, 188, 0.35)",
    type: "linear",
    tickfont: { size: 11, family: DASH_FONT },
    title: { font: { size: 12, family: DASH_FONT } },
  },
  legend: {
    bgcolor: "rgba(26, 34, 46, 0.82)",
    bordercolor: "rgba(79, 140, 255, 0.35)",
    borderwidth: 1,
    font: { size: 11, family: DASH_FONT },
    orientation: "h",
    y: -0.18,
    x: 0.5,
    xanchor: "center",
  },
  colorway: ["#4f8cff", "#3ecf8e", "#f0b429", "#ff6b6b", "#9b8ec4", "#22d3ee", "#fb7185", "#94a3b8"],
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: true, displaylogo: false };

let DATA = {};

function toNum(value) {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  const nums = values.map(toNum).filter(v => v != null);
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function isPercentMetric(metric, task) {
  // R2 / error metrics are not percentages (R2 can be negative → bars vanish on 0–100% axes)
  if (["RMSE", "MAE", "MSE", "R2", "Mean_Distance"].includes(metric)) return false;
  if (/^(R2|RMSE|MAE|MSE)_/.test(metric)) return false;
  if (metric.includes("Drop") || metric.includes("Gap") || metric.includes("Increase")) return true;
  if (task === "classification" && ["Accuracy", "F1", "Precision", "Recall"].includes(metric)) return true;
  return false;
}

function toPercentValues(values) {
  return values.map(v => {
    const n = toNum(v);
    return n == null ? null : n * 100;
  });
}

function percentYAxis(values) {
  const nums = toPercentValues(values).filter(v => v != null);
  const minV = nums.length ? Math.min(...nums) : 0;
  const maxV = nums.length ? Math.max(...nums) : 100;
  const pad = Math.max((maxV - minV) * 0.12, 6);
  const lo = Math.max(0, minV - pad);
  const hi = Math.min(100, maxV + pad + 10);
  return {
    type: "linear",
    autorange: false,
    range: [lo, Math.max(hi, lo + 18)],
    tickmode: "linear",
    dtick: hi - lo <= 30 ? 5 : 10,
    ticksuffix: "%",
    automargin: true,
  };
}

function valueLabel(v, { asPercent = false } = {}) {
  if (v == null) return "";
  if (asPercent) {
    // Keep enough precision so 99.9522% does not display as 100.0%
    const oneDec = Number(v.toFixed(1));
    if ((oneDec === 100 || oneDec === 0) && Math.abs(v - oneDec) > 1e-9) {
      return `${v.toFixed(2)}%`;
    }
    return `${v.toFixed(1)}%`;
  }
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function numericYAxis(values, { asPercent = false, padRatio = 0.14 } = {}) {
  if (asPercent) return percentYAxis(values);
  const nums = values.map(toNum).filter(v => v != null);
  if (!nums.length) {
    return { type: "linear", range: [0, 1], autorange: false, automargin: true };
  }
  const minV = Math.min(...nums);
  const maxV = Math.max(...nums);
  const span = Math.max(maxV - minV, Math.abs(maxV - minV), Math.abs(maxV) * 0.05, Math.abs(minV) * 0.05, 1e-6);
  const pad = span * padRatio;
  // Cosine similarity can be negative — never floor at 0 or use nonnegative rangemode.
  const lo = minV >= 0 ? Math.max(0, minV - pad) : minV - pad;
  const hi = maxV <= 0 ? Math.min(0, maxV + pad) : maxV + pad + span * 0.08;
  const absMax = Math.max(Math.abs(minV), Math.abs(maxV));
  const tickformat = absMax >= 1000 ? ",.0f" : absMax >= 1 ? ".2f" : absMax >= 0.1 ? ".3f" : ".4f";
  return {
    type: "linear",
    autorange: false,
    rangemode: "normal",
    range: [lo, hi],
    automargin: true,
    tickformat,
    zeroline: true,
    zerolinecolor: "#64748b",
  };
}

function privacyMetricLabel(metricId) {
  return PRIVACY_LABELS[metricId] || String(metricId).replace(/_/g, " ");
}

function fidelityMetricLabel(metricId) {
  return FIDELITY_LABELS[metricId] || String(metricId).replace(/_/g, " ");
}

function fidelityUsesPercent(metricId, catalogEntry) {
  return Boolean(catalogEntry?.is_unit_interval)
    || ["Quality_Score", "Quality", "KS_Complement", "JS_Divergence", "Gower_Distance"].includes(metricId);
}

function fidelityScoreFor(row) {
  return toNum(row.NormalizedScore ?? row.Mean ?? row.MetricValue ?? row.Value);
}

function privacyUsesPercent(metricId, catalogEntry) {
  return Boolean(catalogEntry?.is_similarity)
    || metricId === "MIA_AUC"
    || metricId.includes("Cosine");
}

function valuesForAllGenerators(byGen) {
  return GENERATORS.map(g => (byGen[g]?.length ? mean(byGen[g]) : null));
}

function generatorXAxis() {
  return {
    type: "category",
    categoryorder: "array",
    categoryarray: GENERATORS,
    tickangle: -25,
  };
}

function generatorYAxis() {
  return {
    type: "category",
    categoryorder: "array",
    categoryarray: [...GENERATORS].reverse(),
  };
}

function missingBarLabelTrace(values, asPercent) {
  const nums = GENERATORS.map((_, i) => {
    const raw = Array.isArray(values) ? values[i] : null;
    if (raw == null) return null;
    return asPercent ? toNum(raw) * 100 : toNum(raw);
  });
  const missingGens = GENERATORS.filter((_, i) => nums[i] == null);
  if (!missingGens.length) return null;

  const present = nums.filter(v => v != null);
  let yPos = 0;
  if (present.length) {
    const minP = Math.min(...present);
    yPos = asPercent ? minP - 4 : minP - Math.abs(minP) * 0.08 - 0.002;
  }

  return {
    x: missingGens,
    y: missingGens.map(() => yPos),
    type: "scatter",
    mode: "text",
    text: missingGens.map(() => "N/A"),
    textfont: { color: "#94a3b8", size: 12 },
    hoverinfo: "skip",
    showlegend: false,
  };
}

function privacyBarTrace({ x, y, color, asPercent }) {
  const nums = GENERATORS.map((gen, i) => {
    const raw = Array.isArray(y) ? y[i] : null;
    if (raw == null) return null;
    return asPercent ? toNum(raw) * 100 : toNum(raw);
  });
  const barColor = nums.map(v => {
    if (v == null) return "rgba(100,116,139,0.35)";
    if (!asPercent && v < 0) return "#64748b";
    return color;
  });
  return {
    x: GENERATORS,
    y: nums,
    type: "bar",
    marker: { color: barColor },
    text: nums.map(v => (v == null ? "N/A" : valueLabel(v, { asPercent }))),
    textposition: "outside",
    cliponaxis: false,
    constraintext: "none",
    hovertemplate: asPercent
      ? "%{x}: %{y:.3f}%<extra></extra>"
      : "%{x}: %{y:.4f}<extra></extra>",
  };
}

function barChartTrace({ x, y, name, color, asPercent = true, decimals = 1 }) {
  const categories = Array.isArray(x) && x.length ? x : GENERATORS;
  const values = categories.map((_, i) => {
    const raw = Array.isArray(y) ? y[i] : null;
    if (raw == null) return null;
    return asPercent ? toNum(raw) * 100 : toNum(raw);
  });
  const barColor = values.map(v => {
    if (v == null) return "rgba(100,116,139,0.35)";
    if (!asPercent && v < 0) return "#f59e0b";
    return color;
  });
  return {
    x: categories,
    y: values,
    name,
    type: "bar",
    marker: { color: barColor },
    text: values.map(v => (v == null ? "N/A" : asPercent ? `${v.toFixed(decimals)}%` : v.toFixed(decimals))),
    textposition: "outside",
    cliponaxis: false,
    hovertemplate: asPercent
      ? `<b>${name}</b><br>%{x}: %{y:.2f}%<extra></extra>`
      : `<b>${name}</b><br>%{x}: %{y:.4f}<extra></extra>`,
  };
}

function classifierMean(rows, classifier, evaluationType) {
  const vals = rows
    .filter(r => r.Classifier === classifier && r.EvaluationType === evaluationType)
    .map(r => toNum(r.Mean));
  return mean(vals);
}

function isAllDatasets(dataset) {
  return !dataset || dataset === "All datasets";
}

function datasetProblemType(dataset) {
  if (!dataset) return "unknown";
  const match = String(dataset).match(/^(\d+)/);
  if (!match) return "unknown";
  return parseInt(match[1], 10) >= 10 ? "regression" : "classification";
}

function matchesProblemType(dataset, problemType) {
  if (!problemType || problemType === "all") return true;
  return datasetProblemType(dataset) === problemType;
}

function filterDatasetsByProblem(datasets, problemType) {
  return datasets.filter(ds => matchesProblemType(ds, problemType));
}

function problemTypeLabel(problemType) {
  if (problemType === "classification") return "classification";
  if (problemType === "regression") return "regression";
  return "all";
}

function setupProblemTypeSelect(id, onChange, defaultValue = "all") {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = [
    { value: "all", label: "All problem types" },
    { value: "classification", label: "Classification" },
    { value: "regression", label: "Regression" },
  ].map(o => `<option value="${o.value}">${o.label}</option>`).join("");
  sel.value = defaultValue;
  sel.addEventListener("change", onChange);
}

function axisSpec(metric, values, { clampUnit = false } = {}) {
  const nums = values.map(toNum).filter(v => v != null);
  const minV = nums.length ? Math.min(...nums) : 0;
  const maxV = nums.length ? Math.max(...nums) : 1;

  // R² can be strongly negative — use an auto range that includes negatives
  if (metric === "R2") {
    const span = Math.max(maxV - minV, 0.15);
    const lo = minV - span * 0.12;
    const hi = Math.max(maxV + span * 0.12, 0.05);
    return {
      type: "linear",
      range: [lo, hi],
      zeroline: true,
      zerolinecolor: "#94a3b8",
      tickformat: ".2f",
      title: { text: "R²" },
      automargin: true,
    };
  }

  if (
    (UNIT_INTERVAL_METRICS.has(metric) || clampUnit)
    && minV >= -0.02
    && maxV <= 1.05
  ) {
    return {
      type: "linear",
      range: [0, 1],
      tickmode: "linear",
      dtick: 0.1,
      tickformat: ".0%",
      automargin: true,
    };
  }

  if (metric.includes("Drop") || metric.includes("Gap")) {
    const hi = Math.min(1, Math.max(0.1, maxV * 1.12));
    return {
      type: "linear",
      range: [0, hi],
      tickmode: "linear",
      dtick: hi <= 0.2 ? 0.05 : 0.1,
      tickformat: ".2f",
      automargin: true,
    };
  }

  if (LOWER_BETTER_METRICS.has(metric)) {
    const span = Math.max(maxV - minV, maxV * 0.05, 1e-6);
    const lo = Math.max(0, minV - span * 0.08);
    const hi = maxV + span * 0.12;
    return {
      type: maxV / Math.max(lo, 1e-9) > 500 ? "log" : "linear",
      range: maxV / Math.max(lo, 1e-9) > 500 ? undefined : [lo, hi],
      tickformat: maxV >= 100 ? ",.0f" : ".3f",
      automargin: true,
    };
  }

  const span = Math.max(maxV - minV, 0.05);
  const lo = minV < 0 ? minV - span * 0.08 : Math.max(0, minV - span * 0.08);
  return {
    type: "linear",
    range: [lo, maxV + span * 0.12],
    tickformat: ".3f",
    automargin: true,
  };
}

const DATA_VERSION = "20260721e";

async function loadJSON(name) {
  const url = `data/${name}?v=${DATA_VERSION}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      if (name === "meta.json" || name === "statistics.json" || name === "correlation_tradeoff.json") return {};
      return [];
    }
    return res.json();
  } catch {
    if (name === "meta.json" || name === "statistics.json" || name === "correlation_tradeoff.json") return {};
    return [];
  }
}

async function loadAllData() {
  const [meta, utilityAgg, utilityClf, utilityReg, utilityGaps, fidelity, fidelityMetrics, privacy, privacyMetrics, tradeoff,
         weighted, borda, statistics, notebookErrors, coverage, correlationTradeoff] = await Promise.all([
    loadJSON("meta.json"),
    loadJSON("utility_agg.json"),
    loadJSON("utility_classifier.json"),
    loadJSON("utility_regressor.json"),
    loadJSON("utility_gaps.json"),
    loadJSON("fidelity.json"),
    loadJSON("fidelity_metrics.json"),
    loadJSON("privacy.json"),
    loadJSON("privacy_metrics.json"),
    loadJSON("tradeoff.json"),
    loadJSON("rankings_weighted.json"),
    loadJSON("rankings_borda.json"),
    loadJSON("statistics.json"),
    loadJSON("notebook_error_stats.json"),
    loadJSON("coverage.json"),
    loadJSON("correlation_tradeoff.json"),
  ]);

  const stats = statistics && typeof statistics === "object" ? { ...statistics } : {};
  // GitHub Pages sometimes serves a stale statistics.json with empty pca_errors.
  // Fall back to the dedicated notebook export when needed.
  if ((!Array.isArray(stats.pca_errors) || stats.pca_errors.length === 0)
      && Array.isArray(notebookErrors) && notebookErrors.length) {
    stats.pca_errors = notebookErrors;
  }

  DATA = {
    meta: meta || {},
    utilityAgg: utilityAgg || [],
    utilityClf: utilityClf || [],
    utilityReg: utilityReg || [],
    utilityGaps: utilityGaps || [],
    fidelity: fidelity || [],
    fidelityMetrics: fidelityMetrics || [],
    privacy: privacy || [],
    privacyMetrics: privacyMetrics || [],
    tradeoff: tradeoff || [],
    weighted: weighted || [],
    borda: borda || [],
    statistics: stats,
    coverage: coverage || [],
    correlationTradeoff: correlationTradeoff || { analyses: [] },
  };
}

function unique(arr, key) {
  return [...new Set(arr.map(r => r[key]).filter(Boolean))].sort();
}

/** Sort numbered datasets 1…15 by prefix, not lexicographically. */
function uniqueDatasets(rows) {
  return [...new Set(rows.map(r => r.Dataset).filter(Boolean))].sort((a, b) => {
    const na = parseInt(String(a).match(/^(\d+)/)?.[1] || "999", 10);
    const nb = parseInt(String(b).match(/^(\d+)/)?.[1] || "999", 10);
    if (na !== nb) return na - nb;
    return String(a).localeCompare(String(b));
  });
}

function shortDatasetLabel(dataset) {
  const raw = String(dataset).replace(/^\d+\.\s*/, "").trim();
  const aliases = {
    "Concrete Compressive Strength": "Concrete",
    "Real Estate Valuation": "Real Estate",
    "MAGIC Gamma Telescope": "MAGIC",
    "CDC diabetes dataset": "CDC Diabetes",
    "Forest cover dataset": "Forest Cover",
    "Mushroom dataset": "Mushroom",
    "Wine dataset": "Wine",
    "Bank Markting": "Bank Marketing",
    "Metro interstate": "Metro",
    "online shopping": "E-shop",
    "Air Quality": "Air Quality",
    Alzhimers: "Alzheimer",
    Cancer: "Cancer",
    Adult: "Adult",
    "Energy Efficiency": "Energy",
  };
  return aliases[raw] || raw;
}

function plot(id, traces, layout = {}, config = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  const isHeatmap = Array.isArray(traces) && traces.some(t => t && t.type === "heatmap");
  const yaxis = { ...PLOTLY_LAYOUT.yaxis, ...(layout.yaxis || {}) };
  // Heatmaps use categorical generator/dataset labels — never force a linear Y axis.
  if (isHeatmap && yaxis.type !== "category") {
    yaxis.type = "category";
    yaxis.autorange = true;
    delete yaxis.autotypenumbers;
  } else if (yaxis.type !== "category") {
    yaxis.type = "linear";
    yaxis.autotypenumbers = "strict";
  }
  const xaxis = { ...PLOTLY_LAYOUT.xaxis, ...(layout.xaxis || {}) };
  if (isHeatmap && xaxis.type !== "category" && xaxis.type !== "linear") {
    xaxis.type = "category";
  }
  const legend = { ...PLOTLY_LAYOUT.legend, ...(layout.legend || {}) };
  const margin = { ...PLOTLY_LAYOUT.margin, ...(layout.margin || {}) };
  if (legend.orientation === "h" && margin.b < 96) margin.b = 96;
  Plotly.newPlot(
    id,
    traces,
    { ...PLOTLY_LAYOUT, ...layout, xaxis, yaxis, legend, margin },
    { ...PLOTLY_CONFIG, ...config },
  );
}

function fillSelect(id, options, defaultVal) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = options.map(o => `<option value="${o}">${o}</option>`).join("");
  if (defaultVal && options.includes(defaultVal)) sel.value = defaultVal;
}

function aggregateGeneratorMeans(rows, evaluationType) {
  const buckets = {};
  rows
    .filter(r => r.EvaluationType === evaluationType)
    .forEach(r => {
      const value = toNum(r.Mean);
      if (value == null) return;
      if (!buckets[r.Generator]) buckets[r.Generator] = [];
      buckets[r.Generator].push(value);
    });
  return buckets;
}

function renderMetrics() {
  const m = DATA.meta;
  document.getElementById("metric-datasets").textContent = m.n_datasets ?? "—";
  document.getElementById("metric-generators").textContent = (m.generators || GENERATORS).length;
  document.getElementById("metric-files").textContent = m.n_files ?? "—";
  document.getElementById("metric-utility").textContent = m.n_utility_rows?.toLocaleString() ?? "—";
}

function renderOverview() {
  const gaps = DATA.utilityGaps.filter(r => r.Metric === "Accuracy_Drop" || r.Metric === "R2_Drop");
  const datasets = uniqueDatasets(gaps);
  const generators = GENERATORS.filter(g => gaps.some(r => r.Generator === g));

  const z = generators.map(gen =>
    datasets.map(ds => {
      const row = gaps.find(r => r.Generator === gen && r.Dataset === ds);
      return row ? toNum(row.Mean) : null;
    }),
  );

  const zText = z.map(row =>
    row.map(v => (v == null ? "" : `${Math.round(v * 100)}%`)),
  );

  plot("overview-heatmap", [{
    type: "heatmap",
    x: datasets.map(shortDatasetLabel),
    y: generators,
    z,
    zmin: 0,
    zmax: 1,
    colorscale: [
      [0, "#2f5d50"],
      [0.35, "#8fad7a"],
      [0.55, "#c4a35a"],
      [0.75, "#d08b6b"],
      [1, "#9b4d4d"],
    ],
    hovertemplate: "%{y} · %{x}<br>Gap: %{z:.1%}<extra></extra>",
    colorbar: {
      title: { text: "Utility gap", font: { size: 12 } },
      tickformat: ".0%",
      thickness: 14,
      len: 0.75,
      outlinewidth: 0,
    },
    text: zText,
    texttemplate: "%{text}",
    textfont: { size: 10, color: "#f8fafc", family: DASH_FONT },
    xgap: 2,
    ygap: 2,
  }], {
    title: { text: "" },
    height: 500,
    margin: { l: 140, r: 90, t: 24, b: 110 },
    xaxis: { type: "category", tickangle: -35, automargin: true },
    yaxis: { ...generatorYAxis(), automargin: true },
    legend: { orientation: "v", y: 1, x: 1.02 },
  });

  const cov = DATA.coverage;
  if (cov.length) {
    const dsU = uniqueDatasets(cov);
    const zCov = GENERATORS.map(g =>
      dsU.map(ds => {
        const r = cov.find(x => x.Generator === g && x.Dataset === ds);
        return r ? (toNum(r.Available) ? 1 : 0) : 0;
      }),
    );
    const textMat = zCov.map(row => row.map(v => (v ? "✓" : "✗")));
    const hoverMat = zCov.map((row, gi) =>
      row.map((v, di) => (v ? "Available" : "Missing")),
    );

    plot("overview-coverage", [{
      type: "heatmap",
      x: dsU.map(shortDatasetLabel),
      y: GENERATORS,
      z: zCov,
      zmin: 0,
      zmax: 1,
      colorscale: [
        [0, "#1e293b"],
        [0.5, "#1e293b"],
        [0.5, "#059669"],
        [1, "#34d399"],
      ],
      showscale: true,
      colorbar: {
        title: { text: "" },
        tickmode: "array",
        tickvals: [0, 1],
        ticktext: ["Missing", "Available"],
        thickness: 16,
        len: 0.45,
        outlinewidth: 0,
      },
      text: textMat,
      texttemplate: "%{text}",
      textfont: { size: 12, color: "#f8fafc" },
      customdata: hoverMat,
      hovertemplate: "<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
      xgap: 3,
      ygap: 3,
    }], {
      title: { text: "" },
      height: 480,
      margin: { l: 130, r: 100, t: 20, b: 100 },
      xaxis: {
        type: "category",
        tickangle: -40,
        automargin: true,
        side: "bottom",
      },
      yaxis: {
        ...generatorYAxis(),
        automargin: true,
      },
    });
  }
}

function metricsForTask(task) {
  return task === "regression"
    ? ["R2", "RMSE", "MAE"]
    : ["Accuracy", "F1", "Precision", "Recall"];
}

function syncUtilityMetricOptions(task, preferred) {
  const options = metricsForTask(task);
  const current = preferred || document.getElementById("filter-metric")?.value;
  const next = options.includes(current) ? current : options[0];
  fillSelect("filter-metric", options, next);
  return next;
}

function regressorMean(rows, regressor, evaluationType) {
  const vals = rows
    .filter(r => r.Regressor === regressor && r.EvaluationType === evaluationType)
    .map(r => toNum(r.Mean));
  return mean(vals);
}

/** Fidelity-style utility bar + labeled performance heatmap (TSTR/TRTR — not Accuracy Drop). */
function renderUtilityBarAndHeatmap({
  gens,
  task,
  metric,
  evalType,
  dataset,
  allDs,
  barPlotId,
  heatPlotId,
  titleElId,
  heatTitleElId,
  sourceNote = "",
}) {
  const usePercent = isPercentMetric(metric, task);
  const higherBetter = !["RMSE", "MAE", "MSE"].includes(metric);
  const metricRows = DATA.utilityAgg.filter(r =>
    r.TaskType === task
    && r.Metric === metric
    && r.EvaluationType === evalType
    && gens.includes(r.Generator),
  );
  const datasets = uniqueDatasets(metricRows);
  const rows = metricRows.filter(r => allDs || r.Dataset === dataset);

  const titleEl = document.getElementById(titleElId);
  if (titleEl) titleEl.textContent = `${metric} (${evalType}) by generator`;
  const heatTitleEl = heatTitleElId ? document.getElementById(heatTitleElId) : null;
  if (heatTitleEl) heatTitleEl.textContent = `${metric} (${evalType}) · dataset × generator`;

  if (!rows.length && !metricRows.length) {
    plot(barPlotId, [], { title: `No ${metric} ${evalType} data`, height: 220 });
    plot(heatPlotId, [], { title: "No heatmap data", height: 220 });
    return;
  }

  const byGen = {};
  rows.forEach(r => {
    const value = toNum(r.Mean);
    if (!r.Generator || value == null) return;
    if (!byGen[r.Generator]) byGen[r.Generator] = [];
    byGen[r.Generator].push(value);
  });
  const barValues = gens.map(g => (byGen[g]?.length ? mean(byGen[g]) : null));
  const missingCount = barValues.filter(v => v == null).length;
  const missingNote = missingCount ? ` · ${missingCount} generator(s) not evaluated` : "";
  const direction = higherBetter ? "higher = better utility" : "lower = better utility";
  const naTrace = missingBarLabelTrace(barValues, usePercent);

  plot(barPlotId, [
    privacyBarTrace({ x: gens, y: barValues, color: "#3b82f6", asPercent: usePercent }),
    ...(naTrace ? [naTrace] : []),
  ], {
    title: `${metric} · ${evalType}${allDs ? " (mean over datasets)" : ""}${sourceNote} · ${direction}${missingNote}`,
    height: 460,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: numericYAxis(barValues, { asPercent: usePercent }),
    xaxis: generatorXAxis(),
    showlegend: false,
  });

  const heatDs = datasets.length ? datasets : uniqueDatasets(metricRows);
  const heatGens = gens.filter(g => metricRows.some(r => r.Generator === g));
  const z = heatGens.map(g => heatDs.map(d => {
    const matches = metricRows.filter(r => r.Generator === g && r.Dataset === d);
    return mean(matches.map(r => toNum(r.Mean)));
  }));
  const zText = z.map(row =>
    row.map(v => {
      if (v == null) return "";
      return usePercent ? `${Math.round(v * 100)}%` : v.toFixed(2);
    }),
  );
  const zFlat = z.flat().filter(v => v != null);
  const heatmapScale = usePercent
    ? { zmin: 0, zmax: 1 }
    : {
      zmin: zFlat.length ? Math.min(...zFlat) : undefined,
      zmax: zFlat.length ? Math.max(...zFlat) : undefined,
    };

  plot(heatPlotId, heatDs.length && heatGens.length ? [{
    type: "heatmap",
    x: heatDs.map(shortDatasetLabel),
    y: heatGens,
    z,
    ...heatmapScale,
    colorscale: higherBetter
      ? [
        [0, "#9b4d4d"],
        [0.35, "#d08b6b"],
        [0.55, "#c4a35a"],
        [0.75, "#8fad7a"],
        [1, "#2f5d50"],
      ]
      : [
        [0, "#2f5d50"],
        [0.35, "#8fad7a"],
        [0.55, "#c4a35a"],
        [0.75, "#d08b6b"],
        [1, "#9b4d4d"],
      ],
    hovertemplate: `%{y} · %{x}<br>${metric} (${evalType}): %{z:.3f}<extra></extra>`,
    colorbar: {
      title: { text: `${metric} (${evalType})`, font: { size: 12 } },
      tickformat: usePercent ? ".0%" : ".2f",
      thickness: 14,
      len: 0.75,
      outlinewidth: 0,
    },
    text: zText,
    texttemplate: "%{text}",
    textfont: { size: 10, color: "#f8fafc", family: DASH_FONT },
    xgap: 2,
    ygap: 2,
  }] : [], {
    title: `${metric} · ${evalType} · dataset × generator${sourceNote}`,
    height: Math.max(360, 80 + heatGens.length * 42),
    margin: { l: 140, r: 90, t: 44, b: 110 },
    xaxis: { type: "category", tickangle: -35, automargin: true },
    yaxis: { ...generatorYAxis(), automargin: true },
  });
}

function renderUtility() {
  const task = document.getElementById("filter-task")?.value || "classification";
  const metric = syncUtilityMetricOptions(task);
  const evalType = document.getElementById("filter-utility-eval")?.value || "TSTR";
  const dataset = document.getElementById("filter-dataset")?.value;
  const allDs = isAllDatasets(dataset);

  const datasets = uniqueDatasets(DATA.utilityAgg.filter(r => r.TaskType === task));
  fillSelect("filter-dataset", ["All datasets", ...datasets], dataset || "All datasets");

  renderUtilityBarAndHeatmap({
    gens: GENERATORS,
    task,
    metric,
    evalType,
    dataset: document.getElementById("filter-dataset")?.value,
    allDs: isAllDatasets(document.getElementById("filter-dataset")?.value),
    barPlotId: "utility-bar",
    heatPlotId: "utility-heatmap",
    titleElId: "utility-bar-title",
    heatTitleElId: "utility-heat-title",
  });

  const rows = DATA.utilityAgg.filter(r =>
    r.TaskType === task
    && r.Metric === metric
    && (isAllDatasets(document.getElementById("filter-dataset")?.value)
      || r.Dataset === document.getElementById("filter-dataset")?.value),
  );

  const trtrBuckets = aggregateGeneratorMeans(rows, "TRTR");
  const tstrBuckets = aggregateGeneratorMeans(rows, "TSTR");
  const gens = GENERATORS;
  const trtr = valuesForAllGenerators(trtrBuckets);
  const tstr = valuesForAllGenerators(tstrBuckets);
  const yValues = [...trtr, ...tstr].filter(v => v != null);
  const usePercent = isPercentMetric(metric, task);
  const hasBars = yValues.length > 0;
  const selectedDataset = document.getElementById("filter-dataset")?.value;
  const allDsSelected = isAllDatasets(selectedDataset);

  plot("utility-trtr-tstr", hasBars ? [
    barChartTrace({ x: gens, y: trtr, name: "TRTR", color: "#3b82f6", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
    barChartTrace({ x: gens, y: tstr, name: "TSTR", color: "#10b981", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
  ] : [], {
    barmode: "group",
    bargap: 0.18,
    bargroupgap: 0.08,
    title: hasBars
      ? `${metric}: TRTR vs TSTR (${task}${allDsSelected ? ", mean over datasets" : ""})`
      : `No ${metric} data for this ${task} selection`,
    height: 480,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: usePercent ? percentYAxis(yValues) : axisSpec(metric, yValues),
    xaxis: generatorXAxis(),
  });

  const modelTitleEl = document.getElementById("utility-detail-title")
    || document.querySelector("#panel-utility .chart-card:nth-child(2) h3");
  const isRegression = task === "regression";
  if (modelTitleEl) modelTitleEl.textContent = isRegression ? "By regressor" : "By classifier";

  const detailSource = isRegression ? (DATA.utilityReg || []) : (DATA.utilityClf || []);
  const modelKey = isRegression ? "Regressor" : "Classifier";
  const detailRows = detailSource.filter(r =>
    r.Metric === metric && (allDsSelected || r.Dataset === selectedDataset),
  );

  const gensWithDetail = GENERATORS.filter(g => detailRows.some(r => r.Generator === g));
  const genSel = document.getElementById("filter-generator")?.value;
  const gen = gensWithDetail.includes(genSel) ? genSel : (gensWithDetail[0] || GENERATORS[0]);
  fillSelect("filter-generator", gensWithDetail.length ? gensWithDetail : GENERATORS, gen);

  const ds = allDsSelected ? (unique(detailRows, "Dataset")[0] || "") : selectedDataset;
  const detail = detailRows.filter(r => r.Generator === gen && r.Dataset === ds);
  const models = unique(detail, modelKey);
  const trtrModel = models.map(m => (
    isRegression ? regressorMean(detail, m, "TRTR") : classifierMean(detail, m, "TRTR")
  ));
  const tstrModel = models.map(m => (
    isRegression ? regressorMean(detail, m, "TSTR") : classifierMean(detail, m, "TSTR")
  ));
  const modelValues = [...trtrModel, ...tstrModel].filter(v => v != null);
  const hasModelBars = modelValues.length > 0;

  plot("utility-classifier", hasModelBars ? [
    barChartTrace({ x: models, y: trtrModel, name: "TRTR", color: "#3b82f6", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
    barChartTrace({ x: models, y: tstrModel, name: "TSTR", color: "#10b981", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
  ] : [], {
    barmode: "group",
    bargap: 0.12,
    bargroupgap: 0.06,
    title: hasModelBars
      ? `${ds} — ${gen}: ${metric} by ${isRegression ? "regressor" : "classifier"}`
      : `No ${isRegression ? "regressor" : "classifier"} breakdown for ${metric}`,
    height: 480,
    margin: { t: 44, b: 110, l: 64, r: 24 },
    xaxis: { tickangle: -45, type: "category" },
    yaxis: usePercent ? percentYAxis(modelValues) : axisSpec(metric, modelValues),
  });
}

function generatorsForFamily(family) {
  return GENERATOR_FAMILIES[family] || GENERATORS;
}

function syncLeakUtilityMetricOptions(task, preferred) {
  const options = metricsForTask(task);
  const current = preferred || document.getElementById("filter-leak-metric")?.value;
  const next = options.includes(current) ? current : options[0];
  fillSelect("filter-leak-metric", options, next);
  return next;
}

/** Accuracy_Drop / R2_Drop heatmap (same style as Overview / dataleak gap chart). */
function renderUtilityGapHeatmap({ plotId, gens, task, titleSuffix = "" }) {
  const isRegression = task === "regression";
  const gapMetric = isRegression ? "R2_Drop" : "Accuracy_Drop";
  const gaps = (DATA.utilityGaps || []).filter(r =>
    r.Metric === gapMetric && gens.includes(r.Generator),
  );
  const heatDs = uniqueDatasets(gaps);
  const heatGens = gens.filter(g => gaps.some(r => r.Generator === g));
  if (!heatDs.length || !heatGens.length) {
    plot(plotId, [], {
      title: `No ${gapMetric} data${titleSuffix}`,
      height: 280,
    });
    return;
  }

  const z = heatGens.map(g => heatDs.map(d => {
    const row = gaps.find(r => r.Generator === g && r.Dataset === d);
    return row ? toNum(row.Mean) : null;
  }));
  const zText = z.map(row =>
    row.map(v => (v == null ? "" : `${Math.round(v * 100)}%`)),
  );

  plot(plotId, [{
    type: "heatmap",
    x: heatDs.map(shortDatasetLabel),
    y: heatGens,
    z,
    zmin: 0,
    zmax: 1,
    colorscale: [
      [0, "#2f5d50"],
      [0.35, "#8fad7a"],
      [0.55, "#c4a35a"],
      [0.75, "#d08b6b"],
      [1, "#9b4d4d"],
    ],
    hovertemplate: `%{y} · %{x}<br>${gapMetric}: %{z:.1%}<extra></extra>`,
    colorbar: {
      title: { text: gapMetric.replace(/_/g, " "), font: { size: 12 } },
      tickformat: ".0%",
      thickness: 14,
      len: 0.75,
      outlinewidth: 0,
    },
    text: zText,
    texttemplate: "%{text}",
    textfont: { size: 10, color: "#f8fafc", family: DASH_FONT },
    xgap: 2,
    ygap: 2,
  }], {
    title: `${gapMetric.replace(/_/g, " ")}${titleSuffix}`,
    height: Math.max(320, 80 + heatGens.length * 42),
    margin: { l: 140, r: 90, t: 44, b: 110 },
    xaxis: { type: "category", tickangle: -35, automargin: true },
    yaxis: { ...generatorYAxis(), automargin: true },
  });
}

function renderUtilityDataleak() {
  const family = document.getElementById("filter-leak-family")?.value || "All families";
  const task = document.getElementById("filter-leak-task")?.value || "classification";
  const metric = syncLeakUtilityMetricOptions(task);
  const evalType = document.getElementById("filter-leak-eval")?.value || "TSTR";
  const dataset = document.getElementById("filter-leak-dataset")?.value;
  const familyGens = generatorsForFamily(family);
  const sourceNote = GENERATOR_FAMILY_SOURCE[family]
    ? ` · ${GENERATOR_FAMILY_SOURCE[family]}`
    : " · SDV models + Other GANS + Diffusion GANs";

  const datasets = uniqueDatasets(
    DATA.utilityAgg.filter(r => r.TaskType === task && familyGens.includes(r.Generator)),
  );
  fillSelect("filter-leak-dataset", ["All datasets", ...datasets], dataset || "All datasets");
  const selectedDataset = document.getElementById("filter-leak-dataset")?.value;
  const allDs = isAllDatasets(selectedDataset);

  renderUtilityBarAndHeatmap({
    gens: familyGens,
    task,
    metric,
    evalType,
    dataset: selectedDataset,
    allDs,
    barPlotId: "leak-utility-bar",
    heatPlotId: "leak-utility-heatmap",
    titleElId: "leak-utility-bar-title",
    sourceNote,
  });

  const rows = DATA.utilityAgg.filter(r =>
    r.TaskType === task
    && r.Metric === metric
    && familyGens.includes(r.Generator)
    && (allDs || r.Dataset === selectedDataset),
  );

  const trtrBuckets = aggregateGeneratorMeans(rows, "TRTR");
  const tstrBuckets = aggregateGeneratorMeans(rows, "TSTR");
  const gens = familyGens;
  const trtr = gens.map(g => (trtrBuckets[g]?.length ? mean(trtrBuckets[g]) : null));
  const tstr = gens.map(g => (tstrBuckets[g]?.length ? mean(tstrBuckets[g]) : null));
  const yValues = [...trtr, ...tstr].filter(v => v != null);
  const usePercent = isPercentMetric(metric, task);
  const hasBars = yValues.length > 0;
  const missingCount = gens.filter((g, i) => trtr[i] == null && tstr[i] == null).length;
  const missingNote = missingCount ? ` · ${missingCount} generator(s) not evaluated` : "";

  plot("leak-utility-trtr-tstr", hasBars ? [
    barChartTrace({ x: gens, y: trtr, name: "TRTR", color: "#3b82f6", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
    barChartTrace({ x: gens, y: tstr, name: "TSTR", color: "#10b981", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
  ] : [], {
    barmode: "group",
    bargap: 0.18,
    bargroupgap: 0.08,
    title: hasBars
      ? `${metric}: TRTR vs TSTR (${family}${allDs ? ", mean over datasets" : ""})${sourceNote}${missingNote}`
      : `No ${metric} data for ${family} · ${task}`,
    height: 480,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: usePercent ? percentYAxis(yValues) : axisSpec(metric, yValues),
    xaxis: generatorXAxis(),
  });

  const isRegression = task === "regression";
  const modelTitleEl = document.getElementById("leak-utility-detail-title");
  if (modelTitleEl) modelTitleEl.textContent = isRegression ? "By regressor" : "By classifier";

  const detailSource = isRegression ? (DATA.utilityReg || []) : (DATA.utilityClf || []);
  const modelKey = isRegression ? "Regressor" : "Classifier";
  const detailRows = detailSource.filter(r =>
    r.Metric === metric
    && familyGens.includes(r.Generator)
    && (allDs || r.Dataset === selectedDataset),
  );

  const gensWithDetail = familyGens.filter(g => detailRows.some(r => r.Generator === g));
  const genSel = document.getElementById("filter-leak-generator")?.value;
  const gen = gensWithDetail.includes(genSel) ? genSel : (gensWithDetail[0] || familyGens[0]);
  fillSelect("filter-leak-generator", gensWithDetail.length ? gensWithDetail : familyGens, gen);

  const ds = allDs ? (unique(detailRows, "Dataset")[0] || "") : selectedDataset;
  const detail = detailRows.filter(r => r.Generator === gen && r.Dataset === ds);
  const models = unique(detail, modelKey);
  const trtrModel = models.map(m => (
    isRegression ? regressorMean(detail, m, "TRTR") : classifierMean(detail, m, "TRTR")
  ));
  const tstrModel = models.map(m => (
    isRegression ? regressorMean(detail, m, "TSTR") : classifierMean(detail, m, "TSTR")
  ));
  const modelValues = [...trtrModel, ...tstrModel].filter(v => v != null);
  const hasModelBars = modelValues.length > 0;

  plot("leak-utility-classifier", hasModelBars ? [
    barChartTrace({ x: models, y: trtrModel, name: "TRTR", color: "#3b82f6", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
    barChartTrace({ x: models, y: tstrModel, name: "TSTR", color: "#10b981", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
  ] : [], {
    barmode: "group",
    bargap: 0.12,
    bargroupgap: 0.06,
    title: hasModelBars
      ? `${ds} — ${gen}: ${metric} by ${isRegression ? "regressor" : "classifier"}`
      : `No ${isRegression ? "regressor" : "classifier"} breakdown for ${metric}`,
    height: 480,
    margin: { t: 44, b: 110, l: 64, r: 24 },
    xaxis: { tickangle: -45, type: "category" },
    yaxis: usePercent ? percentYAxis(modelValues) : axisSpec(metric, modelValues),
  });

  // Gap heatmap for the selected family (Accuracy_Drop / R2_Drop)
  renderUtilityGapHeatmap({
    plotId: "leak-utility-gap-heatmap",
    gens: familyGens,
    task,
    titleSuffix: ` · ${family}${sourceNote}`,
  });
}

function setupUtilityDataleakFilters() {
  const familyEl = document.getElementById("filter-leak-family");
  if (familyEl) {
    fillSelect("filter-leak-family", Object.keys(GENERATOR_FAMILIES), "All families");
    familyEl.addEventListener("change", renderUtilityDataleak);
  }
  const taskEl = document.getElementById("filter-leak-task");
  if (taskEl) {
    fillSelect("filter-leak-task", ["classification", "regression"], "classification");
    taskEl.addEventListener("change", () => {
      syncLeakUtilityMetricOptions(taskEl.value);
      renderUtilityDataleak();
    });
  }
  fillSelect("filter-leak-eval", ["TSTR", "TRTR"], "TSTR");
  ["filter-leak-metric", "filter-leak-eval", "filter-leak-dataset", "filter-leak-generator"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderUtilityDataleak);
  });
  syncLeakUtilityMetricOptions("classification", "Accuracy");
}

function renderFidelity() {
  const metric = document.getElementById("filter-fidelity-metric")?.value;
  const problemType = document.getElementById("filter-fidelity-problem")?.value || "all";
  const dataset = document.getElementById("filter-fidelity-dataset")?.value;
  const catalog = DATA.fidelityMetrics || [];
  const catalogEntry = catalog.find(m => m.id === metric) || {};
  const label = fidelityMetricLabel(metric);
  const asPercent = fidelityUsesPercent(metric, catalogEntry);
  const higherBetter = catalogEntry.higher_is_better !== false;

  const metricRows = DATA.fidelity.filter(r =>
    r.Metric === metric && matchesProblemType(r.Dataset, problemType),
  );
  const datasets = filterDatasetsByProblem(unique(metricRows, "Dataset"), problemType);
  const datasetDefault = dataset && (dataset === "All datasets" || datasets.includes(dataset))
    ? dataset
    : "All datasets";
  fillSelect("filter-fidelity-dataset", ["All datasets", ...datasets], datasetDefault);

  const selectedDataset = document.getElementById("filter-fidelity-dataset")?.value;
  const allDsSelected = isAllDatasets(selectedDataset);
  const rows = metricRows.filter(r => allDsSelected || r.Dataset === selectedDataset);

  if (!rows.length) {
    plot("fidelity-bar", [], { title: `No data for ${label}`, height: 220 });
    plot("fidelity-heatmap", [], { title: "No heatmap data", height: 220 });
    const titleEl = document.getElementById("fidelity-bar-title");
    if (titleEl) titleEl.textContent = label;
    return;
  }

  const byGen = {};
  rows.forEach(r => {
    const value = fidelityScoreFor(r);
    if (!r.Generator || value == null) return;
    if (!byGen[r.Generator]) byGen[r.Generator] = [];
    byGen[r.Generator].push(value);
  });

  const gens = GENERATORS;
  const barValues = valuesForAllGenerators(byGen);
  const direction = higherBetter ? "higher = better fidelity" : "lower = better fidelity";
  const missingCount = barValues.filter(v => v == null).length;
  const missingNote = missingCount ? ` · ${missingCount} generator(s) not evaluated` : "";
  const problemNote = problemType !== "all"
    ? ` · ${problemTypeLabel(problemType)} datasets`
    : "";

  const titleEl = document.getElementById("fidelity-bar-title");
  if (titleEl) titleEl.textContent = `${label} by generator`;

  const naTrace = missingBarLabelTrace(barValues, asPercent);
  plot("fidelity-bar", [
    privacyBarTrace({ x: gens, y: barValues, color: "#3b82f6", asPercent }),
    ...(naTrace ? [naTrace] : []),
  ], {
    title: `${label}${allDsSelected ? " (mean over datasets)" : ""}${problemNote} · ${direction}${missingNote}`,
    height: 460,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: numericYAxis(barValues, { asPercent }),
    xaxis: generatorXAxis(),
    showlegend: false,
  });

  const heatRows = metricRows;
  const heatDs = datasets;
  const heatGens = GENERATORS;
  const z = heatGens.map(g => heatDs.map(d => {
    const matches = heatRows.filter(r => r.Generator === g && r.Dataset === d);
    return mean(matches.map(fidelityScoreFor));
  }));

  const zFlat = z.flat().filter(v => v != null);
  const heatmapScale = asPercent
    ? { zmin: 0, zmax: 1 }
    : { zmin: zFlat.length ? Math.min(...zFlat) : undefined, zmax: zFlat.length ? Math.max(...zFlat) : undefined };

  plot("fidelity-heatmap", [{
    type: "heatmap",
    x: heatDs.map(d => d.replace(/^\d+\.\s*/, "")),
    y: heatGens,
    z,
    ...heatmapScale,
    colorscale: higherBetter ? "Viridis" : "YlOrRd",
    reversescale: !higherBetter,
    colorbar: {
      title: label,
      tickformat: asPercent ? ".0%" : ".2f",
    },
  }], {
    title: `${label} · dataset × generator${problemType !== "all" ? ` (${problemTypeLabel(problemType)})` : ""}`,
    height: 420,
    yaxis: generatorYAxis(),
  });
}

function setupFidelityFilters() {
  setupProblemTypeSelect("filter-fidelity-problem", renderFidelity);

  const catalog = DATA.fidelityMetrics || [];
  const available = new Set(catalog.map(m => m.id));
  const ordered = FIDELITY_PREFERRED_ORDER.filter(id => available.has(id));
  catalog.forEach(m => {
    if (!ordered.includes(m.id)) ordered.push(m.id);
  });

  const defaultMetric = ordered.includes("Quality_Score")
    ? "Quality_Score"
    : (ordered.includes("KS_Complement") ? "KS_Complement" : ordered[0]);

  const metricSel = document.getElementById("filter-fidelity-metric");
  if (metricSel) {
    metricSel.innerHTML = ordered
      .map(id => `<option value="${id}">${fidelityMetricLabel(id)}</option>`)
      .join("");
    metricSel.value = defaultMetric;
  }

  ["filter-fidelity-metric", "filter-fidelity-dataset"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderFidelity);
  });
}

function renderPrivacy() {
  const metric = document.getElementById("filter-privacy-metric")?.value;
  const problemType = document.getElementById("filter-privacy-problem")?.value || "all";
  const dataset = document.getElementById("filter-privacy-dataset")?.value;
  const catalog = DATA.privacyMetrics || [];
  const catalogEntry = catalog.find(m => m.id === metric) || {};
  const label = privacyMetricLabel(metric);
  const asPercent = privacyUsesPercent(metric, catalogEntry);
  const lowerBetter = catalogEntry.lower_is_better !== false;

  const metricRows = DATA.privacy.filter(r =>
    r.Metric === metric && matchesProblemType(r.Dataset, problemType),
  );
  const datasets = filterDatasetsByProblem(unique(metricRows, "Dataset"), problemType);
  const datasetDefault = dataset && (dataset === "All datasets" || datasets.includes(dataset))
    ? dataset
    : "All datasets";
  fillSelect("filter-privacy-dataset", ["All datasets", ...datasets], datasetDefault);

  const selectedDataset = document.getElementById("filter-privacy-dataset")?.value;
  const allDsSelected = isAllDatasets(selectedDataset);
  const rows = metricRows.filter(r => allDsSelected || r.Dataset === selectedDataset);

  if (!rows.length) {
    plot("privacy-bar", [], { title: `No data for ${label}`, height: 220 });
    plot("privacy-heatmap", [], { title: "No heatmap data", height: 220 });
    const titleEl = document.getElementById("privacy-bar-title");
    if (titleEl) titleEl.textContent = label;
    return;
  }

  const byGen = {};
  rows.forEach(r => {
    const value = toNum(r.Mean ?? r.MetricValue ?? r.Value);
    if (!r.Generator || value == null) return;
    if (!byGen[r.Generator]) byGen[r.Generator] = [];
    byGen[r.Generator].push(value);
  });

  const gens = GENERATORS;
  const barValues = valuesForAllGenerators(byGen);
  const direction = catalogEntry.is_sample_size
    ? "Hungarian matching subsample size (not a privacy risk score)"
    : lowerBetter
      ? "lower = more private"
      : "lower similarity = more private";
  const missingCount = barValues.filter(v => v == null).length;
  const missingNote = missingCount ? ` · ${missingCount} generator(s) not evaluated` : "";
  const problemNote = problemType !== "all"
    ? ` · ${problemTypeLabel(problemType)} datasets`
    : "";

  const titleEl = document.getElementById("privacy-bar-title");
  if (titleEl) titleEl.textContent = `${label} by generator`;

  const naTrace = missingBarLabelTrace(barValues, asPercent);
  plot("privacy-bar", [
    privacyBarTrace({ x: gens, y: barValues, color: "#f59e0b", asPercent }),
    ...(naTrace ? [naTrace] : []),
  ], {
    title: `${label}${allDsSelected ? " (mean over datasets)" : ""}${problemNote} · ${direction}${missingNote}`,
    height: 460,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: numericYAxis(barValues, { asPercent }),
    xaxis: generatorXAxis(),
    showlegend: false,
  });

  const heatRows = metricRows;
  const heatDs = datasets;
  const heatGens = GENERATORS;
  const z = heatGens.map(g => heatDs.map(d => {
    const matches = heatRows.filter(r => r.Generator === g && r.Dataset === d);
    return mean(matches.map(r => toNum(r.Mean ?? r.MetricValue ?? r.Value)));
  }));

  plot("privacy-heatmap", [{
    type: "heatmap",
    x: heatDs.map(d => d.replace(/^\d+\.\s*/, "")),
    y: heatGens,
    z,
    colorscale: lowerBetter ? "YlOrRd" : "RdYlGn",
    reversescale: !lowerBetter,
    colorbar: {
      title: label,
      tickformat: asPercent ? ".0%" : ".2f",
    },
  }], {
    title: `${label} · dataset × generator${problemType !== "all" ? ` (${problemTypeLabel(problemType)})` : ""}`,
    height: 420,
    yaxis: generatorYAxis(),
  });
}

function setupPrivacyFilters() {
  setupProblemTypeSelect("filter-privacy-problem", renderPrivacy);

  const catalog = DATA.privacyMetrics || [];
  const available = new Set(catalog.map(m => m.id));
  const ordered = PRIVACY_PREFERRED_ORDER.filter(id => available.has(id));
  catalog.forEach(m => {
    if (!ordered.includes(m.id)) ordered.push(m.id);
  });

  const defaultMetric = ordered.includes("Mahalanobis_Distance")
    ? "Mahalanobis_Distance"
    : (ordered.includes("NNDR") ? "NNDR" : ordered[0]);

  const metricSel = document.getElementById("filter-privacy-metric");
  if (metricSel) {
    metricSel.innerHTML = ordered
      .map(id => `<option value="${id}">${privacyMetricLabel(id)}</option>`)
      .join("");
    metricSel.value = defaultMetric;
  }

  ["filter-privacy-metric", "filter-privacy-dataset"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderPrivacy);
  });
}

function tradeoffUtilityMetrics(problemType = "all") {
  const rows = DATA.utilityAgg || [];
  const preferredClf = ["Accuracy", "F1", "Precision", "Recall"];
  const preferredReg = ["R2", "RMSE", "MAE"];
  const preferred =
    problemType === "regression" ? preferredReg.concat(preferredClf)
    : problemType === "classification" ? preferredClf.concat(preferredReg)
    : ["Accuracy", "F1", "Precision", "Recall", "R2", "RMSE", "MAE"];
  const available = new Set(
    rows
      .filter(r => {
        if (r.EvaluationType !== "TSTR" || !r.Metric) return false;
        if (String(r.Metric).includes("Drop") || String(r.Metric).includes("Gap") || String(r.Metric).includes("Increase")) return false;
        if (!matchesProblemType(r.Dataset, problemType)) return false;
        return true;
      })
      .map(r => r.Metric),
  );
  return preferred.filter(m => available.has(m)).concat(
    [...available].filter(m => !preferred.includes(m)).sort(),
  );
}

function tradeoffFidelityMetrics() {
  const catalog = DATA.fidelityMetrics || [];
  const fromData = new Set((DATA.fidelity || []).map(r => r.Metric).filter(Boolean));
  const ordered = FIDELITY_PREFERRED_ORDER.filter(id => fromData.has(id));
  const rest = [...fromData].filter(id => !ordered.includes(id)).sort();
  const ids = ordered.concat(rest);
  if (catalog.length) {
    return ids.map(id => {
      const c = catalog.find(x => x.id === id);
      return { id, label: c?.label || fidelityLabel(id), higher_is_better: c?.higher_is_better !== false };
    });
  }
  return ids.map(id => ({ id, label: fidelityLabel(id), higher_is_better: true }));
}

function tradeoffPrivacyMetrics() {
  const catalog = DATA.privacyMetrics || [];
  const fromData = new Set((DATA.privacy || []).map(r => r.Metric).filter(Boolean));
  const preferred = ["NNDR", "MIA_AUC", "Mahalanobis_Distance", "Mean_Distance", "Hungarian_Cosine_Similarity"];
  const ordered = preferred.filter(id => fromData.has(id));
  const rest = [...fromData].filter(id => !ordered.includes(id) && id !== "Num_Matches").sort();
  const ids = ordered.concat(rest);
  return ids.map(id => {
    const c = catalog.find(x => x.id === id);
    let higherIsPrivate = true;
    if (id === "MIA_AUC" || id.includes("Similarity")) higherIsPrivate = false;
    if (id === "NNDR" || id.includes("Distance")) higherIsPrivate = true;
    return {
      id,
      label: c?.label || privacyLabel(id),
      higher_is_private: higherIsPrivate,
    };
  });
}

/** Ordered list of datasets currently in the Compare pane. */
let tradeoffSelectedDatasets = [];
let tradeoffPickerBound = false;
let tradeoffDragDataset = null;

function tradeoffEligibleDatasets(utilMetric, fidMetric, privMetric, problemType = "all") {
  const utilDs = new Set(
    (DATA.utilityAgg || [])
      .filter(r => r.EvaluationType === "TSTR" && r.Metric === utilMetric && toNum(r.Mean) != null)
      .map(r => r.Dataset),
  );
  const fidDs = new Set(
    (DATA.fidelity || [])
      .filter(r => r.Metric === fidMetric && toNum(r.Mean) != null)
      .map(r => r.Dataset),
  );
  const privDs = new Set(
    (DATA.privacy || [])
      .filter(r => r.Metric === privMetric && toNum(r.Mean) != null)
      .map(r => r.Dataset),
  );
  return filterDatasetsByProblem(
    uniqueDatasets(
      [...utilDs].filter(d => fidDs.has(d) && privDs.has(d)).map(d => ({ Dataset: d })),
    ),
    problemType,
  );
}

function setupTradeoffFilters() {
  setupProblemTypeSelect("filter-tradeoff-problem", () => {
    refreshTradeoffUtilityOptions();
    refreshTradeoffDatasetPicker({ ensureMin: true });
    renderTradeoff();
  }, "classification");

  refreshTradeoffUtilityOptions();

  const fidMetrics = tradeoffFidelityMetrics();
  const privMetrics = tradeoffPrivacyMetrics();
  const fidIds = fidMetrics.map(m => m.id);
  fillSelect(
    "filter-tradeoff-fidelity",
    fidIds,
    fidIds.includes("Quality_Score") ? "Quality_Score" : fidIds[0],
  );
  const fidSel = document.getElementById("filter-tradeoff-fidelity");
  if (fidSel) {
    [...fidSel.options].forEach(opt => {
      const m = fidMetrics.find(x => x.id === opt.value);
      if (m) opt.textContent = m.label;
    });
  }
  const privIds = privMetrics.map(m => m.id);
  fillSelect(
    "filter-tradeoff-privacy",
    privIds,
    privIds.includes("NNDR") ? "NNDR" : (privIds.includes("MIA_AUC") ? "MIA_AUC" : privIds[0]),
  );
  const privSel = document.getElementById("filter-tradeoff-privacy");
  if (privSel) {
    [...privSel.options].forEach(opt => {
      const m = privMetrics.find(x => x.id === opt.value);
      if (m) opt.textContent = m.label;
    });
  }

  bindTradeoffDatasetPicker();
  refreshTradeoffDatasetPicker({ ensureMin: true });
}

function refreshTradeoffUtilityOptions() {
  const problemType = document.getElementById("filter-tradeoff-problem")?.value || "all";
  const utilMetrics = tradeoffUtilityMetrics(problemType);
  const prev = document.getElementById("filter-tradeoff-utility")?.value;
  const preferred =
    problemType === "regression"
      ? (utilMetrics.includes("R2") ? "R2" : utilMetrics[0])
      : (utilMetrics.includes("Accuracy") ? "Accuracy" : utilMetrics[0]);
  fillSelect(
    "filter-tradeoff-utility",
    utilMetrics,
    prev && utilMetrics.includes(prev) ? prev : preferred,
  );
}

function currentTradeoffEligible() {
  const util = document.getElementById("filter-tradeoff-utility")?.value;
  const fid = document.getElementById("filter-tradeoff-fidelity")?.value;
  const priv = document.getElementById("filter-tradeoff-privacy")?.value;
  const problemType = document.getElementById("filter-tradeoff-problem")?.value || "all";
  if (!util || !fid || !priv) return [];
  return tradeoffEligibleDatasets(util, fid, priv, problemType);
}

function refreshTradeoffDatasetPicker({ ensureMin = false } = {}) {
  const eligible = currentTradeoffEligible();
  const eligibleSet = new Set(eligible);
  tradeoffSelectedDatasets = tradeoffSelectedDatasets.filter(d => eligibleSet.has(d));

  if (ensureMin && tradeoffSelectedDatasets.length < 2) {
    for (const d of eligible) {
      if (tradeoffSelectedDatasets.length >= 2) break;
      if (!tradeoffSelectedDatasets.includes(d)) tradeoffSelectedDatasets.push(d);
    }
  }

  renderTradeoffDatasetPicker(eligible);
}

function renderTradeoffDatasetPicker(eligible) {
  const availEl = document.getElementById("tradeoff-available");
  const selEl = document.getElementById("tradeoff-selected");
  if (!availEl || !selEl) return;

  const selectedSet = new Set(tradeoffSelectedDatasets);
  const available = eligible.filter(d => !selectedSet.has(d));
  const selected = tradeoffSelectedDatasets.filter(d => eligible.includes(d));

  const chipHtml = (ds, list) => `
    <button type="button" class="ds-chip" draggable="true"
      data-dataset="${escapeAttr(ds)}" data-list="${list}"
      title="${escapeAttr(ds)}">${escapeHtml(shortDatasetLabel(ds))}</button>
  `;

  availEl.innerHTML = available.length
    ? available.map(d => chipHtml(d, "available")).join("")
    : `<div class="ds-chip-empty">No more datasets for this problem type / metrics</div>`;
  selEl.innerHTML = selected.length
    ? selected.map(d => chipHtml(d, "selected")).join("")
    : `<div class="ds-chip-empty">Drag 2+ datasets here to compare</div>`;

  const availCount = document.getElementById("tradeoff-avail-count");
  const selCount = document.getElementById("tradeoff-selected-count");
  if (availCount) availCount.textContent = String(available.length);
  if (selCount) selCount.textContent = String(selected.length);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/'/g, "&#39;");
}

function addTradeoffDataset(ds, index = null) {
  if (!ds || tradeoffSelectedDatasets.includes(ds)) return;
  if (index == null || index < 0 || index > tradeoffSelectedDatasets.length) {
    tradeoffSelectedDatasets.push(ds);
  } else {
    tradeoffSelectedDatasets.splice(index, 0, ds);
  }
}

function removeTradeoffDataset(ds) {
  tradeoffSelectedDatasets = tradeoffSelectedDatasets.filter(d => d !== ds);
}

function moveTradeoffDataset(ds, toList, insertBeforeDs = null) {
  if (!ds) return;
  if (toList === "selected") {
    const fromIdx = tradeoffSelectedDatasets.indexOf(ds);
    if (fromIdx >= 0) tradeoffSelectedDatasets.splice(fromIdx, 1);
    let insertAt = tradeoffSelectedDatasets.length;
    if (insertBeforeDs) {
      const i = tradeoffSelectedDatasets.indexOf(insertBeforeDs);
      if (i >= 0) insertAt = i;
    }
    addTradeoffDataset(ds, insertAt);
  } else {
    removeTradeoffDataset(ds);
  }
}

function bindTradeoffDatasetPicker() {
  if (tradeoffPickerBound) return;
  const availEl = document.getElementById("tradeoff-available");
  const selEl = document.getElementById("tradeoff-selected");
  if (!availEl || !selEl) return;
  tradeoffPickerBound = true;

  const onDragStart = (e) => {
    const chip = e.target.closest(".ds-chip");
    if (!chip) return;
    tradeoffDragDataset = chip.dataset.dataset;
    chip.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", tradeoffDragDataset);
  };
  const onDragEnd = (e) => {
    const chip = e.target.closest(".ds-chip");
    if (chip) chip.classList.remove("dragging");
    tradeoffDragDataset = null;
    availEl.classList.remove("drag-over");
    selEl.classList.remove("drag-over");
  };
  const onDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    const list = e.currentTarget;
    list.classList.add("drag-over");
  };
  const onDragLeave = (e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) {
      e.currentTarget.classList.remove("drag-over");
    }
  };
  const onDrop = (e) => {
    e.preventDefault();
    const listEl = e.currentTarget;
    listEl.classList.remove("drag-over");
    const ds = tradeoffDragDataset || e.dataTransfer.getData("text/plain");
    if (!ds) return;
    const toList = listEl.dataset.list;
    const overChip = e.target.closest(".ds-chip");
    const insertBefore = overChip && overChip.dataset.list === "selected"
      ? overChip.dataset.dataset
      : null;
    moveTradeoffDataset(ds, toList, insertBefore === ds ? null : insertBefore);
    refreshTradeoffDatasetPicker();
    renderTradeoff({ skipPickerRefresh: true });
  };
  const onClick = (e) => {
    const chip = e.target.closest(".ds-chip");
    if (!chip) return;
    const ds = chip.dataset.dataset;
    const list = chip.dataset.list;
    if (list === "available") addTradeoffDataset(ds);
    else removeTradeoffDataset(ds);
    refreshTradeoffDatasetPicker();
    renderTradeoff({ skipPickerRefresh: true });
  };

  [availEl, selEl].forEach(el => {
    el.addEventListener("dragstart", onDragStart);
    el.addEventListener("dragend", onDragEnd);
    el.addEventListener("dragover", onDragOver);
    el.addEventListener("dragleave", onDragLeave);
    el.addEventListener("drop", onDrop);
    el.addEventListener("click", onClick);
  });
}

function buildTradeoffGeneratorMeans(utilMetric, fidMetric, privMetric, selectedDatasets) {
  const dsSet = new Set(selectedDatasets);
  const rows = [];
  for (const gen of GENERATORS) {
    const utilVals = (DATA.utilityAgg || [])
      .filter(r =>
        r.Generator === gen
        && r.EvaluationType === "TSTR"
        && r.Metric === utilMetric
        && dsSet.has(r.Dataset),
      )
      .map(r => toNum(r.Mean))
      .filter(v => v != null);
    const fidVals = (DATA.fidelity || [])
      .filter(r => r.Generator === gen && r.Metric === fidMetric && dsSet.has(r.Dataset))
      .map(r => toNum(r.Mean))
      .filter(v => v != null);
    const privVals = (DATA.privacy || [])
      .filter(r => r.Generator === gen && r.Metric === privMetric && dsSet.has(r.Dataset))
      .map(r => toNum(r.Mean))
      .filter(v => v != null);
    if (!utilVals.length && !fidVals.length && !privVals.length) continue;
    rows.push({
      Generator: gen,
      Utility: mean(utilVals),
      Fidelity: mean(fidVals),
      Privacy: mean(privVals),
      NDatasets: Math.max(utilVals.length, fidVals.length, privVals.length),
    });
  }
  rows.forEach((r, i) => { r.PointId = i + 1; });
  return rows;
}

function renderTradeoff({ skipPickerRefresh = false } = {}) {
  if (!document.getElementById("filter-tradeoff-utility")?.options.length) {
    setupTradeoffFilters();
  }

  const utilMetric = document.getElementById("filter-tradeoff-utility")?.value;
  const fidMetric = document.getElementById("filter-tradeoff-fidelity")?.value;
  const privMetric = document.getElementById("filter-tradeoff-privacy")?.value;
  const problemType = document.getElementById("filter-tradeoff-problem")?.value || "all";
  if (!utilMetric || !fidMetric || !privMetric) {
    renderTradeoffLegacy();
    return;
  }

  if (!skipPickerRefresh) refreshTradeoffDatasetPicker();

  const eligible = currentTradeoffEligible();
  const selected = tradeoffSelectedDatasets.filter(d => eligible.includes(d));
  const rows = buildTradeoffGeneratorMeans(utilMetric, fidMetric, privMetric, selected);

  const fidMeta = tradeoffFidelityMetrics().find(m => m.id === fidMetric);
  const privMeta = tradeoffPrivacyMetrics().find(m => m.id === privMetric);
  const privNote = privMeta?.higher_is_private ? "higher is more private" : "lower is more private";
  const fidNote = fidMeta?.higher_is_better === false ? "lower is better fidelity" : "higher is better fidelity";
  const problemNote = problemType !== "all" ? ` · ${problemTypeLabel(problemType)}` : "";

  const sub = document.getElementById("tradeoff-subtitle");
  if (sub) {
    sub.textContent = `Trade-off across ${selected.length} dataset(s)${problemNote} · TSTR ${utilMetric} · ${fidMeta?.label || fidMetric} · ${privMeta?.label || privMetric} (${privNote})`;
  }
  const listEl = document.getElementById("tradeoff-dataset-list");
  if (listEl) {
    if (!selected.length) {
      listEl.textContent = "Select at least 2 datasets in Compare (drag or click).";
    } else if (selected.length === 1) {
      listEl.textContent = `Only 1 dataset selected (${shortDatasetLabel(selected[0])}) — add more to compare.`;
    } else {
      listEl.textContent = `Datasets: ${selected.map(shortDatasetLabel).join(" · ")}`;
    }
  }

  const setTitle = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };
  setTitle("tradeoff-fidelity-utility-title", `Fidelity vs Utility (${utilMetric})`);
  setTitle("tradeoff-privacy-utility-title", `Privacy vs Utility (${utilMetric})`);
  setTitle("tradeoff-fidelity-privacy-title", `Fidelity vs Privacy`);

  if (selected.length < 2 || !rows.length) {
    ["tradeoff-fidelity-utility", "tradeoff-privacy-utility", "tradeoff-fidelity-privacy"].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.innerHTML = `<p style="padding:2rem;color:#9aa8bc">${
          selected.length < 2
            ? "Drag or click 2+ datasets into Compare to plot generator means."
            : "No generator means for this selection."
        }</p>`;
      }
    });
    const tbody = document.querySelector("#tradeoff-values-table tbody");
    if (tbody) tbody.innerHTML = "";
    return;
  }

  plotNumberedTradeoff(
    "tradeoff-fidelity-utility",
    rows,
    "Fidelity",
    "Utility",
    `Fidelity (${fidMeta?.label || fidMetric}; ${fidNote})`,
    `Mean TSTR ${utilMetric}`,
  );
  plotNumberedTradeoff(
    "tradeoff-privacy-utility",
    rows,
    "Privacy",
    "Utility",
    `Privacy (${privMeta?.label || privMetric}; ${privNote})`,
    `Mean TSTR ${utilMetric}`,
  );
  plotNumberedTradeoff(
    "tradeoff-fidelity-privacy",
    rows,
    "Privacy",
    "Fidelity",
    `Privacy (${privMeta?.label || privMetric}; ${privNote})`,
    `Fidelity (${fidMeta?.label || fidMetric})`,
  );

  const tbody = document.querySelector("#tradeoff-values-table tbody");
  if (tbody) {
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td style="color:${GENERATOR_COLORS[r.Generator] || "#ccc"};font-weight:700">${r.PointId}</td>
        <td style="color:${GENERATOR_COLORS[r.Generator] || "#ccc"}">${r.Generator}</td>
        <td>${fmtTrade(r.Utility)}</td>
        <td>${fmtTrade(r.Fidelity)}</td>
        <td>${fmtTrade(r.Privacy)}</td>
        <td>${r.NDatasets}</td>
      </tr>
    `).join("");
  }
}

function fmtTrade(v) {
  const n = toNum(v);
  return n == null ? "—" : n.toFixed(3);
}

function plotNumberedTradeoff(plotId, rows, xKey, yKey, xlabel, ylabel) {
  const xs = rows.map(r => toNum(r[xKey]));
  const ys = rows.map(r => toNum(r[yKey]));
  const colors = rows.map(r => GENERATOR_COLORS[r.Generator] || "#888");
  const ids = rows.map(r => String(r.PointId));

  const paired = xs.map((x, i) => ({ x, y: ys[i] })).filter(p => p.x != null && p.y != null);
  let fitTrace = null;
  if (paired.length >= 3) {
    const n = paired.length;
    const mx = paired.reduce((s, p) => s + p.x, 0) / n;
    const my = paired.reduce((s, p) => s + p.y, 0) / n;
    let num = 0;
    let den = 0;
    paired.forEach(p => {
      num += (p.x - mx) * (p.y - my);
      den += (p.x - mx) ** 2;
    });
    if (den > 1e-12) {
      const slope = num / den;
      const intercept = my - slope * mx;
      const xMin = Math.min(...paired.map(p => p.x));
      const xMax = Math.max(...paired.map(p => p.x));
      const pad = (xMax - xMin) * 0.05 || 0.02;
      const x0 = xMin - pad;
      const x1 = xMax + pad;
      fitTrace = {
        type: "scatter",
        mode: "lines",
        x: [x0, x1],
        y: [intercept + slope * x0, intercept + slope * x1],
        name: "OLS fit",
        line: { color: "rgba(238,242,247,0.7)", width: 2 },
        hoverinfo: "skip",
      };
    }
  }

  const markers = {
    type: "scatter",
    mode: "markers+text",
    x: xs,
    y: ys,
    text: ids,
    textposition: "middle center",
    textfont: {
      family: DASH_FONT,
      size: 12,
      color: colors.map(c => contrastText(c)),
    },
    marker: {
      size: 22,
      color: colors,
      line: { width: 1.2, color: "#111" },
    },
    customdata: rows.map(r => r.Generator),
    hovertemplate: "#%{text} %{customdata}<br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
    name: "Generators",
    showlegend: false,
  };

  const legendTraces = GENERATORS.filter(g => rows.some(r => r.Generator === g)).map((g, i) => ({
    type: "scatter",
    mode: "markers",
    x: [null],
    y: [null],
    name: `${i + 1}. ${g}`,
    marker: { size: 10, color: GENERATOR_COLORS[g], line: { width: 1, color: "#111" } },
  }));

  const traces = [markers, ...legendTraces];
  if (fitTrace) traces.unshift(fitTrace);

  plot(plotId, traces, {
    height: 440,
    margin: { l: 64, r: 24, t: 20, b: 88 },
    xaxis: { title: xlabel, automargin: true, zeroline: false },
    yaxis: { title: ylabel, automargin: true, zeroline: false },
    showlegend: true,
    legend: { orientation: "h", y: -0.22, x: 0.5, xanchor: "center" },
  });
}

function contrastText(hex) {
  try {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return lum > 0.55 ? "#111111" : "#FFFFFF";
  } catch {
    return "#FFFFFF";
  }
}

function renderTradeoffLegacy() {
  const t = DATA.tradeoff;
  if (!t.length) return;
  const el = document.getElementById("tradeoff-fidelity-utility");
  if (!el) return;
  const rows = t.map((r, i) => ({
    PointId: i + 1,
    Generator: r.Generator,
    Utility: toNum(r.Utility),
    Fidelity: toNum(r.Fidelity),
    Privacy: toNum(r.Privacy),
  }));
  plotNumberedTradeoff(el.id, rows, "Fidelity", "Utility", "Fidelity", "Utility");
}

function bindTradeoffFilters() {
  ["filter-tradeoff-utility", "filter-tradeoff-fidelity", "filter-tradeoff-privacy"].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      refreshTradeoffDatasetPicker({ ensureMin: true });
      renderTradeoff({ skipPickerRefresh: true });
    });
  });
}

function renderRankings() {
  const w = DATA.weighted;
  const b = DATA.borda;

  if (w.length) {
    const scoreCol = w[0].OverallScore !== undefined ? "OverallScore" : "WeightedScore";
    const values = w.map(r => toNum(r[scoreCol]));
    plot("rankings-weighted", [
      barChartTrace({ x: w.map(r => r.Generator), y: values, name: "Score", color: "#8b5cf6", asPercent: true }),
    ], {
      title: "Weighted overall ranking (40% utility, 30% privacy, 30% fidelity)",
      height: 420,
      margin: { t: 44, b: 88, l: 56, r: 24 },
      yaxis: percentYAxis(values),
      showlegend: false,
    });
  }

  if (b.length) {
    const values = b.map(r => toNum(r.BordaScore ?? r.wins));
    const maxScore = Math.max(...values.filter(v => v != null), 1);
    plot("rankings-borda", [
      barChartTrace({ x: b.map(r => r.generator || r.Generator), y: values, name: "Borda", color: "#06b6d4", asPercent: false, decimals: 0 }),
    ], {
      title: "Borda count ranking",
      height: 420,
      yaxis: {
        type: "linear",
        range: [0, maxScore * 1.15],
        tickformat: ".0f",
        automargin: true,
      },
      showlegend: false,
    });
  }
}

const STATS_METRIC_OPTIONS = [
  { value: "Mean_Error_Pct", label: "Mean Error %" },
  { value: "Median_Error_Pct", label: "Median Error %" },
  { value: "Std_Error_Pct", label: "Std Error %" },
];

function pcaErrorRows() {
  return DATA.statistics?.pca_errors || [];
}

function statsMetricField() {
  return document.getElementById("filter-stats-metric")?.value || "Mean_Error_Pct";
}

function statsMetricLabel(field) {
  return STATS_METRIC_OPTIONS.find(o => o.value === field)?.label || field;
}

function renderPcaErrorTable(rows) {
  const tbody = document.querySelector("#stats-error-table tbody");
  if (!tbody) return;
  const ordered = GENERATORS
    .map(g => rows.find(r => r.Generator === g))
    .filter(Boolean);
  if (!ordered.length) {
    tbody.innerHTML = `<tr><td colspan="5">No average-error statistics for this selection.</td></tr>`;
    return;
  }
  tbody.innerHTML = ordered.map(r => `
    <tr>
      <td>${r.Generator}</td>
      <td>${toNum(r.Mean_Error_Pct)?.toFixed(6) ?? "—"}</td>
      <td>${toNum(r.Median_Error_Pct)?.toFixed(6) ?? "—"}</td>
      <td>${toNum(r.Std_Error_Pct)?.toFixed(6) ?? "—"}</td>
      <td>${r.Source_Group || "—"}</td>
    </tr>
  `).join("");
}

function renderPcaErrorCharts() {
  const rows = pcaErrorRows();
  const field = statsMetricField();
  const datasetSel = document.getElementById("filter-stats-dataset")?.value || "All datasets";

  const datasets = uniqueDatasets(rows);
  fillSelect("filter-stats-dataset", ["All datasets", ...datasets], datasetSel);
  fillSelect(
    "filter-stats-metric",
    STATS_METRIC_OPTIONS.map(o => o.value),
    field,
  );
  const metricEl = document.getElementById("filter-stats-metric");
  if (metricEl) {
    [...metricEl.options].forEach(opt => {
      opt.textContent = statsMetricLabel(opt.value);
    });
  }

  const selectedDataset = document.getElementById("filter-stats-dataset")?.value || "All datasets";
  const metricField = statsMetricField();
  const metricLabel = statsMetricLabel(metricField);

  const barRows = selectedDataset === "All datasets"
    ? rows
    : rows.filter(r => r.Dataset === selectedDataset);

  const barTitle = document.getElementById("stats-bar-title");
  if (barTitle) {
    barTitle.textContent = selectedDataset === "All datasets"
      ? `Average ${metricLabel} across datasets`
      : `Average error by model — ${shortDatasetLabel(selectedDataset)}`;
  }

  const barY = GENERATORS.map(g => {
    const subset = barRows.filter(r => r.Generator === g);
    if (!subset.length) return null;
    return mean(subset.map(r => toNum(r[metricField])).filter(v => v != null));
  });

  // Mean_Error_Pct values are already on a percent scale (e.g. 10.5 = 10.5%).
  plot("stats-error-bar", [
    barChartTrace({
      x: GENERATORS,
      y: barY,
      name: metricLabel,
      color: "#3b82f6",
      asPercent: false,
      decimals: 2,
    }),
  ], {
    height: 420,
    showlegend: false,
    yaxis: { title: metricLabel, ticksuffix: "%", automargin: true, rangemode: "tozero" },
    xaxis: { tickangle: -25, automargin: true },
  });

  const heatDatasets = uniqueDatasets(rows);
  const z = heatDatasets.map(ds =>
    GENERATORS.map(g => {
      const hit = rows.find(r => r.Dataset === ds && r.Generator === g);
      return hit ? toNum(hit[metricField]) : null;
    }),
  );
  const flat = z.flat().filter(v => v != null);
  const zmax = flat.length ? Math.max(...flat) : 1;

  const heatTitle = document.getElementById("stats-heat-title");
  if (heatTitle) heatTitle.textContent = `${metricLabel} heatmap`;

  plot("stats-error-heatmap", [{
    type: "heatmap",
    x: GENERATORS,
    y: heatDatasets.map(shortDatasetLabel),
    z,
    zmin: 0,
    zmax,
    colorscale: "YlOrRd",
    colorbar: { title: "%" },
    hovertemplate: "%{y} · %{x}<br>%{z:.2f}%<extra></extra>",
  }], {
    height: 480,
    xaxis: { tickangle: -25, automargin: true, type: "category", categoryarray: GENERATORS },
    yaxis: {
      type: "category",
      categoryarray: [...heatDatasets.map(shortDatasetLabel)].reverse(),
      automargin: true,
    },
    margin: { l: 110, r: 40, t: 20, b: 80 },
  });

  const tableRows = selectedDataset === "All datasets"
    ? GENERATORS.map(g => {
        const subset = rows.filter(r => r.Generator === g);
        if (!subset.length) return null;
        return {
          Generator: g,
          Mean_Error_Pct: mean(subset.map(r => toNum(r.Mean_Error_Pct)).filter(v => v != null)),
          Median_Error_Pct: mean(subset.map(r => toNum(r.Median_Error_Pct)).filter(v => v != null)),
          Std_Error_Pct: mean(subset.map(r => toNum(r.Std_Error_Pct)).filter(v => v != null)),
          Source_Group: "mean across datasets",
        };
      }).filter(Boolean)
    : rows.filter(r => r.Dataset === selectedDataset);

  const tableTitle = document.querySelector("#panel-statistics .table-wrap .section-title");
  if (tableTitle) {
    tableTitle.textContent = selectedDataset === "All datasets"
      ? "Average error by model (mean across 15 datasets)"
      : `Average error by model — ${shortDatasetLabel(selectedDataset)}`;
  }
  renderPcaErrorTable(tableRows);
}

function renderWilcoxonHeatmaps() {
  const w = DATA.statistics?.wilcoxon || [];
  if (!w.length) return;

  const metrics = [...new Set(w.map(r => r.Metric))].slice(0, 4);
  metrics.forEach((metric, i) => {
    const sub = w.filter(r => r.Metric === metric);
    const gens = [...new Set(sub.flatMap(r => [r.Generator_A, r.Generator_B]))];
    const mat = gens.map(g1 => gens.map(g2 => {
      if (g1 === g2) return 0;
      const r = sub.find(x =>
        (x.Generator_A === g1 && x.Generator_B === g2)
        || (x.Generator_A === g2 && x.Generator_B === g1),
      );
      return r ? -Math.log10(toNum(r.p_bh ?? r.p_raw) || 1) : null;
    }));
    const flat = mat.flat().filter(v => v != null && v > 0);
    const zmax = flat.length ? Math.max(...flat) : 5;
    plot(`stats-heat-${i}`, [{
      type: "heatmap",
      x: gens,
      y: gens,
      z: mat,
      zmin: 0,
      zmax,
      colorscale: "Reds",
      colorbar: { title: "-log10(p)" },
    }], { title: `Significance: ${metric}`, height: 380 });
  });
}

function renderStatistics() {
  renderPcaErrorCharts();
  renderWilcoxonHeatmaps();
}

function setupStatisticsFilters() {
  ["filter-stats-dataset", "filter-stats-metric"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderPcaErrorCharts);
  });
  const datasets = uniqueDatasets(pcaErrorRows());
  fillSelect("filter-stats-dataset", ["All datasets", ...datasets], datasets[0] || "All datasets");
  fillSelect(
    "filter-stats-metric",
    STATS_METRIC_OPTIONS.map(o => o.value),
    "Mean_Error_Pct",
  );
  const metricEl = document.getElementById("filter-stats-metric");
  if (metricEl) {
    [...metricEl.options].forEach(opt => {
      opt.textContent = statsMetricLabel(opt.value);
    });
  }
}

function setupTabs() {
  document.querySelectorAll("nav.tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("nav.tabs button").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.panel).classList.add("active");
      window.dispatchEvent(new Event("resize"));
    });
  });
}

function setupFilters() {
  const taskEl = document.getElementById("filter-task");
  if (taskEl) {
    taskEl.addEventListener("change", () => {
      syncUtilityMetricOptions(taskEl.value);
      renderUtility();
    });
  }
  fillSelect("filter-utility-eval", ["TSTR", "TRTR"], "TSTR");
  ["filter-metric", "filter-utility-eval", "filter-dataset", "filter-generator"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderUtility);
  });

  fillSelect("filter-task", ["classification", "regression"], "classification");
  syncUtilityMetricOptions("classification", "Accuracy");
}

async function init() {
  const loading = document.getElementById("app-loading");
  const main = document.getElementById("app-main");
  try {
    loading.style.display = "block";
    loading.textContent = "Loading benchmark data…";
    await loadAllData();
    loading.style.display = "none";
    main.style.display = "block";

    renderMetrics();
    setupTabs();
    setupFilters();
    setupUtilityDataleakFilters();
    setupFidelityFilters();
    setupPrivacyFilters();
    setupStatisticsFilters();
    bindTradeoffFilters();
    renderOverview();
    renderUtility();
    renderUtilityDataleak();
    renderFidelity();
    renderPrivacy();
    renderTradeoff();
    renderRankings();
    renderStatistics();
  } catch (err) {
    console.error(err);
    loading.textContent = `Failed to load dashboard data: ${err.message}. Hard refresh (Ctrl+Shift+R) or rebuild with python dashboard/build_pages.py`;
    main.style.display = "none";
  }
}

init();
