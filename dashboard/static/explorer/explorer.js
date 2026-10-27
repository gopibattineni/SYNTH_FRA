const DATA_BASE = "../data";
const DATA_VERSION = "20260928c";
const GENERATORS = [
  "CTGAN",
  "CopulaGAN",
  "TVAE",
  "GaussianCopula",
  "WGAN_GP",
  "CTABGAN",
  "TabDDPM",
  "ForestDiffusion",
];

const METRIC_META = {
  Accuracy_Gap: { label: "Accuracy gap (TRTR−TSTR)", better: "lower", task: "classification" },
  F1_Gap: { label: "F1 gap", better: "lower", task: "classification" },
  Precision_Gap: { label: "Precision gap", better: "lower", task: "classification" },
  Recall_Gap: { label: "Recall gap", better: "lower", task: "classification" },
  ROC_AUC_Gap: { label: "ROC-AUC gap", better: "lower", task: "classification" },
  Accuracy: { label: "Accuracy", better: "higher", task: "classification" },
  F1: { label: "F1", better: "higher", task: "classification" },
  Precision: { label: "Precision", better: "higher", task: "classification" },
  Recall: { label: "Recall", better: "higher", task: "classification" },
  ROC_AUC: { label: "ROC-AUC", better: "higher", task: "classification" },
  R2_Gap: { label: "R² gap (TRTR−TSTR)", better: "lower", task: "regression" },
  RMSE_Increase: { label: "RMSE increase", better: "lower", task: "regression" },
  MAE_Increase: { label: "MAE increase", better: "lower", task: "regression" },
  R2: { label: "R²", better: "higher", task: "regression" },
  RMSE: { label: "RMSE", better: "lower", task: "regression" },
  MAE: { label: "MAE", better: "lower", task: "regression" },
  Quality_Score: { label: "SDV Quality Score", better: "higher" },
  KS_Complement: { label: "KS Complement", better: "higher" },
  MMD: { label: "MMD", better: "lower" },
  Wasserstein_Distance: { label: "Wasserstein distance", better: "lower" },
  MIA_AUC: { label: "MIA AUC", better: "lower" },
  NNDR: { label: "NNDR", better: "higher" },
  Mahalanobis_Distance: { label: "Mahalanobis distance", better: "higher" },
  Mean_Distance: { label: "Mean distance", better: "higher" },
  Median_Distance: { label: "Median distance", better: "higher" },
};

const DEFAULTS = {
  classification: { Utility: "Accuracy_Gap", Fidelity: "Quality_Score", Privacy: "MIA_AUC" },
  regression: { Utility: "R2_Gap", Fidelity: "Quality_Score", Privacy: "MIA_AUC" },
};

const state = {
  meta: null,
  utility: [],
  fidelity: [],
  privacy: [],
  figuresCatalog: null,
};

async function loadJSON(name) {
  const res = await fetch(`${DATA_BASE}/${name}?v=${DATA_VERSION}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${name}`);
  return res.json();
}

function fmt(n, digits = 4) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1000) return Number(n).toFixed(1);
  if (Math.abs(n) >= 10) return Number(n).toFixed(3);
  return Number(n).toFixed(digits);
}

function meanSd(mean, sd) {
  if (mean == null) return "—";
  if (sd == null) return fmt(mean);
  return `${fmt(mean)} ± ${fmt(sd)}`;
}

function shortDataset(ds) {
  return String(ds).replace(/^\d+\.\s*/, "");
}

function datasetSort(a, b) {
  const na = parseInt(String(a).match(/^(\d+)/)?.[1] || "999", 10);
  const nb = parseInt(String(b).match(/^(\d+)/)?.[1] || "999", 10);
  return na - nb || String(a).localeCompare(String(b));
}

function taskOfDataset(dataset) {
  const n = parseInt(String(dataset).match(/^(\d+)/)?.[1] || "0", 10);
  return n >= 10 ? "regression" : "classification";
}

function betterFor(metricId) {
  return METRIC_META[metricId]?.better || "higher";
}

function labelFor(metricId) {
  return METRIC_META[metricId]?.label || String(metricId).replaceAll("_", " ");
}

function heatColor(value, min, max, better) {
  if (value == null || min == null || max == null || min === max) {
    return "rgba(36, 48, 65, 0.6)";
  }
  let t = (value - min) / (max - min);
  t = Math.min(1, Math.max(0, t));
  if (better === "lower") t = 1 - t;
  const r = Math.round(255 * (1 - t) * 0.85 + 40);
  const g = Math.round(180 * t + 40);
  const b = Math.round(120 + 80 * (1 - Math.abs(t - 0.5) * 2));
  return `rgba(${r}, ${g}, ${b}, 0.55)`;
}

function currentRows() {
  const category = document.getElementById("bm-category").value;
  if (category === "Fidelity") return state.fidelity;
  if (category === "Privacy") return state.privacy;
  return state.utility;
}

function metricOptions() {
  const task = document.getElementById("bm-task").value;
  const category = document.getElementById("bm-category").value;
  const rows = currentRows().filter((r) => {
    if (category === "Utility") {
      return r.TaskType === task;
    }
    return taskOfDataset(r.Dataset) === task;
  });

  let ids;
  if (category === "Utility") {
    // Prefer gap / increase metrics, then base TRTR/TSTR metrics
    const preferred = Object.keys(METRIC_META).filter(
      (k) => !METRIC_META[k].task || METRIC_META[k].task === task
    );
    const present = new Set(rows.map((r) => r.Metric));
    ids = preferred.filter((id) => present.has(id));
    for (const id of [...present].sort()) {
      if (!ids.includes(id) && !String(id).endsWith("_Drop")) ids.push(id);
    }
  } else {
    ids = [...new Set(rows.map((r) => r.Metric))].sort();
  }
  return ids;
}

function matrixForMetric(metricId) {
  const task = document.getElementById("bm-task").value;
  const category = document.getElementById("bm-category").value;
  let rows = currentRows().filter((r) => r.Metric === metricId);
  if (category === "Utility") {
    // For base metrics like Accuracy, prefer TSTR (utility of synthetic)
    const hasEval = rows.some((r) => r.EvaluationType);
    if (hasEval && !/_Gap$|_Increase$|_Drop$/.test(metricId)) {
      const tstr = rows.filter((r) => r.EvaluationType === "TSTR");
      if (tstr.length) rows = tstr;
    }
    rows = rows.filter((r) => r.TaskType === task);
  } else {
    rows = rows.filter((r) => taskOfDataset(r.Dataset) === task);
  }

  const datasets = [...new Set(rows.map((r) => r.Dataset))].sort(datasetSort);
  const generators = GENERATORS.filter((g) => rows.some((r) => r.Generator === g));
  const values = generators.map((gen) =>
    datasets.map((ds) => {
      const hit = rows.find((r) => r.Generator === gen && r.Dataset === ds);
      return hit ? hit.Mean : null;
    })
  );
  const display = generators.map((gen) =>
    datasets.map((ds) => {
      const hit = rows.find((r) => r.Generator === gen && r.Dataset === ds);
      return hit ? meanSd(hit.Mean, hit.Std) : null;
    })
  );
  return { datasets, generators, values, display, better: betterFor(metricId) };
}

function refreshMetricSelect() {
  const sel = document.getElementById("bm-metric");
  const task = document.getElementById("bm-task").value;
  const category = document.getElementById("bm-category").value;
  const preferred = DEFAULTS[task]?.[category];
  const opts = metricOptions();
  sel.innerHTML = opts.map((id) => `<option value="${id}">${labelFor(id)}</option>`).join("");
  if (preferred && opts.includes(preferred)) sel.value = preferred;
}

function refreshDatasetSelect() {
  const task = document.getElementById("bm-task").value;
  const rows = currentRows().filter((r) =>
    r.TaskType ? r.TaskType === task : taskOfDataset(r.Dataset) === task
  );
  const datasets = [...new Set(rows.map((r) => r.Dataset))].sort(datasetSort);
  const sel = document.getElementById("bm-dataset");
  const prev = sel.value;
  sel.innerHTML =
    `<option value="">All datasets</option>` +
    datasets.map((d) => `<option value="${d}">${shortDataset(d)}</option>`).join("");
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

function renderRanking() {
  const metric = document.getElementById("bm-metric").value;
  const matrix = matrixForMetric(metric);
  document.getElementById("ranking-title").textContent = `Generator ranking — ${labelFor(metric)}`;
  const ranked = matrix.generators
    .map((gen, i) => {
      const vals = matrix.values[i].filter((v) => v != null);
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      return { generator: gen, mean, n: vals.length };
    })
    .filter((r) => r.mean != null);
  ranked.sort((a, b) =>
    matrix.better === "lower" ? a.mean - b.mean : b.mean - a.mean
  );
  document.getElementById("ranking-list").innerHTML = ranked
    .map(
      (r, i) => `
      <div class="rank-card">
        <div class="rank-num">#${i + 1}</div>
        <div class="rank-body">
          <strong>${r.generator}</strong>
          <span class="meta-line">${fmt(r.mean)} mean over ${r.n} datasets · ${matrix.better} better</span>
        </div>
        <div class="rank-bar-track">
          <div class="rank-bar" style="width:${Math.max(8, 100 - i * (90 / Math.max(ranked.length - 1, 1)))}%"></div>
        </div>
      </div>`
    )
    .join("");
}

function renderHeatmap() {
  const metric = document.getElementById("bm-metric").value;
  const matrix = matrixForMetric(metric);
  document.getElementById("heatmap-title").textContent = `Heatmap — ${labelFor(metric)}`;
  const flat = matrix.values.flat().filter((v) => v != null);
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;
  const head = `<tr><th>Generator</th>${matrix.datasets
    .map((d) => `<th>${shortDataset(d)}</th>`)
    .join("")}</tr>`;
  const body = matrix.generators
    .map((gen, i) => {
      const cells = matrix.datasets
        .map((ds, j) => {
          const v = matrix.values[i][j];
          const disp = matrix.display[i][j] || "—";
          const bg = heatColor(v, min, max, matrix.better);
          return `<td style="background:${bg}" title="${ds} · ${gen}: ${disp}">${
            v == null ? "—" : fmt(v, 3)
          }</td>`;
        })
        .join("");
      return `<tr><th>${gen}</th>${cells}</tr>`;
    })
    .join("");
  document.getElementById("heatmap").innerHTML = `
    <div class="table-wrap">
      <table class="heat-table"><thead>${head}</thead><tbody>${body}</tbody></table>
    </div>
    <p class="hint">Colour scale: greener ≈ better (${matrix.better}). Values are 10-seed means.</p>
  `;
}

function renderTable() {
  const task = document.getElementById("bm-task").value;
  const category = document.getElementById("bm-category").value;
  const dataset = document.getElementById("bm-dataset").value;
  const metric = document.getElementById("bm-metric").value;
  document.getElementById("table-title").textContent = `${category} Mean ± SD${
    dataset ? ` — ${shortDataset(dataset)}` : ""
  }`;

  let rows = currentRows();
  if (category === "Utility") rows = rows.filter((r) => r.TaskType === task);
  else rows = rows.filter((r) => taskOfDataset(r.Dataset) === task);
  if (dataset) rows = rows.filter((r) => r.Dataset === dataset);

  // Wide: one column per related metric for the pillar, focused on selected metric family
  let metrics;
  if (category === "Utility") {
    if (/_Gap$|_Increase$|_Drop$/.test(metric)) {
      metrics = [metric];
    } else {
      metrics = [`${metric}_TRTR`, `${metric}_TSTR`, `${metric}_Gap`].filter(Boolean);
      // Rows store Metric=Accuracy with EvaluationType
      const tableRows = [];
      const pairs = [...new Set(rows.map((r) => `${r.Dataset}||${r.Generator}`))];
      for (const key of pairs) {
        const [ds, gen] = key.split("||");
        const trtr = rows.find(
          (r) => r.Dataset === ds && r.Generator === gen && r.Metric === metric && r.EvaluationType === "TRTR"
        );
        const tstr = rows.find(
          (r) => r.Dataset === ds && r.Generator === gen && r.Metric === metric && r.EvaluationType === "TSTR"
        );
        const gap = rows.find(
          (r) =>
            r.Dataset === ds &&
            r.Generator === gen &&
            (r.Metric === `${metric}_Gap` || r.Metric === `${metric}_Increase` || r.Metric === `${metric}_Drop`)
        );
        tableRows.push({
          Dataset: ds,
          Generator: gen,
          TRTR: meanSd(trtr?.Mean, trtr?.Std),
          TSTR: meanSd(tstr?.Mean, tstr?.Std),
          Gap: meanSd(gap?.Mean, gap?.Std),
        });
      }
      tableRows.sort(
        (a, b) => datasetSort(a.Dataset, b.Dataset) || GENERATORS.indexOf(a.Generator) - GENERATORS.indexOf(b.Generator)
      );
      const thead = document.querySelector("#bm-table thead");
      const tbody = document.querySelector("#bm-table tbody");
      thead.innerHTML = `<tr><th>Dataset</th><th>Generator</th><th>${metric} TRTR</th><th>${metric} TSTR</th><th>Gap</th></tr>`;
      tbody.innerHTML = tableRows
        .map(
          (r) =>
            `<tr><td>${shortDataset(r.Dataset)}</td><td>${r.Generator}</td><td>${r.TRTR}</td><td>${r.TSTR}</td><td>${r.Gap}</td></tr>`
        )
        .join("");
      return;
    }
  } else {
    metrics = metricOptions();
  }

  const pairMap = new Map();
  for (const r of rows) {
    if (category !== "Utility" && !metrics.includes(r.Metric)) continue;
    if (category === "Utility" && r.Metric !== metric) continue;
    const key = `${r.Dataset}||${r.Generator}`;
    if (!pairMap.has(key)) pairMap.set(key, { Dataset: r.Dataset, Generator: r.Generator });
    const row = pairMap.get(key);
    if (category === "Utility") {
      row[metric] = meanSd(r.Mean, r.Std);
    } else {
      row[r.Metric] = meanSd(r.Mean, r.Std);
    }
  }
  const tableRows = [...pairMap.values()].sort(
    (a, b) => datasetSort(a.Dataset, b.Dataset) || GENERATORS.indexOf(a.Generator) - GENERATORS.indexOf(b.Generator)
  );
  const cols = category === "Utility" ? [metric] : metrics;
  const thead = document.querySelector("#bm-table thead");
  const tbody = document.querySelector("#bm-table tbody");
  thead.innerHTML = `<tr><th>Dataset</th><th>Generator</th>${cols
    .map((c) => `<th>${labelFor(c)}</th>`)
    .join("")}</tr>`;
  tbody.innerHTML = tableRows
    .map(
      (r) =>
        `<tr><td>${shortDataset(r.Dataset)}</td><td>${r.Generator}</td>${cols
          .map((c) => `<td>${r[c] ?? "—"}</td>`)
          .join("")}</tr>`
    )
    .join("");
}

async function refreshAll() {
  refreshMetricSelect();
  refreshDatasetSelect();
  renderRanking();
  renderHeatmap();
  renderTable();
}

async function init() {
  bindViewTabs();

  const [meta, utility, fidelity, privacy] = await Promise.all([
    loadJSON("meta.json"),
    loadJSON("utility_agg.json"),
    loadJSON("fidelity.json"),
    loadJSON("privacy.json"),
  ]);
  state.meta = meta;
  state.utility = utility;
  state.fidelity = fidelity;
  state.privacy = privacy;

  document.getElementById("stat-datasets").textContent = meta.n_datasets ?? "—";
  document.getElementById("stat-generators").textContent = (meta.generators || GENERATORS).length;
  document.getElementById("stat-seeds").textContent = meta.n_generator_seeds ?? 10;
  document.getElementById("stat-runs").textContent = (meta.n_runs ?? 1200).toLocaleString();
  document.getElementById("source-note").textContent =
    `${meta.source || "multi-seed"} · ${meta.aggregation || "Mean ± SD over 10 seeds"}`;

  document.getElementById("bm-task").addEventListener("change", refreshAll);
  document.getElementById("bm-category").addEventListener("change", refreshAll);
  document.getElementById("bm-metric").addEventListener("change", () => {
    renderRanking();
    renderHeatmap();
    renderTable();
  });
  document.getElementById("bm-dataset").addEventListener("change", renderTable);

  await refreshAll();
  await loadFigures();
}

function bindViewTabs() {
  document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".view-tab").forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      const view = tab.dataset.view;
      document.getElementById("view-results").classList.toggle("hidden", view !== "results");
      document.getElementById("view-figures").classList.toggle("hidden", view !== "figures");
    });
  });
}

async function loadFigures() {
  try {
    const res = await fetch(`figures/catalog.json?v=${DATA_VERSION}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`catalog ${res.status}`);
    state.figuresCatalog = await res.json();
  } catch (err) {
    document.getElementById("figures-note").textContent = `Figures unavailable: ${err.message}`;
    return;
  }

  const cats = state.figuresCatalog.categories || [];
  const sel = document.getElementById("fig-category");
  sel.innerHTML =
    `<option value="">All categories</option>` +
    cats.map((c) => `<option value="${c.id}">${c.label}</option>`).join("");
  document.getElementById("figures-note").textContent =
    `${state.figuresCatalog.n_figures} diagrams from ${state.figuresCatalog.source}`;

  document.getElementById("fig-category").addEventListener("change", renderFigures);
  document.getElementById("fig-task").addEventListener("change", renderFigures);
  document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
  document.getElementById("lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox") closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
  renderFigures();
}

function renderFigures() {
  const category = document.getElementById("fig-category").value;
  const task = document.getElementById("fig-task").value;
  const figs = (state.figuresCatalog?.figures || []).filter((f) => {
    if (category && f.category !== category) return false;
    if (task && f.task && f.task !== task) return false;
    if (task && !f.task && category !== "workflow" && category !== "main") return false;
    return true;
  });

  const grid = document.getElementById("figure-grid");
  if (!figs.length) {
    grid.innerHTML = `<p class="hint">No figures match these filters.</p>`;
    return;
  }
  grid.innerHTML = figs
    .map(
      (f) => `
      <article class="figure-card" data-src="figures/${f.file}" data-title="${f.title}" data-cat="${f.category_label}">
        <img src="figures/${f.file}" alt="${f.title}" loading="lazy" />
        <div class="fig-meta">
          <p class="fig-cat">${f.category_label}${f.task ? " · " + f.task : ""}</p>
          <h3>${f.title}</h3>
        </div>
      </article>`
    )
    .join("");

  grid.querySelectorAll(".figure-card").forEach((card) => {
    card.addEventListener("click", () => openLightbox(card.dataset.src, card.dataset.title, card.dataset.cat));
  });
}

function openLightbox(src, title, cat) {
  const box = document.getElementById("lightbox");
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-img").alt = title;
  document.getElementById("lightbox-caption").textContent = `${cat} — ${title}`;
  box.classList.remove("hidden");
}

function closeLightbox() {
  document.getElementById("lightbox").classList.add("hidden");
}

init().catch((err) => {
  document.getElementById("source-note").textContent = `Failed to load: ${err.message}`;
});
