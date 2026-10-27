const state = {
  datasets: [],
  generators: [],
  selectedDataset: null,
  selectedGenerators: new Set(),
  taskFilter: "all",
  activeJobId: null,
  activeGenerator: null,
  pollTimer: null,
  overview: null,
  metrics: [],
  view: "benchmark",
};

const els = {
  datasetGrid: document.getElementById("dataset-grid"),
  generatorGroups: document.getElementById("generator-groups"),
  generateBtn: document.getElementById("generate-btn"),
  selectAllGenerators: document.getElementById("select-all-generators"),
  clearGenerators: document.getElementById("clear-generators"),
  nSamples: document.getElementById("n-samples"),
  seed: document.getElementById("seed"),
  selectionSummary: document.getElementById("selection-summary"),
  resultsPanel: document.getElementById("results-panel"),
  jobStatusChip: document.getElementById("job-status-chip"),
  progressBar: document.getElementById("progress-bar"),
  statusMessage: document.getElementById("status-message"),
  resultCards: document.getElementById("result-cards"),
  previewSection: document.getElementById("preview-section"),
  previewTitle: document.getElementById("preview-title"),
  previewTable: document.getElementById("preview-table"),
  downloadActive: document.getElementById("download-active"),
  errorBox: document.getElementById("error-box"),
  bmTask: document.getElementById("bm-task"),
  bmCategory: document.getElementById("bm-category"),
  bmMetric: document.getElementById("bm-metric"),
  bmDataset: document.getElementById("bm-dataset"),
  rankingList: document.getElementById("ranking-list"),
  heatmap: document.getElementById("heatmap"),
  bmTable: document.getElementById("bm-table"),
  benchmarkSource: document.getElementById("benchmark-source"),
  rankingTitle: document.getElementById("ranking-title"),
  heatmapTitle: document.getElementById("heatmap-title"),
  tableTitle: document.getElementById("table-title"),
  viewBenchmark: document.getElementById("view-benchmark"),
  viewGenerate: document.getElementById("view-generate"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function fmt(n, digits = 4) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(1);
  if (Math.abs(n) >= 10) return n.toFixed(3);
  return n.toFixed(digits);
}

function defaultMetricFor(task, category) {
  const prefs = {
    classification: {
      Utility: "Accuracy_Gap",
      Fidelity: "Quality_Score",
      Privacy: "MIA_AUC",
      Compute: "total_time_seconds",
    },
    regression: {
      Utility: "R2_Gap",
      Fidelity: "Quality_Score",
      Privacy: "MIA_AUC",
      Compute: "total_time_seconds",
    },
  };
  return prefs[task]?.[category] || null;
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
      state.view = tab.dataset.view;
      els.viewBenchmark.classList.toggle("hidden", state.view !== "benchmark");
      els.viewGenerate.classList.toggle("hidden", state.view !== "generate");
    });
  });
}

async function loadOverview() {
  state.overview = await api("/api/benchmark/overview");
  document.getElementById("stat-datasets").textContent = state.overview.n_datasets;
  document.getElementById("stat-generators").textContent = state.overview.n_generators;
  document.getElementById("stat-seeds").textContent = state.overview.n_generator_seeds;
  document.getElementById("stat-runs").textContent = state.overview.n_runs || "1,200";
  els.benchmarkSource.textContent =
    `${state.overview.source} · ${state.overview.aggregation}`;

  const taskOpts = state.overview.tasks
    .map((t) => `<option value="${t}">${t[0].toUpperCase()}${t.slice(1)}</option>`)
    .join("");
  els.bmTask.innerHTML = taskOpts;

  const dsOpts =
    `<option value="">All datasets</option>` +
    state.overview.datasets.map((d) => `<option value="${d}">${d}</option>`).join("");
  els.bmDataset.innerHTML = dsOpts;
}

async function refreshMetricOptions() {
  const task = els.bmTask.value;
  const category = els.bmCategory.value;
  state.metrics = await api(
    `/api/benchmark/metrics?task=${encodeURIComponent(task)}&category=${encodeURIComponent(category)}`
  );
  const preferred = defaultMetricFor(task, category);
  const options = state.metrics
    .filter((m) => m.category === category || !m.category)
    .map(
      (m) =>
        `<option value="${m.id}" ${m.id === preferred ? "selected" : ""}>${m.label}</option>`
    )
    .join("");
  els.bmMetric.innerHTML = options || `<option value="">No metrics</option>`;
  if (preferred && ![...els.bmMetric.options].some((o) => o.value === preferred)) {
    // keep first available
  } else if (preferred) {
    els.bmMetric.value = preferred;
  }
}

async function renderRanking() {
  const task = els.bmTask.value;
  const metric = els.bmMetric.value;
  if (!metric) {
    els.rankingList.innerHTML = `<p class="hint">Select a metric.</p>`;
    return;
  }
  const meta = state.metrics.find((m) => m.id === metric);
  els.rankingTitle.textContent = `Generator ranking — ${meta?.label || metric}`;
  const rows = await api(
    `/api/benchmark/ranking?task=${encodeURIComponent(task)}&metric=${encodeURIComponent(metric)}`
  );
  const better = rows[0]?.better || meta?.better || "higher";
  els.rankingList.innerHTML = rows
    .map(
      (r, i) => `
      <div class="rank-card">
        <div class="rank-num">#${r.rank}</div>
        <div class="rank-body">
          <strong>${r.generator}</strong>
          <span class="meta-line">${fmt(r.mean_over_datasets)} mean over ${r.n_datasets} datasets · ${better} better</span>
        </div>
        <div class="rank-bar-track">
          <div class="rank-bar" style="width:${Math.max(8, 100 - i * (90 / Math.max(rows.length - 1, 1)))}%"></div>
        </div>
      </div>`
    )
    .join("");
}

function heatColor(value, min, max, better) {
  if (value == null || min == null || max == null || min === max) {
    return "rgba(36, 48, 65, 0.6)";
  }
  let t = (value - min) / (max - min);
  t = Math.min(1, Math.max(0, t));
  // For "lower better", invert so low values are green-ish
  if (better === "lower") t = 1 - t;
  const r = Math.round(255 * (1 - t) * 0.85 + 40);
  const g = Math.round(180 * t + 40);
  const b = Math.round(120 + 80 * (1 - Math.abs(t - 0.5) * 2));
  return `rgba(${r}, ${g}, ${b}, 0.55)`;
}

async function renderHeatmap() {
  const task = els.bmTask.value;
  const metric = els.bmMetric.value;
  if (!metric) {
    els.heatmap.innerHTML = "";
    return;
  }
  const matrix = await api(
    `/api/benchmark/matrix?task=${encodeURIComponent(task)}&metric=${encodeURIComponent(metric)}`
  );
  els.heatmapTitle.textContent = `Heatmap — ${matrix.label}`;

  const flat = matrix.values.flat().filter((v) => v != null);
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;

  const head = `<tr><th>Generator</th>${matrix.datasets
    .map((d) => `<th>${d}</th>`)
    .join("")}</tr>`;
  const body = matrix.generators
    .map((gen, i) => {
      const cells = matrix.datasets
        .map((ds, j) => {
          const v = matrix.values[i][j];
          const disp = matrix.display[i][j] || (v == null ? "—" : fmt(v));
          const bg = heatColor(v, min, max, matrix.better);
          return `<td style="background:${bg}" title="${ds} · ${gen}: ${disp}">${
            v == null ? "—" : fmt(v, 3)
          }</td>`;
        })
        .join("");
      return `<tr><th>${gen}</th>${cells}</tr>`;
    })
    .join("");

  els.heatmap.innerHTML = `
    <div class="table-wrap table-wrap-heat">
      <table class="heat-table"><thead>${head}</thead><tbody>${body}</tbody></table>
    </div>
    <p class="hint">Colour scale: greener ≈ better (${matrix.better}). Values are 10-seed means.</p>
  `;
}

async function renderTable() {
  const task = els.bmTask.value;
  const category = els.bmCategory.value;
  const dataset = els.bmDataset.value;
  const qs = new URLSearchParams({ task, category });
  if (dataset) qs.set("dataset", dataset);
  const table = await api(`/api/benchmark/table?${qs.toString()}`);
  els.tableTitle.textContent = `${category} Mean ± SD${dataset ? ` — ${dataset}` : ""}`;

  const metricCols = table.columns.filter((c) => c !== "dataset" && c !== "generator");
  const thead = els.bmTable.querySelector("thead");
  const tbody = els.bmTable.querySelector("tbody");
  thead.innerHTML = `<tr>
    <th>Dataset</th><th>Generator</th>
    ${metricCols.map((c) => `<th>${c.replaceAll("_", " ")}</th>`).join("")}
  </tr>`;
  tbody.innerHTML = table.rows
    .map(
      (row) => `<tr>
        <td>${row.dataset}</td>
        <td>${row.generator}</td>
        ${metricCols.map((c) => `<td>${row[c] ?? "—"}</td>`).join("")}
      </tr>`
    )
    .join("");
}

async function refreshBenchmark() {
  try {
    await refreshMetricOptions();
    await Promise.all([renderRanking(), renderHeatmap(), renderTable()]);
  } catch (err) {
    els.benchmarkSource.textContent = `Failed to load results: ${err.message}`;
  }
}

function bindBenchmarkControls() {
  els.bmTask.addEventListener("change", refreshBenchmark);
  els.bmCategory.addEventListener("change", refreshBenchmark);
  els.bmMetric.addEventListener("change", async () => {
    await Promise.all([renderRanking(), renderHeatmap()]);
  });
  els.bmDataset.addEventListener("change", renderTable);
}

function updateSelectionSummary() {
  const ds = state.datasets.find((d) => d.id === state.selectedDataset);
  const count = state.selectedGenerators.size;
  if (!ds) {
    els.selectionSummary.textContent = "Select a dataset and at least one generator.";
  } else if (count === 0) {
    els.selectionSummary.textContent = `${ds.name} selected — choose one or more generators.`;
  } else {
    els.selectionSummary.textContent = `${ds.name} · ${count} generator${count > 1 ? "s" : ""} · ${els.nSamples.value} rows · seed ${els.seed.value}`;
  }
  els.generateBtn.disabled = !ds || count === 0;
}

function renderDatasets() {
  const filtered = state.datasets.filter((ds) =>
    state.taskFilter === "all" ? true : ds.task === state.taskFilter
  );

  els.datasetGrid.innerHTML = filtered
    .map(
      (ds) => `
      <button
        type="button"
        class="dataset-card ${state.selectedDataset === ds.id ? "selected" : ""}"
        data-id="${ds.id}"
        aria-pressed="${state.selectedDataset === ds.id}"
      >
        <span class="badge badge-${ds.task}">${ds.task}</span>
        <h3>${ds.name}</h3>
        <p>${ds.description || `Target: ${ds.target_col}`}</p>
        <div class="meta-line">${ds.row_count} rows · ${ds.column_count} cols</div>
      </button>
    `
    )
    .join("");

  els.datasetGrid.querySelectorAll(".dataset-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedDataset = card.dataset.id;
      renderDatasets();
      updateSelectionSummary();
    });
  });
}

function renderGenerators() {
  const families = ["GAN", "Diffusion", "SDV"];
  els.generatorGroups.innerHTML = families
    .map((family) => {
      const gens = state.generators.filter((g) => g.family === family);
      return `
        <div class="generator-family">
          <h3>${family}</h3>
          <div class="cards">
            ${gens
              .map((g) => {
                const selected = state.selectedGenerators.has(g.id);
                const unavailable = !g.available;
                return `
                  <button
                    type="button"
                    class="generator-card ${selected ? "selected" : ""} ${unavailable ? "unavailable" : ""}"
                    data-id="${g.id}"
                    ${unavailable ? "disabled title=\"" + (g.reason || "Unavailable") + "\"" : ""}
                    aria-pressed="${selected}"
                  >
                    <h3>${g.name}</h3>
                    <p>${g.description}</p>
                    ${g.auto_setup ? '<div class="meta-line">Auto-setup on first use</div>' : ""}
                  </button>
                `;
              })
              .join("")}
          </div>
        </div>
      `;
    })
    .join("");

  els.generatorGroups.querySelectorAll(".generator-card:not(:disabled)").forEach((card) => {
    card.addEventListener("click", () => {
      const id = card.dataset.id;
      if (state.selectedGenerators.has(id)) {
        state.selectedGenerators.delete(id);
      } else {
        state.selectedGenerators.add(id);
      }
      renderGenerators();
      updateSelectionSummary();
    });
  });
}

function bindFilters() {
  document.querySelectorAll(".filter-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".filter-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.taskFilter = tab.dataset.filter;
      renderDatasets();
    });
  });

  els.selectAllGenerators.addEventListener("click", () => {
    state.generators.filter((g) => g.available).forEach((g) => state.selectedGenerators.add(g.id));
    renderGenerators();
    updateSelectionSummary();
  });

  els.clearGenerators.addEventListener("click", () => {
    state.selectedGenerators.clear();
    renderGenerators();
    updateSelectionSummary();
  });

  els.nSamples.addEventListener("input", updateSelectionSummary);
  els.seed.addEventListener("input", updateSelectionSummary);
}

async function startGeneration() {
  els.generateBtn.disabled = true;
  els.resultsPanel.classList.remove("hidden");
  els.previewSection.classList.add("hidden");
  els.errorBox.classList.add("hidden");
  els.resultCards.innerHTML = "";
  setJobUi({ status: "queued", progress: 0, message: "Starting job…" });

  try {
    const job = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: state.selectedDataset,
        generator_ids: [...state.selectedGenerators],
        n_samples: Number(els.nSamples.value),
        seed: Number(els.seed.value),
      }),
    });
    state.activeJobId = job.job_id;
    pollJob();
  } catch (err) {
    setJobUi({ status: "failed", progress: 1, message: "Failed to start", error: err.message });
    els.generateBtn.disabled = false;
  }
}

function setJobUi({ status, progress, message, error }) {
  els.jobStatusChip.textContent = status;
  els.jobStatusChip.className = `status-chip ${status}`;
  els.progressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
  els.statusMessage.textContent = message || "";
  if (error) {
    els.errorBox.textContent = error;
    els.errorBox.classList.remove("hidden");
  }
}

async function pollJob() {
  if (!state.activeJobId) return;
  try {
    const job = await api(`/api/jobs/${state.activeJobId}`);
    setJobUi(job);
    if (job.status === "running" || job.status === "queued") {
      state.pollTimer = setTimeout(pollJob, 1200);
      return;
    }
    if (job.status === "completed") {
      renderResultCards(job);
      els.generateBtn.disabled = false;
      return;
    }
    if (job.status === "failed") {
      setJobUi({ ...job, error: job.error || "Generation failed" });
      els.generateBtn.disabled = false;
    }
  } catch (err) {
    setJobUi({ status: "failed", progress: 1, message: "Polling failed", error: err.message });
    els.generateBtn.disabled = false;
  }
}

function renderResultCards(job) {
  const entries = Object.entries(job.results || {});
  els.resultCards.innerHTML = entries
    .map(
      ([id, meta]) => `
      <div class="result-card" data-id="${id}">
        <h4>${meta.generator_name}</h4>
        <div class="score">${meta.quality_score != null ? `Quality: ${meta.quality_score.toFixed(3)}` : "Quality: n/a"}</div>
        <div class="meta-line">${meta.rows} rows</div>
      </div>
    `
    )
    .join("");

  els.resultCards.querySelectorAll(".result-card").forEach((card) => {
    card.addEventListener("click", () => showPreview(card.dataset.id));
  });

  if (entries.length) {
    showPreview(entries[0][0]);
  }
}

async function showPreview(generatorId) {
  state.activeGenerator = generatorId;
  els.resultCards.querySelectorAll(".result-card").forEach((c) => {
    c.classList.toggle("active", c.dataset.id === generatorId);
  });

  const gen = state.generators.find((g) => g.id === generatorId);
  els.previewTitle.textContent = `${gen?.name || generatorId} — preview`;
  els.previewSection.classList.remove("hidden");

  const preview = await api(`/api/jobs/${state.activeJobId}/preview/${generatorId}?limit=15`);
  const thead = els.previewTable.querySelector("thead");
  const tbody = els.previewTable.querySelector("tbody");
  thead.innerHTML = `<tr>${preview.columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  tbody.innerHTML = preview.rows
    .map(
      (row) =>
        `<tr>${preview.columns.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`
    )
    .join("");
}

els.downloadActive.addEventListener("click", () => {
  if (state.activeJobId && state.activeGenerator) {
    window.location.href = `/api/jobs/${state.activeJobId}/download/${state.activeGenerator}`;
  }
});

els.generateBtn.addEventListener("click", startGeneration);

async function init() {
  bindViewTabs();
  bindFilters();
  bindBenchmarkControls();

  const [datasets, generators] = await Promise.all([
    api("/api/datasets"),
    api("/api/generators"),
  ]);
  state.datasets = datasets;
  state.generators = generators;

  state.generators
    .filter((g) => g.available)
    .slice(0, 2)
    .forEach((g) => state.selectedGenerators.add(g.id));

  renderDatasets();
  renderGenerators();
  updateSelectionSummary();

  await loadOverview();
  await refreshBenchmark();
}

init().catch((err) => {
  if (els.selectionSummary) {
    els.selectionSummary.textContent = `Failed to load app data: ${err.message}`;
  }
  if (els.benchmarkSource) {
    els.benchmarkSource.textContent = `Failed to load app data: ${err.message}`;
  }
});
