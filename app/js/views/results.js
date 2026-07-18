/* Vista Resultados: dos listas separadas (estudios y runs) con buscador. Al seleccionar un
   estudio se abre su análisis (study.js); al seleccionar un run, el suyo (run.js). El panel
   derecho aloja el detalle. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt } = global.TFM;
  const S = () => global.TFM.state;

  let filterRuns = "";
  let filterStudies = "";
  let activeId = null;

  function runItem(run) {
    const s = run.summary || {};
    const active = activeId === run.run_id ? " active" : "";
    return `<div class="result-item${active}" onclick="TFM.views.results.openRun('${escapeHtml(run.run_id)}')">
      <div class="title">${escapeHtml(run.label || run.run_id)}</div>
      <div class="sub">${escapeHtml(run.run_kind || "")} · rank-IC ${fmt(s.mean_rank_ic)} · ${escapeHtml(run.status || "")}</div>
    </div>`;
  }

  function studyItem(study) {
    const active = activeId === study.study_id ? " active" : "";
    return `<div class="result-item${active}" onclick="TFM.views.results.openStudy('${escapeHtml(study.study_id)}')">
      <div class="title">${escapeHtml(study.name || study.study_id)}</div>
      <div class="sub">${escapeHtml(study.kind || "")} · ${escapeHtml(study.status || "")}</div>
    </div>`;
  }

  function refreshRuns() {
    const node = el("results-runs");
    if (!node) return;
    const q = filterRuns.toLowerCase();
    const rows = S().runs.filter((r) => JSON.stringify(r).toLowerCase().includes(q));
    el("results-runs-count").textContent = `${rows.length}`;
    node.innerHTML = rows.length ? rows.map(runItem).join("") : `<p class="empty">Sin runs.</p>`;
  }

  function refreshStudies() {
    const node = el("results-studies");
    if (!node) return;
    const q = filterStudies.toLowerCase();
    const rows = S().studies.filter((s) => JSON.stringify(s).toLowerCase().includes(q));
    el("results-studies-count").textContent = `${rows.length}`;
    node.innerHTML = rows.length ? rows.map(studyItem).join("") : `<p class="empty">Sin estudios.</p>`;
  }

  function openRun(runId) {
    activeId = runId;
    refreshRuns();
    refreshStudies();
    global.TFM.views.run.open(runId, el("results-detail"));
  }

  function openStudy(studyId) {
    activeId = studyId;
    refreshRuns();
    refreshStudies();
    global.TFM.views.study.open(studyId, el("results-detail"));
  }

  async function render(container) {
    container.innerHTML = `
      <div class="grid results-grid">
        <section class="panel">
          <div class="list-header"><h3>Estudios</h3><span class="count" id="results-studies-count">…</span></div>
          <input id="search-studies" placeholder="Buscar estudio" oninput="TFM.views.results.searchStudies(this.value)">
          <div id="results-studies" class="scroll" style="margin-top:10px"></div>
          <div class="list-header" style="margin-top:18px"><h3>Runs</h3><span class="count" id="results-runs-count">…</span></div>
          <input id="search-runs" placeholder="Buscar por etiqueta, hash o study" oninput="TFM.views.results.searchRuns(this.value)">
          <div id="results-runs" class="scroll" style="margin-top:10px"></div>
        </section>
        <section class="panel" id="results-detail">
          <h3>Selecciona un estudio o un run</h3>
          <p class="muted">A la izquierda tienes los estudios y los runs registrados. Al seleccionar uno verás su resumen, rendimiento, aprendizaje, cartera y stocks.</p>
        </section>
      </div>`;
    await global.TFM.loadStudies();
    await global.TFM.loadJobsAndRuns();
    refreshStudies();
    refreshRuns();
  }

  global.TFM.views.results = {
    render,
    refreshRuns,
    refreshStudies,
    openRun,
    openStudy,
    searchRuns(v) { filterRuns = v; refreshRuns(); },
    searchStudies(v) { filterStudies = v; refreshStudies(); },
  };
})(window);
