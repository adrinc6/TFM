/* Vista Resultados: dos tablas grandes conmutables (estudios ↔ runs). Al entrar se muestra la
   tabla de estudios; un botón arriba-derecha conmuta a la tabla de todos los runs. Un clic en una
   fila entra al análisis correspondiente (study.js / run.js), con botón "Volver a la lista". */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct } = global.TFM;
  const S = () => global.TFM.state;

  let mode = "studies";   // "studies" | "runs"
  let filter = "";

  // Métricas de un estudio derivadas de los runs del registro (n.º de runs y mejor rank-IC).
  function studyMetrics(studyId) {
    const members = S().runs.filter((r) => r.study_id === studyId);
    let best = null;
    for (const r of members) {
      const ic = (r.summary || {}).mean_rank_ic;
      if (typeof ic === "number" && (best === null || ic > best)) best = ic;
    }
    return { n_runs: members.length, best_rank_ic: best };
  }

  // --- Tabla de estudios ---
  function studyRows() {
    const q = filter.toLowerCase();
    const rows = S().studies.filter((s) => JSON.stringify(s).toLowerCase().includes(q));
    if (!rows.length) return `<tr><td colspan="6" class="empty">Sin estudios.</td></tr>`;
    return rows.map((s) => {
      const d = studyMetrics(s.study_id);
      const official = s.protocol_version === 2;
      const decision = s.decision_summary || {};
      const selectedIc = official ? decision.signal_rank_ic : d.best_rank_ic;
      const progress = `${s.completed_evaluations ?? 0}/${s.evaluation_budget?.total ?? 48}`;
      const metric = official && ["queued", "running"].includes(String(s.status || ""))
        ? progress : fmt(selectedIc);
      const metricClass = typeof selectedIc === "number"
        ? (selectedIc >= 0 ? "positive" : "negative") : "";
      const note = official && decision.verdict
        ? `${decision.verdict} · señal: ${decision.signal_winner || "—"} · cartera: ${decision.portfolio_winner || "—"}`
        : (s.description || "").slice(0, 80);
      return `<tr class="click" onclick="TFM.views.results.openStudy('${escapeHtml(s.study_id)}')">
        <td><strong>${escapeHtml(s.name || s.study_id)}</strong><br><small class="muted mono">${escapeHtml(s.study_id)}</small></td>
        <td>${escapeHtml(s.kind || "—")}</td>
        <td>${escapeHtml(s.status || "—")}</td>
        <td>${d.n_runs || "—"}</td>
        <td class="${metricClass}">${metric}</td>
        <td>${escapeHtml(note)}</td>
      </tr>`;
    }).join("");
  }

  function studiesTable() {
    return `<div class="table-wrap scroll" style="max-height:none">
      <table class="data"><thead><tr>
        <th>Estudio</th><th>Tipo</th><th>Estado</th><th>Runs</th><th>Rank-IC seleccionado / progreso</th><th>Veredicto / notas</th>
      </tr></thead><tbody>${studyRows()}</tbody></table></div>`;
  }

  // --- Tabla de runs ---
  function runRows() {
    const q = filter.toLowerCase();
    const rows = S().runs.filter((r) => JSON.stringify(r).toLowerCase().includes(q));
    if (!rows.length) return `<tr><td colspan="7" class="empty">Sin runs.</td></tr>`;
    return rows.map((r) => {
      const s = r.summary || {};
      return `<tr class="click" onclick="TFM.views.results.openRun('${escapeHtml(r.run_id)}')">
        <td class="col-run"><strong>${escapeHtml(r.label || r.run_id)}</strong><br><small class="muted mono">${escapeHtml(r.run_id)}</small></td>
        <td>${escapeHtml(r.run_kind || "—")}</td>
        <td class="${(s.mean_rank_ic || 0) >= 0 ? "positive" : "negative"}">${fmt(s.mean_rank_ic)}</td>
        <td class="${(s.cagr_difference || 0) >= 0 ? "positive" : "negative"}">${pct(s.cagr_difference)}</td>
        <td class="${typeof s.max_drawdown_benchmark === "number" ? (s.max_drawdown <= s.max_drawdown_benchmark ? "positive" : "negative") : ""}">${pct(s.max_drawdown)}</td>
        <td>${escapeHtml(String(r.created_at_utc || "").slice(0, 10))}</td>
        <td>${escapeHtml(r.status || "—")}</td>
      </tr>`;
    }).join("");
  }

  function runsTable() {
    return `<div class="table-wrap scroll" style="max-height:none">
      <table class="data"><thead><tr>
        <th class="col-run">Run</th><th>Tipo</th><th>rank-IC</th><th>CAGR vs bench</th><th>Max DD</th><th>Fecha</th><th>Estado</th>
      </tr></thead><tbody>${runRows()}</tbody></table></div>`;
  }

  // --- Render de la lista (tabla grande + botón conmutador) ---
  function renderList() {
    const container = el("results");
    const title = mode === "studies" ? "Estudios" : "Runs individuales";
    const toggleLabel = mode === "studies" ? "Ver runs individuales ▸" : "◂ Ver estudios";
    const count = mode === "studies" ? S().studies.length : S().runs.length;
    container.innerHTML = `
      <div class="list-toolbar">
        <div><h3 style="margin:0">${title}</h3><span class="count">${count} en total</span></div>
        <div class="toolbar-actions">
          <input id="results-filter" placeholder="Buscar…" value="${escapeHtml(filter)}" oninput="TFM.views.results.setFilter(this.value)">
          <button class="button results-toggle" onclick="TFM.views.results.toggle()">${toggleLabel}</button>
        </div>
      </div>
      <div id="results-table">${mode === "studies" ? studiesTable() : runsTable()}</div>`;
  }

  function refreshTable() {
    const node = el("results-table");
    if (node) node.innerHTML = mode === "studies" ? studiesTable() : runsTable();
  }

  // --- Detalle: envuelve el análisis con un botón de volver ---
  function openDetail(kind, id) {
    const container = el("results");
    container.innerHTML = `
      <div class="list-toolbar">
        <button class="button ghost" onclick="TFM.views.results.renderList()">◂ Volver a la lista</button>
        ${kind === "study" ? '<span id="study-live-actions"></span>' : ""}
      </div>
      <section class="panel" id="results-detail"></section>`;
    const target = el("results-detail");
    if (kind === "study") global.TFM.views.study.open(id, target);
    else global.TFM.views.run.open(id, target);
  }

  // --- API pública de la vista (el router llama render()) ---
  async function render() {
    await global.TFM.loadStudies();
    await global.TFM.loadJobsAndRuns();
    renderList();
  }

  global.TFM.views.results = {
    render,
    renderList,
    refreshRuns() { if (el("results-table") && mode === "runs") refreshTable(); },
    refreshStudies() { if (el("results-table") && mode === "studies") refreshTable(); },
    toggle() { mode = mode === "studies" ? "runs" : "studies"; filter = ""; renderList(); },
    setFilter(v) { filter = v; refreshTable(); },
    openStudy(id) { openDetail("study", id); },
    openRun(id) { openDetail("run", id); },
  };
})(window);
