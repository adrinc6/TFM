/* Rankings: puntuaciones de los agentes por acción en un snapshot concreto del run.
   Permite elegir snapshot, ordenar por el agente (o meta) elegido y buscar por ticker.
   Sustituye al antiguo export ranking_by_snapshot.csv leyendo agent_scores.parquet directo.
   Consume /api/ranking/snapshots y /api/ranking. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, table } = global.TFM;

  // { runId, snapshots, snapshot, sort, order }
  const ctx = {};

  // Columnas ordenables por las que tiene sentido rankear las acciones: los 5 agentes y el meta.
  // El value es la columna real de agent_scores.parquet; el label, su nombre legible.
  const SORTABLE = [
    ["meta_score", "Meta"],
    ["quality", "Calidad"],
    ["value", "Valor"],
    ["growth", "Crecimiento"],
    ["momentum", "Momentum"],
    ["risk", "Riesgo"],
  ];

  // Columnas mostradas en la tabla (orden fijo, legible), con su etiqueta.
  const COLUMNS = ["ticker", "meta_score", "meta_rank",
                   "quality", "value", "growth", "momentum", "risk"];
  const LABELS = {
    ticker: "Ticker", meta_score: "Meta", meta_rank: "Rango meta",
    quality: "Calidad", value: "Valor", growth: "Crecimiento",
    momentum: "Momentum", risk: "Riesgo",
  };

  async function render(container, runId) {
    ctx.runId = runId;
    ctx.sort = "meta_score";
    ctx.order = "desc";
    container.innerHTML = `<p class="muted">Cargando rankings…</p>`;
    let meta;
    try { meta = await api("/api/ranking/snapshots?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    if (!meta.available) {
      container.innerHTML = `<div class="notice">${escapeHtml(meta.message || "Este run no conserva el ranking de agentes.")}</div>`;
      return;
    }
    ctx.snapshots = meta.snapshots || [];
    ctx.snapshot = ctx.snapshots.length ? ctx.snapshots[0] : "";
    container.innerHTML = `
      <section class="parameter-group"><h4>Rankings por snapshot</h4>
        <p class="muted">Puntuaciones de los agentes para cada acción en la fecha elegida. Ordena por el agente que quieras y busca un ticker.</p>
        <div class="formgrid">
          <label class="field">Snapshot<select id="rk-snapshot" onchange="TFM.views.rankings.onSnapshot()">
            ${ctx.snapshots.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("")}
          </select></label>
          <label class="field">Ordenar por<select id="rk-sort" onchange="TFM.views.rankings.onSort()">
            ${SORTABLE.map(([id, label]) => `<option value="${id}">${label}</option>`).join("")}
          </select></label>
          <label class="field">Buscar ticker<input id="rk-filter" placeholder="AAPL" data-filter-table="rk-table"></label>
        </div>
      </section>
      <div id="rk-output" class="muted">Cargando…</div>`;
    load();
  }

  function onSnapshot() { ctx.snapshot = el("rk-snapshot").value; load(); }
  function onSort() { ctx.sort = el("rk-sort").value; load(); }

  async function load() {
    const out = el("rk-output");
    if (!out) return;
    out.innerHTML = `<p class="muted">Cargando snapshot ${escapeHtml(ctx.snapshot)}…</p>`;
    const params = global.TFM.qs({
      run_id: ctx.runId, snapshot_date: ctx.snapshot,
      sort: ctx.sort, order: ctx.order, limit: 2000,
    });
    let data;
    try { data = await api("/api/ranking?" + params); }
    catch (e) { out.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    const rows = (data.rows || []).map((r) => {
      const picked = {};
      COLUMNS.forEach((k) => { picked[k] = r[k]; });
      return picked;
    });
    // La tabla ya trae ordenación por click de cabecera y respeta el orden del backend (Meta desc
    // por defecto). El id permite que el buscador de ticker filtre sus filas.
    const rendered = table(rows, {
      columns: COLUMNS, labels: LABELS, decimals: 4,
      limit: 2000, sortDateDesc: false,
      empty: "Sin acciones para este snapshot.",
    }).replace('<table class="data">', '<table class="data" id="rk-table">');
    out.innerHTML =
      `<h4>Ranking de acciones <small class="muted">snapshot ${escapeHtml(ctx.snapshot)} · ${rows.length} acciones</small></h4>` +
      rendered;
  }

  global.TFM.views.rankings = { render, onSnapshot, onSort };
})(window);
