/* Explorador de stocks: buscador de ticker + tres modos de análisis de la acción seleccionada:
   - Cartera: si está o estuvo en cartera, cuándo se compró/vendió y su P&L.
   - Puntuaciones: scores y percentiles actuales de los agentes, histórico en gráfico y
     explicabilidad de agentes.
   - Estudio de ratios: desbloquea el selector de ratio y muestra la tabla de ratios con
     percentil y la historia del ratio elegido frente a la media S&P 500.
   Consume /api/stocks, /api/stock/summary, /api/stock/history, /api/stock/agents, /api/ticker. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct, table } = global.TFM;

  // { runId, oosStart, oosEnd, ticker, mode, metrics }
  const ctx = { mode: "cartera" };

  const MODES = [
    ["cartera", "Cartera"],
    ["puntuaciones", "Puntuaciones"],
    ["ratios", "Estudio de ratios"],
  ];

  function metricOptions(groups) {
    return Object.entries(groups || {})
      .map(([group, items]) =>
        `<optgroup label="${escapeHtml(group)}">${items.map((it) => `<option value="${escapeHtml(it[0])}">${escapeHtml(it[1])}</option>`).join("")}</optgroup>`
      )
      .join("");
  }

  async function render(container, runId) {
    ctx.runId = runId;
    ctx.ticker = "";
    ctx.mode = "cartera";
    container.innerHTML = `<p class="muted">Cargando explorador…</p>`;
    let meta;
    try { meta = await api("/api/stocks?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    if (!meta.compatible) {
      container.innerHTML = `<div class="notice">${escapeHtml(meta.message || "Este run no conserva el panel de stocks.")}</div>`;
      return;
    }
    ctx.oosStart = meta.oos_start || "";
    ctx.oosEnd = meta.oos_end || "";
    ctx.metrics = meta.metrics;
    container.innerHTML = `
      <section class="parameter-group"><h4>Explorador de stocks</h4>
        <p class="muted">Datos point-in-time inmutables de este run. Elige una acción y un modo de análisis.</p>
        <div class="formgrid">
          <label class="field">Ticker<input id="stk-ticker" list="stk-list" placeholder="AAPL" oninput="TFM.views.stocks.search()" onchange="TFM.views.stocks.selectTicker()"></label>
          <datalist id="stk-list"></datalist>
          <label class="field">Modo<select id="stk-mode" onchange="TFM.views.stocks.setMode()">
            ${MODES.map(([id, label]) => `<option value="${id}">${label}</option>`).join("")}
          </select></label>
          <label class="field">Ratio<select id="stk-metric" disabled onchange="TFM.views.stocks.load()">${metricOptions(meta.metrics)}</select></label>
        </div>
        <div id="stk-range-row" class="formgrid" style="display:none">
          <label class="field">Rango<select id="stk-range" onchange="TFM.views.stocks.setRange()">
            <option value="oos">Periodo OOS del run</option>
            <option value="full">Histórico completo del run</option>
            <option value="custom">Rango manual</option>
          </select></label>
          <label class="field">Inicio<input id="stk-start" type="date" value="${escapeHtml(ctx.oosStart)}"></label>
          <label class="field">Fin<input id="stk-end" type="date" value="${escapeHtml(ctx.oosEnd)}"></label>
        </div>
      </section>
      <div id="stk-output" class="muted">Busca y selecciona una acción.</div>`;
    search();
  }

  async function search() {
    const q = el("stk-ticker").value.trim();
    try {
      const data = await api("/api/stocks?run_id=" + encodeURIComponent(ctx.runId) + "&query=" + encodeURIComponent(q));
      el("stk-list").innerHTML = (data.tickers || []).map((t) => `<option value="${escapeHtml(t)}"></option>`).join("");
    } catch (e) { /* silencioso */ }
  }

  // Al confirmar un ticker (Enter/selección del datalist) se carga el modo activo.
  function selectTicker() {
    const t = el("stk-ticker").value.trim().toUpperCase();
    if (!t) return;
    ctx.ticker = t;
    load();
  }

  function setMode() {
    ctx.mode = el("stk-mode").value;
    // El selector de ratio solo se activa en el modo de estudio de ratios.
    const metric = el("stk-metric");
    if (metric) metric.disabled = ctx.mode !== "ratios";
    // El rango de fechas solo aplica al estudio de ratios (historia del ratio).
    const rangeRow = el("stk-range-row");
    if (rangeRow) rangeRow.style.display = ctx.mode === "ratios" ? "" : "none";
    if (ctx.ticker) load();
  }

  function setRange() {
    const mode = el("stk-range").value;
    if (mode === "oos") { el("stk-start").value = ctx.oosStart || ""; el("stk-end").value = ctx.oosEnd || ""; }
    if (mode === "full") { el("stk-start").value = ""; el("stk-end").value = ""; }
    load();
  }

  // --- Modo Cartera: si la acción está/estuvo en cartera, cuándo entró/salió y su P&L ---
  function positionCard(pos) {
    if (!pos || pos.weight == null) {
      return `<div class="notice">Esta acción no está actualmente en cartera en la última fecha del run.</div>`;
    }
    return `<div class="cards">
      <div class="card"><div class="metric">${pct(pos.weight, 2)}</div><div class="metric-label">Peso actual</div></div>
      <div class="card"><div class="metric">${fmt(pos.entry_price, 2)}</div><div class="metric-label">Precio de entrada</div></div>
      <div class="card"><div class="metric">${fmt(pos.valuation_price, 2)}</div><div class="metric-label">Precio de valoración</div></div>
      <div class="card"><div class="metric ${(pos.unrealized_return_pct || 0) >= 0 ? "pos" : "neg"}">${fmt(pos.unrealized_return_pct, 2)} %</div><div class="metric-label">Rentabilidad latente</div></div>
    </div>`;
  }

  function ordersTable(orders) {
    const rows = global.TFM.sortByDateDesc(orders || [], "snapshot_date").map((o) =>
      `<tr><td>${escapeHtml(String(o.snapshot_date).slice(0, 10))}</td>
        <td>${escapeHtml(o.side)} · ${escapeHtml(o.reason || "")}</td>
        <td>${fmt(Number(o.price), 2)}</td>
        <td>${pct(o.weight_before, 1)} → ${pct(o.weight_after, 1)}</td>
        <td class="${typeof o.realized_return_pct === "number" ? (o.realized_return_pct >= 0 ? "positive" : "negative") : ""}">${typeof o.realized_return_pct === "number" ? fmt(o.realized_return_pct, 2) + " %" : "—"}</td>
        <td>${fmt((Number(o.commission) || 0) + (Number(o.slippage) || 0), 4)}</td></tr>`
    ).join("") || `<tr><td colspan="6" class="muted">Esta acción nunca ha estado en la cartera de este run.</td></tr>`;
    return `<div class="table-wrap"><table class="data"><thead><tr>
        <th>Fecha</th><th>Operación</th><th>Precio</th><th>Peso antes → después</th><th>P&L venta</th><th>Costes</th>
      </tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  async function loadCartera(out) {
    const [summary, ticker_data] = await Promise.all([
      api(`/api/stock/summary?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
      api(`/api/ticker?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
    ]);
    if (summary.found === false) { out.innerHTML = `<div class="notice">Sin datos para ${escapeHtml(ctx.ticker)} en este run.</div>`; return; }
    out.innerHTML =
      `<h4>Situación en cartera de ${escapeHtml(ctx.ticker)}</h4>` + positionCard(summary.position) +
      `<details><summary>Precio point-in-time y operaciones</summary><div id="stk-price"></div></details>` +
      `<details><summary>Historial de operaciones (compras, ventas y P&L)</summary>${ordersTable(ticker_data.orders)}</details>`;
    global.TFMCharts.clear();
    global.TFMCharts.tickerPrice(el("stk-price"), ticker_data.prices, ticker_data.orders);
  }

  // --- Modo Puntuaciones: scores/percentiles actuales + histórico + explicabilidad ---
  function scoreCards(score) {
    const s = score || {};
    const row = (label, value, rank) =>
      `<div class="card"><div class="metric">${fmt(value, 3)}</div>
        <div class="metric-label">${label} · percentil ${rank == null ? "—" : pct(rank, 1)}</div></div>`;
    return `<div class="cards">
      ${row("Calidad", s.quality, s.quality_rank)}
      ${row("Momentum", s.momentum, s.momentum_rank)}
      ${row("Valor", s.value, s.value_rank)}
      ${row("Meta", s.meta_score, s.meta_rank)}
    </div>`;
  }

  function agentExplain(rows) {
    const pos = rows.filter((x) => x.local_contribution > 0).slice(0, 2).map((x) => x.feature.replace("factor_", "")).join(", ");
    const neg = rows.filter((x) => x.local_contribution < 0).slice(0, 2).map((x) => x.feature.replace("factor_", "")).join(", ");
    return `El agente favorece por ${pos || "ninguna contribución positiva destacada"}${neg ? `; penaliza por ${neg}` : ""}.`;
  }

  function agentsBlock(a) {
    const groups = {};
    (a.contributions || []).forEach((c) => (groups[c.agent] ??= []).push(c));
    const cards = Object.entries(groups).map(([agent, rows]) => {
      const t = table(rows.map((r) => ({
        variable: r.feature.replace("factor_", ""),
        valor_factor: r.factor_value,
        contribucion: r.local_contribution,
        direccion: r.direction,
      })), { decimals: 4 });
      return `<article class="card"><h3>${escapeHtml(agent)}</h3><p>${escapeHtml(agentExplain(rows))}</p>${t}</article>`;
    }).join("") || `<p class="muted">Sin contribuciones locales para este run/fecha.</p>`;
    return `<details><summary>Por qué puntúan los agentes</summary><div class="cards">${cards}</div>
      <h4>Pesos del meta-agente</h4>${table(a.weights || [], { decimals: 4 })}
      <details><summary>Importancia global del modelo de esta fecha</summary>${table(a.global_importance || [], { decimals: 4 })}</details></details>`;
  }

  async function loadPuntuaciones(out) {
    const [summary, agents] = await Promise.all([
      api(`/api/stock/summary?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
      api(`/api/stock/agents?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
    ]);
    if (summary.found === false) { out.innerHTML = `<div class="notice">Sin datos para ${escapeHtml(ctx.ticker)} en este run.</div>`; return; }
    out.innerHTML =
      `<h4>Puntuaciones actuales de ${escapeHtml(ctx.ticker)} <small class="muted">snapshot ${escapeHtml(summary.snapshot_date)}</small></h4>` +
      scoreCards(summary.scores) +
      `<details><summary>Histórico de puntuaciones</summary><div id="stk-scores"></div></details>` +
      `<div id="stk-agents"></div>`;
    global.TFMCharts.clear();
    global.TFMCharts.stockScoreHistory(el("stk-scores"), agents.scores);
    el("stk-agents").innerHTML = agentsBlock(agents);
  }

  // --- Modo Estudio de ratios: tabla de ratios con percentil + historia del ratio elegido ---
  function ratiosTable(summary) {
    const rows = (summary.ratios || [])
      .map((r) => `<tr><td>${escapeHtml(r.label)}</td><td>${fmt(r.value, 3)}</td><td>${r.percentile == null ? "—" : pct(r.percentile, 1)}</td><td>${r.observations}</td></tr>`)
      .join("");
    return `<div class="table-wrap"><table class="data"><thead><tr><th>Ratio</th><th>Valor</th><th>Percentil</th><th>N</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  async function loadRatios(out) {
    const metric = el("stk-metric").value;
    const start = el("stk-start").value;
    const end = el("stk-end").value;
    const [summary, history] = await Promise.all([
      api(`/api/stock/summary?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
      api(`/api/stock/history?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}&metric=${encodeURIComponent(metric)}&start=${start}&end=${end}`),
    ]);
    if (summary.found === false) { out.innerHTML = `<div class="notice">Sin datos para ${escapeHtml(ctx.ticker)} en este run.</div>`; return; }
    out.innerHTML =
      `<details><summary>Ratios de ${escapeHtml(ctx.ticker)} · percentil en el universo del snapshot</summary>${ratiosTable(summary)}</details>` +
      `<details><summary>Historia de ${escapeHtml(history.metric_label || metric)}</summary><div id="stk-chart"></div></details>`;
    global.TFMCharts.clear();
    global.TFMCharts.stockHistory(el("stk-chart"), history);
  }

  async function load() {
    if (!ctx.ticker) return;
    const out = el("stk-output");
    out.innerHTML = `<p class="muted">Analizando ${escapeHtml(ctx.ticker)}…</p>`;
    try {
      if (ctx.mode === "cartera") return await loadCartera(out);
      if (ctx.mode === "puntuaciones") return await loadPuntuaciones(out);
      return await loadRatios(out);
    } catch (e) {
      out.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
    }
  }

  global.TFM.views.stocks = { render, search, selectTicker, setMode, setRange, load };
})(window);
