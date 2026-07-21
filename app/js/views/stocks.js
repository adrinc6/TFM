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
    ["puntuaciones", "Puntuaciones Agentes"],
    ["ratios", "Estudio de ratios"],
  ];

  // "Resumen" es el primer valor y el que carga por defecto al entrar en modo ratios. Fuera de ese
  // modo el selector se deshabilita y se vacía; al volver a ratios se restablece a "Resumen".
  function metricOptions(groups) {
    const resumen = `<option value="__summary__">Resumen</option>`;
    const rest = Object.entries(groups || {})
      .map(([group, items]) =>
        `<optgroup label="${escapeHtml(group)}">${items.map((it) => `<option value="${escapeHtml(it[0])}">${escapeHtml(it[1])}</option>`).join("")}</optgroup>`
      )
      .join("");
    return resumen + rest;
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
    ctx.fullStart = meta.full_start || meta.oos_start || "";
    ctx.fullEnd = meta.full_end || meta.oos_end || "";
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
          <!-- Tercer control: cambia según el modo. En Cartera queda vacío (se conserva el hueco);
               en Puntuaciones es el select de snapshot; en Ratios es el select de Ratio. -->
          <label class="field" id="stk-third-field">
            <span id="stk-third-label">Ratio</span>
            <select id="stk-metric" disabled onchange="TFM.views.stocks.setMetric()">${metricOptions(meta.metrics)}</select>
            <select id="stk-snapshot" style="display:none" onchange="TFM.views.stocks.onSnapshot()"></select>
          </label>
        </div>
        <div id="stk-range-row" class="formgrid" style="display:none">
          <label class="field" id="stk-start-field">Inicio<input id="stk-start" type="date"
            min="${escapeHtml(ctx.fullStart)}" max="${escapeHtml(ctx.fullEnd)}" value="${escapeHtml(ctx.oosStart)}"></label>
          <label class="field">Fin<input id="stk-end" type="date" onchange="TFM.views.stocks.onEndDate()"
            min="${escapeHtml(ctx.fullStart)}" max="${escapeHtml(ctx.fullEnd)}" value="${escapeHtml(ctx.oosEnd)}"></label>
          <label class="field" id="stk-recalc-field" style="align-self:end">
            <span>&nbsp;</span><button class="button" onclick="TFM.views.stocks.load()">Recalcular</button></label>
        </div>
      </section>
      <div id="stk-output" class="muted">Busca y selecciona una acción.</div>`;
    syncThirdControl();   // el modo inicial es cartera: oculta el select de Ratio (deja el hueco)
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

  // Ajusta el tercer control de la fila superior según el modo:
  //  - cartera: nada (hueco vacío, la celda se conserva);
  //  - puntuaciones: select de snapshot ("Puntuaciones Agentes");
  //  - ratios: select de Ratio (Resumen + ratios).
  function syncThirdControl() {
    const label = el("stk-third-label"), metric = el("stk-metric"), snap = el("stk-snapshot");
    if (!metric || !snap) return;
    const isRatios = ctx.mode === "ratios";
    const isPunt = ctx.mode === "puntuaciones";
    metric.style.display = isRatios ? "" : "none";
    metric.disabled = !isRatios;
    metric.value = isRatios ? "__summary__" : "";
    snap.style.display = isPunt ? "" : "none";
    if (label) label.textContent = isRatios ? "Ratio" : (isPunt ? "Puntuaciones Agentes" : " ");
  }

  function setMode() {
    ctx.mode = el("stk-mode").value;
    syncThirdControl();
    // La fila de fechas (Inicio/Fin/Recalcular) es SOLO del estudio de ratios. En puntuaciones el
    // snapshot lo da el select de arriba; en cartera no hay fechas.
    const rangeRow = el("stk-range-row");
    if (rangeRow) rangeRow.style.display = ctx.mode === "ratios" ? "" : "none";
    syncRangeFields();
    // En ratios se carga ya (por defecto, el Resumen); en el resto, también.
    if (ctx.ticker) load();
  }

  // Al elegir un ratio (incluido "Resumen") se recolocan los campos y se recalcula al instante.
  function setMetric() {
    syncRangeFields();
    if (ctx.ticker && el("stk-metric").value) load();
  }

  // Cambiar la fecha Fin recalcula al instante en el Resumen de ratios (Fin = el snapshot).
  function onEndDate() {
    if (!ctx.ticker) return;
    if (el("stk-metric").value === "__summary__") load();
  }

  // Cambiar el snapshot en modo Puntuaciones recarga las puntuaciones y porqués de esa fecha.
  function onSnapshot() {
    if (ctx.ticker && ctx.mode === "puntuaciones") load();
  }

  // Marca un campo de fecha como bloqueado (readonly + gris) o editable.
  function lockField(input, locked) {
    if (!input) return;
    input.readOnly = locked;
    input.style.opacity = locked ? "0.5" : "";
    input.style.cursor = locked ? "not-allowed" : "";
  }

  // Campos de fecha en modo ratios (sin selector de rango): Inicio y Fin con defaults del OOS.
  //  - Resumen: Inicio BLOQUEADO (solo cuenta Fin, que es el snapshot); Fin editable.
  //  - Ratio concreto: Inicio y Fin editables (definen el rango del gráfico).
  // En ambos casos el botón Recalcular aplica los cambios.
  function syncRangeFields() {
    const start = el("stk-start"), end = el("stk-end");
    if (!start || !end) return;
    if (!start.value) start.value = ctx.oosStart || ctx.fullStart || "";
    if (!end.value) end.value = ctx.oosEnd || ctx.fullEnd || "";
    const isSummary = el("stk-metric") && el("stk-metric").value === "__summary__";
    lockField(start, isSummary);   // Inicio bloqueado solo en el Resumen de ratios
    lockField(end, false);         // Fin siempre editable
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
    // Los 5 agentes (los que haya en el score) + Meta. Cada tarjeta solo si el valor existe.
    const cards = [
      ["Calidad", s.quality, s.quality_rank],
      ["Valor", s.value, s.value_rank],
      ["Crecimiento", s.growth, s.growth_rank],
      ["Momentum", s.momentum, s.momentum_rank],
      ["Riesgo", s.risk, s.risk_rank],
    ].filter(([, value]) => typeof value === "number" && Number.isFinite(value))
      .map(([label, value, rank]) => row(label, value, rank));
    // Meta va siempre al final (es la combinación, no un agente).
    cards.push(row("Meta", s.meta_score, s.meta_rank));
    return `<div class="cards score-cards">${cards.join("")}</div>`;
  }

  function agentExplain(rows) {
    const pos = rows.filter((x) => x.local_contribution > 0).slice(0, 2).map((x) => x.feature.replace("factor_", "")).join(", ");
    const neg = rows.filter((x) => x.local_contribution < 0).slice(0, 2).map((x) => x.feature.replace("factor_", "")).join(", ");
    return `El agente favorece por ${pos || "ninguna contribución positiva destacada"}${neg ? `; penaliza por ${neg}` : ""}.`;
  }

  // Nombres legibles de los agentes (mismo orden que las tarjetas de puntuación).
  const AGENT_LABELS = { quality: "Calidad", value: "Valor", growth: "Crecimiento",
                         momentum: "Momentum", risk: "Riesgo" };

  // Tabla-resumen MUY compacta de los pesos del meta en esta fecha: qué peso da a cada agente y por
  // qué (su rank-IC reciente). Es lo primero que se ve; el detalle por agente va debajo, plegado.
  function metaWeightsSummary(weights) {
    const rows = (weights || []).filter((w) => w.agent in AGENT_LABELS)
      .sort((x, y) => (y.weight || 0) - (x.weight || 0));
    if (!rows.length) return `<p class="muted">Sin pesos del meta-agente para esta fecha.</p>`;
    const body = rows.map((w) => {
      const status = w.weight_status === "fallback_equal" ? "equiponderado (sin señal)" : "aprendido";
      return `<tr><td>${escapeHtml(AGENT_LABELS[w.agent] || w.agent)}</td>
        <td>${pct(w.weight, 1)}</td>
        <td>${w.mean_rank_ic == null ? "—" : fmt(w.mean_rank_ic, 3)}</td>
        <td class="muted">${status}</td></tr>`;
    }).join("");
    return `<div class="table-wrap"><table class="data"><thead><tr>
      <th>Agente</th><th>Peso meta</th><th>Rank-IC reciente</th><th>Estado</th>
      </tr></thead><tbody>${body}</tbody></table></div>`;
  }

  // Una tarjeta PLEGADA por agente, a lo ANCHO (una sobre otra). En la cabecera, las cifras clave del
  // agente para esta acción/fecha (puntuación, percentil y peso en el meta); dentro, la explicación y
  // las 5 variables de MAYOR IMPACTO (positivo o negativo) en la puntuación de ESTA acción y fecha
  // (contribuciones locales de LightGBM, no la importancia global del modelo).
  function agentDetailCard(agent, rows, weight, score) {
    const label = AGENT_LABELS[agent] || agent;
    const chips = [];
    if (score && typeof score.value === "number") chips.push(`puntuación ${fmt(score.value, 3)}`);
    if (score && typeof score.rank === "number") chips.push(`percentil ${pct(score.rank, 1)}`);
    if (weight != null) chips.push(`peso meta ${pct(weight, 1)}`);
    const head = chips.length ? ` <small class="muted">· ${chips.join(" · ")}</small>` : "";
    // Top-5 por impacto local (ya vienen así del pipeline), ordenadas por rango de importancia.
    const ordered = rows.slice().sort((x, y) => (x.importance_rank || 1e9) - (y.importance_rank || 1e9));
    const t = table(ordered.map((r) => ({
      "#": r.importance_rank,
      variable: r.feature.replace("factor_", ""),
      valor: r.factor_value,
      contribución: r.local_contribution,
      dirección: r.direction === "positive" ? "favorece" : (r.direction === "negative" ? "penaliza" : r.direction),
    })), { decimals: 4, sortable: true, sortDateDesc: false });
    return `<details class="card agent-card"><summary><strong>${escapeHtml(label)}</strong>${head}</summary>
      <p>${escapeHtml(agentExplain(rows))}</p>
      <p class="muted">Las 5 variables de mayor impacto (positivo o negativo) en la puntuación de esta acción en este snapshot.</p>${t}</details>`;
  }

  function agentsBlock(a) {
    const groups = {};
    (a.contributions || []).forEach((c) => (groups[c.agent] ??= []).push(c));
    const weightByAgent = {};
    (a.weights || []).forEach((w) => (weightByAgent[w.agent] = w.weight));
    // Puntuación y percentil de cada agente para esta acción/fecha (de la fila de scores seleccionada).
    const s = (a.selected_scores && a.selected_scores[0]) || {};
    const scoreOf = (agent) => ({ value: s[agent], rank: s[`${agent}_rank`] });
    // Orden estable de agentes (el de AGENT_LABELS), solo los que tienen contribuciones.
    const order = Object.keys(AGENT_LABELS).filter((agent) => groups[agent]);
    const cards = order.map((agent) =>
      agentDetailCard(agent, groups[agent], weightByAgent[agent], scoreOf(agent))).join("")
      || `<p class="muted">Sin contribuciones locales para este run/fecha.</p>`;
    return `<details><summary>Por qué puntúan los agentes</summary>
      <h4>Resumen: pesos del meta-agente en esta fecha</h4>
      ${metaWeightsSummary(a.weights)}
      <h4>Detalle por agente</h4>
      <div class="agent-cards">${cards}</div>
      <details><summary>Importancia global del modelo de esta fecha</summary>${table(a.global_importance || [], { decimals: 4 })}</details></details>`;
  }

  // Puebla el select de snapshot con las fechas disponibles del ticker, conservando la selección si
  // sigue existiendo; si no, deja la última (la más reciente).
  function fillSnapshotSelect(dates) {
    const snap = el("stk-snapshot");
    if (!snap) return "";
    const list = dates || [];
    const previous = snap.value;
    snap.innerHTML = list.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("");
    const chosen = list.includes(previous) ? previous : (list.length ? list[list.length - 1] : "");
    snap.value = chosen;
    return chosen;
  }

  async function loadPuntuaciones(out) {
    // El snapshot lo fija el select de arriba (solo contiene fechas válidas). El endpoint de agentes
    // exige la fecha EXACTA; al venir del select nunca es inválida, así que no hay IndexError.
    const chosen = (el("stk-snapshot") && el("stk-snapshot").value) || "";
    const snapParam = chosen ? `&snapshot_date=${encodeURIComponent(chosen)}` : "";
    const [agents, ticker_data] = await Promise.all([
      api(`/api/stock/agents?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}${snapParam}`),
      api(`/api/ticker?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
    ]);
    // Poblar el select con los snapshots del ticker (primera carga o cambio de ticker) y quedarnos
    // con el snapshot efectivamente mostrado.
    const shown = fillSnapshotSelect(agents.snapshot_dates) || agents.snapshot_date || "";
    // Puntuaciones de ese snapshot: la fila seleccionada del propio endpoint de agentes (evita una
    // segunda llamada y garantiza que corresponden a la MISMA fecha que los porqués).
    const scoreRow = (agents.selected_scores && agents.selected_scores[0])
      || (agents.scores && agents.scores.find((r) => String(r.snapshot_date).slice(0, 10) === shown))
      || {};
    if (!agents.scores || !agents.scores.length) {
      out.innerHTML = `<div class="notice">Sin datos para ${escapeHtml(ctx.ticker)} en este run.</div>`; return;
    }
    out.innerHTML =
      `<h4>Puntuaciones de ${escapeHtml(ctx.ticker)} <small class="muted">snapshot ${escapeHtml(shown)}</small></h4>` +
      scoreCards(scoreRow) +
      `<details><summary>Histórico de puntuaciones <small class="muted">(línea verde = compra, roja = venta)</small></summary><div id="stk-scores"></div></details>` +
      `<div id="stk-agents"></div>`;
    global.TFMCharts.clear();
    global.TFMCharts.stockScoreHistory(el("stk-scores"), agents.scores, ticker_data.orders);
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
    if (!metric) { out.innerHTML = `<p class="muted">Elige un ratio para analizar ${escapeHtml(ctx.ticker)}.</p>`; return; }
    // Resumen: una sola fecha (Fin). Muestra la tabla de ratios de ese snapshot, sin gráfico.
    if (metric === "__summary__") {
      const date = el("stk-end").value;
      const summary = await api(`/api/stock/summary?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}&date=${date}`);
      if (summary.found === false) { out.innerHTML = `<div class="notice">Sin datos para ${escapeHtml(ctx.ticker)} en este run.</div>`; return; }
      out.innerHTML =
        `<h4>Resumen de ratios de ${escapeHtml(ctx.ticker)} <small class="muted">snapshot ${escapeHtml(summary.snapshot_date || date)}</small></h4>` +
        ratiosTable(summary);
      global.TFMCharts.clear();
      return;
    }
    // Ratio concreto: SOLO el gráfico de la historia del ratio (con líneas de compra/venta). No se
    // muestra la tabla de percentiles. Al ser lo único, se presenta directo, sin desplegable.
    const start = el("stk-start").value;
    const end = el("stk-end").value;
    const [history, ticker_data] = await Promise.all([
      api(`/api/stock/history?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}&metric=${encodeURIComponent(metric)}&start=${start}&end=${end}`),
      api(`/api/ticker?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ctx.ticker)}`),
    ]);
    out.innerHTML =
      `<h4>Historia de ${escapeHtml(history.metric_label || metric)} · ${escapeHtml(ctx.ticker)}
        <small class="muted">(línea verde = compra, roja = venta)</small></h4><div id="stk-chart"></div>`;
    global.TFMCharts.clear();
    global.TFMCharts.stockHistory(el("stk-chart"), history, ticker_data.orders);
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

  global.TFM.views.stocks = { render, search, selectTicker, setMode, setMetric, onEndDate, onSnapshot, load };
})(window);
