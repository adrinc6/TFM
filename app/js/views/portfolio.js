/* Cartera: composición (dona) + lifecycle de posiciones. Trades: resumen con P&L realizado de
   ventas + filtros por año/fecha/ticker + tabla de órdenes. Consume /api/portfolio y /api/trades. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct, table } = global.TFM;

  let tradeCtx = { runId: null, years: [] };
  let compositionCtx = { runId: null, dates: [] };

  // Composición actual: posiciones de la última fecha, con el motivo de entrada (de la orden de
  // compra que abrió cada posición) junto a precio de compra, peso, precio actual y P&L latente.
  function compositionTable(rows, selectedDate, isLatest) {
    if (!rows.length) return `<p class="muted">Este run no registró posiciones en la cartera.</p>`;
    const body = rows.map((p) => {
      const pnl = p.pnl_pct;
      const rc = typeof pnl === "number" && Number.isFinite(pnl) ? (pnl >= 0 ? "positive" : "negative") : "";
      return `<tr>
        <td>${escapeHtml(String(p.ticker || ""))}</td>
        <td>${escapeHtml(String(p.entry_date || "—"))}</td>
        <td>${fmt(Number(p.entry_price), 2)}</td>
        <td>${escapeHtml(p.reason || "—")}</td>
        <td>${pct(p.weight, 1)}</td>
        <td>${p.months_held == null ? "—" : `${p.months_held} meses`}</td>
        <td>${p.exit_date || ""}</td>
        <td>${p.final_price == null || Number.isNaN(Number(p.final_price)) ? "—" : fmt(Number(p.final_price), 2)}</td>
        <td class="${rc}">${typeof pnl === "number" && Number.isFinite(pnl) ? `${fmt(pnl, 2)} %` : "—"}</td>
      </tr>`;
    }).join("");
    return `<p class="muted">${isLatest ? "Última fecha disponible" : "Composición histórica"}: ${escapeHtml(selectedDate)}</p>
      <div class="table-wrap"><table class="data"><thead><tr>
        <th class="sortable" data-key="ticker">Ticker</th><th>Fecha de compra</th><th class="sortable" data-key="entry">Precio de compra</th>
        <th>Motivo</th><th class="sortable" data-key="weight">Peso</th>
        <th>Tiempo mantenido</th><th>Fecha de venta</th><th class="sortable" data-key="cur">Precio en fecha</th><th class="sortable" data-key="pnl">P&L</th>
      </tr></thead><tbody>${body}</tbody></table></div>`;
  }

  async function refreshComposition() {
    const selectedDate = el("pf-date")?.value;
    let data;
    try {
      data = await api("/api/portfolio?" + global.TFM.qs({ run_id: compositionCtx.runId, date: selectedDate }));
    } catch (e) {
      el("pf-current").innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
      return;
    }
    el("pf-current").innerHTML = compositionTable(data.rows || [], data.selected_date, data.is_latest);
    global.TFMCharts.clearContainer(el("pf-chart"));
    global.TFMCharts.portfolioComposition(el("pf-chart"), data.rows);
  }

  async function loadHistoricalComposition(date, targetId) {
    const target = el(targetId);
    if (!target || target.dataset.loaded === "true") return;
    target.innerHTML = `<p class="muted">Cargando composición…</p>`;
    try {
      const data = await api("/api/portfolio?" + global.TFM.qs({ run_id: compositionCtx.runId, date }));
      target.innerHTML = compositionTable(data.rows || [], data.selected_date, data.is_latest);
      target.dataset.loaded = "true";
    } catch (e) {
      target.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
    }
  }

  function historicalCompositions(dates) {
    const historicDates = dates.slice(0, -1);
    if (!historicDates.length) return "";
    return `<details><summary>Composiciones históricas (${historicDates.length})</summary>
      <p class="muted">De más reciente a más antigua. Abre una fecha para cargar sus posiciones y su desenlace posterior.</p>
      ${historicDates.slice().reverse().map((date, index) => {
        const targetId = `pf-history-${index}`;
        return `<details ontoggle="if(this.open) TFM.views.portfolio.loadHistoricalComposition('${date}', '${targetId}')"><summary>${date}</summary>
          <div id="${targetId}" data-loaded="false"></div></details>`;
      }).join("")}
    </details>`;
  }

  async function render(container, runId) {
    container.innerHTML = `<section class="parameter-group">
        <h4>Composición de cartera</h4>
        <label class="field">Fecha de composición<select id="pf-date" onchange="TFM.views.portfolio.refreshComposition()"></select></label>
        <div id="pf-current"></div><div id="pf-chart"></div>
        <div id="pf-history"></div>
      </section>`;
    let data;
    try {
      data = await api("/api/portfolio?run_id=" + encodeURIComponent(runId));
    } catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    compositionCtx.runId = runId;
    compositionCtx.dates = data.dates || [];
    const select = el("pf-date");
    select.innerHTML = compositionCtx.dates.slice().reverse().map((date) => `<option value="${date}">${date}</option>`).join("");
    if (data.selected_date) select.value = data.selected_date;
    el("pf-current").innerHTML = compositionTable(data.rows || [], data.selected_date, data.is_latest);
    global.TFMCharts.clearContainer(el("pf-chart"));
    global.TFMCharts.portfolioComposition(el("pf-chart"), data.rows);
    el("pf-history").innerHTML = historicalCompositions(compositionCtx.dates);
  }

  // --- Trades con filtros y P&L realizado ---
  async function renderTrades(container, runId) {
    tradeCtx.runId = runId;
    container.innerHTML = `
      <div class="parameter-group">
        <h4>Filtros</h4>
        <div class="formgrid">
          <label class="field">Año<select id="tr-year" onchange="TFM.views.portfolio.reloadTrades()"><option value="">Todos</option></select></label>
          <label class="field">Desde<input id="tr-start" type="date" onchange="TFM.views.portfolio.reloadTrades()"></label>
          <label class="field">Hasta<input id="tr-end" type="date" onchange="TFM.views.portfolio.reloadTrades()"></label>
          <label class="field">Ticker<input id="tr-ticker" placeholder="AAPL" onkeydown="if(event.key==='Enter')TFM.views.portfolio.reloadTrades()"></label>
        </div>
        <div class="actions"><button class="button" onclick="TFM.views.portfolio.reloadTrades()">Aplicar</button>
          <button class="button ghost" onclick="TFM.views.portfolio.clearTrades()">Limpiar</button></div>
      </div>
      <div id="tr-summary"></div>
      <details><summary>Órdenes <small class="muted">(las ventas muestran su P&L realizado)</small></summary>
        <div id="tr-table"></div></details>`;
    await loadTrades();
  }

  function query() {
    const p = { run_id: tradeCtx.runId };
    const y = el("tr-year"); if (y && y.value) p.year = y.value;
    const s = el("tr-start"); if (s && s.value) p.start = s.value;
    const e = el("tr-end"); if (e && e.value) p.end = e.value;
    const t = el("tr-ticker"); if (t && t.value.trim()) p.ticker = t.value.trim();
    return global.TFM.qs(p);
  }

  async function loadTrades() {
    let data;
    try { data = await api("/api/trades?" + query()); }
    catch (e) { el("tr-summary").innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }

    // Rellena el desplegable de años la primera vez.
    if (data.years && data.years.length && el("tr-year") && el("tr-year").options.length <= 1) {
      el("tr-year").innerHTML = `<option value="">Todos</option>` +
        data.years.map((y) => `<option value="${y}">${y}</option>`).join("");
    }

    const s = data.summary || {};
    const pnlClass = (v) => (typeof v === "number" && v >= 0 ? "positive" : "negative");
    // P&L neto realizado (suma de ganancias y pérdidas por venta, en puntos porcentuales).
    const netPnl = (typeof s.total_gain_pct === "number" || typeof s.total_loss_pct === "number")
      ? (s.total_gain_pct || 0) + (s.total_loss_pct || 0) : null;
    el("tr-summary").innerHTML = `<div class="cards">
      <div class="card"><div class="metric">${s.orders ?? "—"}</div><div class="metric-label">Órdenes</div></div>
      <div class="card"><div class="metric positive">${s.buys ?? "—"}</div><div class="metric-label">Compras</div></div>
      <div class="card"><div class="metric negative">${s.sells ?? "—"}</div><div class="metric-label">Ventas</div></div>
      <div class="card"><div class="metric positive">${s.sells_with_gain ?? "—"}</div><div class="metric-label">Ventas con ganancia</div></div>
      <div class="card"><div class="metric negative">${s.sells_with_loss ?? "—"}</div><div class="metric-label">Ventas con pérdida</div></div>
      <div class="card"><div class="metric ${pnlClass(s.avg_realized_return_pct)}">${s.avg_realized_return_pct == null ? "—" : fmt(s.avg_realized_return_pct, 2) + " %"}</div><div class="metric-label">Retorno medio por venta</div></div>
      <div class="card"><div class="metric ${pnlClass(netPnl)}">${netPnl == null ? "—" : fmt(netPnl, 2) + " %"}</div><div class="metric-label">P&L neto realizado (acumulado)</div></div>
      <div class="card"><div class="metric">${fmt((Number(s.commission) || 0) + (Number(s.slippage) || 0), 2)}</div><div class="metric-label">Costes totales (comisión + slippage)</div></div>
    </div>`;

    // Tabla de órdenes: fecha (más reciente arriba), pesos inicial→final, y en ventas siempre el
    // P&L realizado (la ganancia se materializa al vender). Se ocultan las filas sin efecto real
    // (peso inicial = peso final): no aportan información y no son operaciones.
    const withEffect = (data.rows || []).filter((o) => {
      const before = Number(o.weight_before) || 0, after = Number(o.weight_after) || 0;
      return Math.abs(before - after) > 1e-9;
    });
    const ordersRows = global.TFM.sortByDateDesc(withEffect, "snapshot_date");
    const rows = ordersRows.map((o) => {
      const realized = o.realized_return_pct;
      const isSell = o.side === "sell";
      const rc = typeof realized === "number" ? (realized >= 0 ? "positive" : "negative") : "";
      const pnlCell = isSell
        ? `<span class="${rc}">${typeof realized === "number" ? fmt(realized, 2) + " %" : "—"}</span>`
        : "—";
      return `<tr>
        <td>${escapeHtml(String(o.snapshot_date).slice(0, 10))}</td>
        <td>${escapeHtml(o.ticker)}</td>
        <td>${escapeHtml(o.side)}</td>
        <td>${escapeHtml(o.reason || "")}</td>
        <td>${fmt(Number(o.price), 2)}</td>
        <td>${pct(o.weight_before, 1)} → ${pct(o.weight_after, 1)}</td>
        <td>${pnlCell}</td>
        <td>${fmt((Number(o.commission) || 0) + (Number(o.slippage) || 0), 4)}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="8" class="empty">Sin órdenes para el filtro seleccionado.</td></tr>`;
    el("tr-table").innerHTML = `
      <input class="table-search" data-filter-table="tr-orders" placeholder="Filtrar órdenes (ticker, motivo…)">
      <div class="table-wrap"><table class="data" id="tr-orders"><thead><tr>
        <th class="sortable" data-key="date">Fecha</th><th class="sortable" data-key="ticker">Ticker</th>
        <th class="sortable" data-key="side">Operación</th><th>Motivo</th>
        <th class="sortable" data-key="price">Precio</th><th>Peso antes → después</th>
        <th class="sortable" data-key="pnl">P&L venta</th><th class="sortable" data-key="cost">Costes</th>
      </tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  global.TFM.views.portfolio = {
    render, renderTrades,
    refreshComposition,
    loadHistoricalComposition,
    reloadTrades: loadTrades,
    clearTrades() {
      ["tr-year", "tr-start", "tr-end", "tr-ticker"].forEach((id) => { if (el(id)) el(id).value = ""; });
      loadTrades();
    },
  };
})(window);
