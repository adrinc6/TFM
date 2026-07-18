/* Cartera: composición (dona) + lifecycle de posiciones (tabla). Trades: resumen + órdenes.
   Consume /api/portfolio (position_lifecycle) y /api/trades. */
(function (global) {
  "use strict";
  const { api, escapeHtml, fmt, table } = global.TFM;

  async function render(container, runId) {
    container.innerHTML = `<div id="pf-chart"></div><h4>Posiciones y valoración</h4><div id="pf-table"></div>`;
    let data;
    try { data = await api("/api/portfolio?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    global.TFMCharts.portfolioComposition(document.getElementById("pf-chart"), data.rows);
    document.getElementById("pf-table").innerHTML = table(data.rows, { limit: 300, decimals: 4 });
  }

  async function renderTrades(container, runId) {
    container.innerHTML = `<div id="tr-summary"></div><h4>Órdenes</h4><div id="tr-table"></div>`;
    let data;
    try { data = await api("/api/trades?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    const s = data.summary || {};
    document.getElementById("tr-summary").innerHTML = `<div class="cards">
      <div class="card"><div class="metric">${s.orders ?? "—"}</div><div class="metric-label">Órdenes</div></div>
      <div class="card"><div class="metric positive">${s.buys ?? "—"}</div><div class="metric-label">Compras</div></div>
      <div class="card"><div class="metric negative">${s.sells ?? "—"}</div><div class="metric-label">Ventas</div></div>
      <div class="card"><div class="metric">${fmt(s.commission, 4)}</div><div class="metric-label">Comisión total</div></div>
      <div class="card"><div class="metric">${fmt(s.slippage, 4)}</div><div class="metric-label">Slippage total</div></div>
    </div>`;
    document.getElementById("tr-table").innerHTML = table(data.rows, { limit: 500, decimals: 5 });
  }

  global.TFM.views.portfolio = { render, renderTrades };
})(window);
