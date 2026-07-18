/* Detalle de un run: métricas cabecera + pestañas Resumen / Rendimiento / Aprendizaje /
   Cartera / Trades / Stocks / Ticker. Cada pestaña delega en su vista de análisis. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt } = global.TFM;

  let current = null; // { runId, manifest }

  const TABS = [
    ["resumen", "Resumen"],
    ["rendimiento", "Rendimiento"],
    ["aprendizaje", "Aprendizaje"],
    ["cartera", "Cartera"],
    ["trades", "Trades"],
    ["stocks", "Stocks"],
    ["ticker", "Ticker"],
  ];

  function metricCard(value, label, decimals = 4, signed = false) {
    const cls = signed && typeof value === "number" ? (value >= 0 ? " pos" : " neg") : "";
    return `<div class="card"><div class="metric${cls}">${fmt(value, decimals)}</div><div class="metric-label">${escapeHtml(label)}</div></div>`;
  }

  async function open(runId, container) {
    global.TFMCharts.clear();
    container.innerHTML = `<p class="muted">Cargando run…</p>`;
    let manifest;
    try {
      manifest = await api("/api/run/" + encodeURIComponent(runId));
    } catch (e) {
      container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
      return;
    }
    current = { runId, manifest };
    const s = manifest.summary || {};
    const intent = manifest.intent || {};
    container.innerHTML = `
      <h3>${escapeHtml(intent.label || runId)}</h3>
      <p class="muted mono">${escapeHtml(runId)} · <span class="tag">${escapeHtml(manifest.run_kind || "")}</span></p>
      <div class="cards">
        ${metricCard(s.mean_rank_ic, "rank-IC medio (OOS)")}
        ${metricCard(s.cagr_difference, "CAGR vs benchmark", 4, true)}
        ${metricCard(s.max_drawdown, "Max drawdown", 4, true)}
        ${metricCard(s.beat_rate, "% años que baten SPY", 3)}
      </div>
      ${intent.description ? `<p class="muted" style="margin-top:12px">${escapeHtml(intent.description)}</p>` : ""}
      <div class="tabs" id="run-tabs">${TABS.map(([id, label], i) => `<button data-tab="${id}" class="${i === 0 ? "active" : ""}" onclick="TFM.views.run.tab('${id}')">${label}</button>`).join("")}</div>
      <div id="run-tab-body"></div>
      <details style="margin-top:16px"><summary>Configuración efectiva</summary><pre>${escapeHtml(JSON.stringify(manifest.effective_settings || {}, null, 2))}</pre></details>`;
    tab("resumen");
  }

  async function tab(name) {
    document.querySelectorAll("#run-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    const body = el("run-tab-body");
    global.TFMCharts.clear();
    const runId = current.runId;
    const views = global.TFM.views;
    if (name === "resumen") return views.performance.renderSummary(body, runId, current.manifest);
    if (name === "rendimiento") return views.performance.render(body, runId);
    if (name === "aprendizaje") return views.learning.render(body, runId);
    if (name === "cartera") return views.portfolio.render(body, runId);
    if (name === "trades") return views.portfolio.renderTrades(body, runId);
    if (name === "stocks") return views.stocks.render(body, runId);
    if (name === "ticker") return views.ticker.render(body, runId);
  }

  global.TFM.views.run = { open, tab };
})(window);
