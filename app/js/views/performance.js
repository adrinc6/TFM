/* Rendimiento: resumen (cards + equity + alfa) y detalle anual (tablas + gráficos).
   Consume /api/performance (equity + annual). */
(function (global) {
  "use strict";
  const { api, escapeHtml, fmt, table } = global.TFM;

  const ANNUAL_LABELS = {
    year: "Año", portfolio_return: "Cartera", benchmark_return: "Benchmark", alpha: "Alfa",
    beats_benchmark: "Bate SPY", max_drawdown_year: "Max DD", information_ratio_year: "IR",
  };

  // Pestaña Resumen del run: cards del summary + curva + alfa + configuración efectiva (solo aquí).
  async function renderSummary(container, runId, manifest) {
    container.innerHTML = `<details><summary>Gráficos de rentabilidad</summary><div id="perf-charts"></div></details>
      <details><summary>Rentabilidad anual</summary><div id="perf-annual"></div></details>
      <details><summary>Configuración efectiva</summary>${global.TFM.views.run.settingsCards(manifest.effective_settings)}</details>`;
    let data;
    try { data = await api("/api/performance?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    const charts = document.getElementById("perf-charts");
    global.TFMCharts.equityChart(charts, data.equity);
    global.TFMCharts.alphaBars(charts, data.annual);
    document.getElementById("perf-annual").innerHTML = table(data.annual, { columns: Object.keys(ANNUAL_LABELS), labels: ANNUAL_LABELS, decimals: 4 });
  }

  // Pestaña Rendimiento completa
  async function render(container, runId) {
    container.innerHTML = `<details><summary>Gráficos de rentabilidad</summary><div id="perf-charts"></div></details>
      <details><summary>Rentabilidad anual</summary><div id="perf-annual"></div></details>
      <details><summary>Equity por snapshot</summary><div id="perf-equity"></div></details>`;
    let data;
    try { data = await api("/api/performance?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    const charts = document.getElementById("perf-charts");
    global.TFMCharts.equityChart(charts, data.equity);
    global.TFMCharts.alphaBars(charts, data.annual);
    document.getElementById("perf-annual").innerHTML = table(data.annual, { columns: Object.keys(ANNUAL_LABELS), labels: ANNUAL_LABELS, decimals: 4 });
    document.getElementById("perf-equity").innerHTML = table(data.equity, { limit: 300, decimals: 4 });
  }

  global.TFM.views.performance = { render, renderSummary };
})(window);
