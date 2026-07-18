/* Aprendizaje: rank-IC por agente (resumen) + serie por cohorte, con gráfico de rank-IC en el
   tiempo y evolución de pesos meta. Consume /api/learning y /api/performance (para pesos, que
   viven en meta_weights.parquet vía /api/stock/agents no aplica aquí: se piden aparte). */
(function (global) {
  "use strict";
  const { api, escapeHtml, table } = global.TFM;

  const SUMMARY_LABELS = {
    agent: "Agente", mean_rank_ic: "rank-IC medio", positive_fraction: "% cohortes IC>0",
    rank_ic_std: "Desv. típica", n_cohorts: "Cohortes",
  };

  async function render(container, runId) {
    container.innerHTML = `
      <div id="learn-chart"></div>
      <h4>Aprendizaje por agente</h4><div id="learn-summary"></div>
      <div id="learn-weights"></div>`;
    let data;
    try { data = await api("/api/learning?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    global.TFMCharts.rankIcOverTime(document.getElementById("learn-chart"), data.series);
    document.getElementById("learn-summary").innerHTML =
      table(data.summary, { columns: Object.keys(SUMMARY_LABELS), labels: SUMMARY_LABELS, decimals: 4 });

    // Evolución de pesos del meta-agente (endpoint agregado del backend)
    try {
      const weights = await api("/api/meta_weights?run_id=" + encodeURIComponent(runId));
      if (weights.rows && weights.rows.length) {
        global.TFMCharts.metaWeights(document.getElementById("learn-weights"), weights.rows);
      }
    } catch (e) { /* opcional: si el run no lo tiene, se omite */ }
  }

  global.TFM.views.learning = { render };
})(window);
