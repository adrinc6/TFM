/* Detalle de un estudio: resumen (hipótesis, métrica de selección, años reservados),
   decisión por fase y comparativa de sus runs. Enlaza a cada run miembro. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, table } = global.TFM;

  function decisionBlock(decision) {
    if (!decision || !Object.keys(decision).length) return "";
    return `<details style="margin-top:12px"><summary>Decisión por fase</summary><pre>${escapeHtml(JSON.stringify(decision, null, 2))}</pre></details>`;
  }

  async function open(studyId, container) {
    global.TFMCharts.clear();
    container.innerHTML = `<p class="muted">Cargando estudio…</p>`;
    let data;
    try {
      data = await api("/api/study/" + encodeURIComponent(studyId));
    } catch (e) {
      container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
      return;
    }
    const m = data.manifest || {};
    const runs = data.runs || [];
    container.innerHTML = `
      <h3>${escapeHtml(m.name || studyId)}</h3>
      <p class="muted mono">${escapeHtml(studyId)} · <span class="tag">${escapeHtml(m.kind || "")}</span> <span class="tag">${escapeHtml(m.status || "")}</span></p>
      ${m.description ? `<p class="muted">${escapeHtml(m.description)}</p>` : ""}
      <div class="cards" style="margin-top:12px">
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(m.selection_metric || "rank-IC")}</div><div class="metric-label">Métrica de selección</div></div>
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(String(m.selection_until_year || "—"))}</div><div class="metric-label">Selección hasta año</div></div>
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(JSON.stringify(m.reserved_years || "—"))}</div><div class="metric-label">Años reservados</div></div>
        <div class="card"><div class="metric">${runs.length}</div><div class="metric-label">Runs del estudio</div></div>
      </div>
      ${decisionBlock(data.decision)}
      <h4 style="margin-top:18px">Comparativa de runs</h4>
      <div id="study-chart"></div>
      <div id="study-runs"></div>`;

    // Gráfico comparativo por rank-IC medio
    const withIc = runs.map((r) => ({ run_id: r.run_id, label: r.label || r.run_id, mean_rank_ic: (r.summary || {}).mean_rank_ic }));
    global.TFMCharts.studyComparison(el("study-chart"), withIc, "mean_rank_ic", "rank-IC medio por run");

    // Tabla clicable de runs miembro
    const rows = runs.map((r) => {
      const s = r.summary || {};
      return `<tr class="click" onclick="TFM.views.results.openRun('${escapeHtml(r.run_id)}')">
        <td>${escapeHtml(r.label || r.run_id)}<br><small class="muted mono">${escapeHtml(r.run_id)}</small></td>
        <td>${escapeHtml(r.run_kind || "")}</td>
        <td class="${(s.mean_rank_ic || 0) >= 0 ? "positive" : "negative"}">${fmt(s.mean_rank_ic)}</td>
        <td>${fmt(s.cagr_difference)}</td>
        <td>${escapeHtml(r.status || "")}</td>
      </tr>`;
    }).join("");
    el("study-runs").innerHTML = runs.length
      ? `<div class="table-wrap scroll"><table class="data"><thead><tr><th>Run</th><th>Tipo</th><th>rank-IC</th><th>CAGR vs bench</th><th>Estado</th></tr></thead><tbody>${rows}</tbody></table></div>`
      : `<p class="empty">Este estudio aún no tiene runs registrados.</p>`;
  }

  global.TFM.views.study = { open };
})(window);
