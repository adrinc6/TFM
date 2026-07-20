/* Detalle de un estudio: resumen (hipótesis, métrica de selección, años reservados),
   decisión por fase y comparativa de sus runs. Enlaza a cada run miembro. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct, table } = global.TFM;

  function decisionBlock(decision) {
    if (!decision || !Object.keys(decision).length) return "";
    return `<details style="margin-top:12px"><summary>Decisión por fase</summary><pre>${escapeHtml(JSON.stringify(decision, null, 2))}</pre></details>`;
  }

  // Etiquetas legibles de cada fase del ciclo del study.
  const PHASE_LABELS = {
    "1": "Fase 1 · ejes de modelo aislados",
    "2": "Fase 2 · combinación greedy",
    "3": "Fase 3 · afinado de hiperparámetros",
    "4_cartera": "Fase 4 · Cartera",
    "5_perfiles": "Fase 5 · Inversores",
  };

  // Comparativa por fases: una sección por fase; dentro, una fila por run con barra proporcional
  // de rank-IC y tarjetas de CAGR vs bench, Information Ratio y % que bate al SPY. No se muestra el
  // nombre del escenario (el usuario ya sabe cuál eligió).
  function phaseComparison(comparison) {
    const rows = (comparison || []).filter((r) => r && r.run_id);
    if (!rows.length) return `<p class="muted">Este estudio no tiene comparativa por fases.</p>`;
    const maxAbs = Math.max(...rows.map((r) => Math.abs(Number(r.mean_rank_ic) || 0)), 1e-9);
    // Agrupar por fase respetando el orden natural del ciclo.
    const order = ["1", "2", "3", "4_cartera", "5_perfiles"];
    const groups = {};
    for (const r of rows) {
      const key = String(r.phase);
      (groups[key] ??= []).push(r);
    }
    const keys = Object.keys(groups).sort((a, b) => {
      const ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    return keys.map((key) => {
      const items = groups[key].slice().sort((a, b) => (Number(b.mean_rank_ic) || 0) - (Number(a.mean_rank_ic) || 0));
      const bars = items.map((r) => barRow(r, maxAbs)).join("");
      const label = PHASE_LABELS[key] || `Fase ${key}`;
      return `<details><summary>${escapeHtml(label)} · ${items.length} runs</summary>
        <div class="phase-list">${bars}</div></details>`;
    }).join("");
  }

  // Etiqueta de la variable/valor que este run prueba (p.ej. "train_lookback_years = 6"), NO el
  // nombre del run/estudio (eso ya no se muestra: la fase agrupa y el usuario sabe qué eligió).
  function scenarioLabel(r) {
    if (r.axis && r.overrides && r.axis in r.overrides) return `${r.axis} = ${r.overrides[r.axis]}`;
    return String(r.scenario || "baseline");
  }

  function barRow(r, maxAbs) {
    const ic = Number(r.mean_rank_ic) || 0;
    const widthPct = Math.min(100, (Math.abs(ic) / maxAbs) * 100);
    const barCls = ic >= 0 ? "bar-pos" : "bar-neg";
    const card = (value, label, cls) =>
      `<div class="mini-card"><div class="mini-value ${cls || ""}">${value}</div><div class="mini-label">${label}</div></div>`;
    const cagrCls = (Number(r.cagr_difference) || 0) >= 0 ? "positive" : "negative";
    const onClick = r.run_id ? ` onclick="TFM.views.results.openRun('${escapeHtml(r.run_id)}')"` : "";
    return `<div class="phase-row click"${onClick}>
      <div class="bar-track"><div class="bar-fill ${barCls}" style="width:${widthPct.toFixed(1)}%"></div>
        <span class="bar-value">${escapeHtml(scenarioLabel(r))} · rank-IC ${fmt(ic, 3)}</span></div>
      <div class="mini-cards">
        ${card(pct(r.cagr_difference, 2), "CAGR vs bench", cagrCls)}
        ${card(fmt(r.information_ratio, 2), "Info Ratio")}
        ${card(pct(r.beat_rate, 0), "bate SPY")}
      </div>
    </div>`;
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
      <h4 style="margin-top:18px">Comparativa por fases</h4>
      <p class="muted">Barra proporcional al rank-IC; a la derecha, CAGR vs benchmark, Information Ratio y % de años que bate al SPY de cada run.</p>
      <div id="study-phases">${phaseComparison(data.comparison)}</div>`;
  }

  global.TFM.views.study = { open };
})(window);
