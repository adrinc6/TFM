/* Detalle de un estudio: resumen (hipótesis, métrica de selección, años reservados),
   decisión por fase y comparativa de sus runs. Enlaza a cada run miembro. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct, table } = global.TFM;
  let liveLogTimer = null;

  function stopLiveLog() {
    if (liveLogTimer !== null) window.clearInterval(liveLogTimer);
    liveLogTimer = null;
  }

  async function refreshLiveLog(studyId) {
    const status = document.getElementById("study-live-log-status");
    const output = document.getElementById("study-live-log-output");
    if (!status || !output) return;
    try {
      const live = await api("/api/study/" + encodeURIComponent(studyId) + "/live-log");
      const runs = live.active_runs || [];
      status.textContent = runs.length
        ? `${runs.length} run${runs.length === 1 ? "" : "s"} activo${runs.length === 1 ? "" : "s"}${live.current_phase ? " · fase " + live.current_phase : ""}`
        : "No hay runs activos asociados a este study.";
      output.textContent = (live.lines || []).join("\n") || "Esperando la primera línea de ejecución…";
      output.scrollTop = output.scrollHeight;
    } catch (error) {
      status.textContent = "No se pudo leer la consola: " + error.message;
    }
  }

  function installLiveActions(studyId, canResume, manifest) {
    const actions = document.getElementById("study-live-actions");
    if (!actions) return;
    actions.innerHTML = `<button class="button ghost" id="toggle-study-live-log">▣ Ver consola</button>
      ${canResume ? '<button class="primary" id="resume-full-study">Reanudar full study</button>' : ""}`;
    const toggle = document.getElementById("toggle-study-live-log");
    toggle.addEventListener("click", () => {
      const panel = document.getElementById("study-live-log");
      const results = document.getElementById("study-results-content");
      const visible = !panel.hidden;
      panel.hidden = visible;
      results.hidden = !visible;
      toggle.textContent = visible ? "▣ Ver consola" : "▣ Ocultar consola";
      stopLiveLog();
      if (!visible) {
        refreshLiveLog(studyId);
        if (["queued", "running"].includes(String(manifest.status || ""))) {
          liveLogTimer = window.setInterval(() => refreshLiveLog(studyId), 3000);
        }
      }
    });
  }

  function decisionBlock(decision) {
    if (!decision || !Object.keys(decision).length) return "";
    return `<details style="margin-top:12px"><summary>Decisión por fase</summary><pre>${escapeHtml(JSON.stringify(decision, null, 2))}</pre></details>`;
  }

  // Etiquetas legibles de cada fase del ciclo del study.
  const PHASE_LABELS = {
    "1": "Fase 1 · ejes de modelo aislados",
    "2": "Fase 2 · combinación greedy",
    "3": "Fase 3 · afinado de hiperparámetros",
    "3b_seed_stability": "Validación de estabilidad · semillas prefijadas",
    "4_cartera": "Fase 4 · Cartera",
    "4b_cost_stress": "Estrés de costes · no seleccionable",
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
    const order = ["1", "2", "3", "3b_seed_stability", "4_cartera", "4b_cost_stress", "5_perfiles"];
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
      const items = groups[key].slice().sort((a, b) => {
        const aBaseline = String(a.scenario || "") === "baseline" ? 0 : 1;
        const bBaseline = String(b.scenario || "") === "baseline" ? 0 : 1;
        if (aBaseline !== bBaseline) return aBaseline - bBaseline;
        const axis = String(a.axis || "").localeCompare(String(b.axis || ""));
        return axis || String(a.scenario || "").localeCompare(String(b.scenario || ""));
      });
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

  // --- Rendimiento por run -----------------------------------------------------------------
  // Orden canónico de etapas del pipeline; solo se muestran las que cada run realmente ejecutó.
  const PERF_STAGES = ["dataset", "features", "agents", "backtest"];
  const STAGE_LABEL = { dataset: "Dataset", features: "Features", agents: "Agents", backtest: "Backtest" };

  function fmtSeconds(s) {
    if (typeof s !== "number" || !Number.isFinite(s)) return "—";
    if (s < 90) return `${s.toFixed(1)} s`;
    const m = Math.floor(s / 60), r = Math.round(s % 60);
    return r ? `${m} min ${r} s` : `${m} min`;
  }

  // Celda de una etapa, estilo "cuenta de resultados": si la etapa se calculó de verdad, el tiempo
  // va normal; si fue reciclado/cacheado, va entre paréntesis (como los negativos en contabilidad).
  function stageCell(exec, stage) {
    const t = (exec.stage_timings_seconds || {})[stage];
    if (t === undefined) return `<td class="muted">—</td>`;
    const shown = fmtSeconds(t);
    const recycled = (exec.stage_source || {})[stage] === "recycled";
    return recycled ? `<td class="mono muted">(${shown})</td>` : `<td class="mono">${shown}</td>`;
  }

  // Total del run: preferimos el wall observado por telemetría; si falta, sumamos las etapas.
  function runTotalSeconds(exec) {
    const wall = (exec.telemetry || {}).wall_seconds;
    if (typeof wall === "number" && Number.isFinite(wall)) return wall;
    const timings = exec.stage_timings_seconds || {};
    const values = Object.values(timings).filter((v) => typeof v === "number" && Number.isFinite(v));
    return values.length ? values.reduce((a, b) => a + b, 0) : NaN;
  }

  // Etiqueta de fila de la tabla de rendimiento: aquí SÍ importa distinguir cada run individual,
  // así que si no hay eje/override que lo identifique, usamos el run_id real (no el genérico
  // "baseline", que colapsaría todas las filas al mismo texto).
  function perfRunLabel(r) {
    // El intent.label del run trae algo como "optimization · Fase 1 · execution_lag_days_15";
    // mostramos solo a partir de "Fase" (quitando el prefijo del tipo de estudio).
    if (r.intent_label) {
      const idx = r.intent_label.indexOf("Fase");
      return idx >= 0 ? r.intent_label.slice(idx) : r.intent_label;
    }
    if (r.axis && r.overrides && r.axis in r.overrides) return `${r.axis} = ${r.overrides[r.axis]}`;
    if (r.run_id) return String(r.run_id);
    return String(r.scenario || "baseline");
  }

  function performanceTable(runs) {
    const withExec = (runs || []).filter((r) => r && r.execution && r.execution.stage_timings_seconds);
    if (!withExec.length) return `<p class="muted">Todavía no hay métricas de rendimiento para los runs de este estudio.</p>`;
    // Solo columnas de etapa que algún run haya ejecutado (evita columnas vacías).
    const activeStages = PERF_STAGES.filter((st) =>
      withExec.some((r) => (r.execution.stage_timings_seconds || {})[st] !== undefined));
    const rows = withExec.slice().sort((a, b) => runTotalSeconds(b.execution) - runTotalSeconds(a.execution));
    const head = `<tr><th>Run</th><th>Total</th>${activeStages.map((s) => `<th>${STAGE_LABEL[s] || s}</th>`).join("")
      }<th>CPU</th><th>Núcleos ef.</th><th>Pico RAM</th><th>Hilos</th></tr>`;
    const body = rows.map((r) => {
      const exec = r.execution, tel = exec.telemetry || {};
      const label = escapeHtml(perfRunLabel(r) || "");
      const onClick = r.run_id ? ` onclick="TFM.views.results.openRun('${escapeHtml(r.run_id)}')"` : "";
      const rss = typeof tel.rss_bytes_at_finish === "number"
        ? `${(tel.rss_bytes_at_finish / 1024 / 1024 / 1024).toFixed(2)} GB` : "—";
      const cores = typeof tel.effective_logical_cores === "number" ? fmt(tel.effective_logical_cores, 2) : "—";
      return `<tr class="click"${onClick}>
        <td>${label}</td>
        <td class="mono"><strong>${fmtSeconds(runTotalSeconds(exec))}</strong></td>
        ${activeStages.map((s) => stageCell(exec, s)).join("")}
        <td class="mono">${fmtSeconds(tel.process_cpu_seconds)}</td>
        <td class="mono">${cores}</td>
        <td class="mono">${rss}</td>
        <td class="mono">${typeof tel.model_threads === "number" ? tel.model_threads : "—"}</td>
      </tr>`;
    }).join("");
    return `<div class="table-wrap"><table class="data perf-table">
      <thead>${head}</thead><tbody>${body}</tbody></table></div>`;
  }

  function performanceBlock(runs) {
    return `<details data-keep-closed style="margin-top:22px"><summary>Rendimiento de los runs</summary>
      <p class="muted" style="margin-top:8px">Tiempo total y por etapa de cada run. Los tiempos
        <em>entre paréntesis</em> son etapas recicladas (restauradas de caché, no recalculadas); el resto se
        calcularon de verdad. A la derecha, CPU consumida, núcleos lógicos efectivos, pico de RAM e hilos por modelo.</p>
      ${performanceTable(runs)}</details>`;
  }

  async function open(studyId, container) {
    stopLiveLog();
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
    // Un study en cola o ejecución no se puede reanudar: el proceso original conserva el lock y
    // sus checkpoints. Solo una terminación anómala habilita la reanudación explícita.
    const resumable = ["failed", "interrupted", "cancelled"].includes(String(m.status || ""));
    container.innerHTML = `
      <h3>${escapeHtml(m.name || studyId)}</h3>
      <p class="muted mono">${escapeHtml(studyId)} · <span class="tag">${escapeHtml(m.kind || "")}</span> <span class="tag">${escapeHtml(m.status || "")}</span></p>
      ${m.description ? `<p class="muted">${escapeHtml(m.description)}</p>` : ""}
      <div class="cards" style="margin-top:12px">
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(m.selection_metric || "rank-IC")}</div><div class="metric-label">Métrica de selección</div></div>
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(String(m.selection_until_year || "—"))}</div><div class="metric-label">Selección hasta año</div></div>
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(JSON.stringify(m.reserved_years || "—"))}</div><div class="metric-label">Años reservados</div></div>
        <div class="card"><div class="metric">${runs.length}</div><div class="metric-label">Runs del estudio</div></div>
        <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(String(m.current_phase || "—"))}</div><div class="metric-label">Fase/checkpoint actual</div></div>
      </div>
      ${m.current_scenario ? `<p class="muted">Escenario actual: <code>${escapeHtml(m.current_scenario)}</code></p>` : ""}
      <section id="study-live-log" class="notice" hidden style="margin-top:12px">
        <strong>Consola de ejecución (solo lectura)</strong>
        <span id="study-live-log-status" class="muted" style="margin-left:8px">Conectando…</span>
        <pre id="study-live-log-output" class="mono" style="height:26em; overflow:auto; margin:10px 0 0; white-space:pre-wrap">Esperando…</pre>
      </section>
      <div id="study-results-content">
        ${decisionBlock(data.decision)}
        <h4 style="margin-top:18px">Comparativa por fases</h4>
        <p class="muted">Barra proporcional al rank-IC; a la derecha, CAGR vs benchmark, Information Ratio y % de años que bate al SPY de cada run.</p>
        <div id="study-phases">${phaseComparison(data.comparison)}</div>
        ${performanceBlock(runs)}
      </div>`;
    installLiveActions(studyId, resumable, m);
    const resume = document.getElementById("resume-full-study");
    if (resume) resume.addEventListener("click", async () => {
      if (!confirm("Se verificarán y reutilizarán los runs completos; los incompletos se repetirán. ¿Continuar?")) return;
      try {
        const job = await api("/api/optimization", {
          settings: global.TFM.state.defaults,
          study: { name: m.name || "optimization-official", hypothesis: m.hypothesis || "", resume_study_id: studyId },
        });
        resume.disabled = true;
        resume.textContent = `Reanudación ${job.job_id} iniciada`;
      } catch (error) {
        alert(error.message);
      }
    });
  }

  global.TFM.views.study = { open };
})(window);
