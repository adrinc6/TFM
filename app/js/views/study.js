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

  function installLiveActions(studyId, canResume, manifest, decision) {
    const actions = document.getElementById("study-live-actions");
    if (!actions) return;
    // El botón de reanudar se muestra SIEMPRE (mismo formato que los demás); solo se habilita
    // cuando el study terminó de forma anómala (failed/interrupted/cancelled). En cualquier otro
    // estado se ve deshabilitado con un motivo, en vez de desaparecer.
    const resumeTitle = canResume
      ? "Verifica y reutiliza los runs completos; repite los incompletos y continúa el ciclo."
      : "Solo disponible si el estudio quedó interrumpido, fallido o cancelado.";
    actions.innerHTML = `<button class="button ghost" id="toggle-study-live-log">▣ Ver consola</button>
      <button class="button ghost" id="toggle-robustez">◔ Robustez</button>
      <button class="button primary" id="resume-full-study"${canResume ? "" : " disabled"} title="${escapeHtml(resumeTitle)}">Reanudar full study</button>`;
    const toggle = document.getElementById("toggle-study-live-log");
    const robToggle = document.getElementById("toggle-robustez");
    const consolePanel = () => document.getElementById("study-live-log");
    const robPanel = () => document.getElementById("study-robustez-panel");
    const results = () => document.getElementById("study-results-content");

    // Los tres estados (resultados / consola / robustez) son mutuamente excluyentes.
    toggle.addEventListener("click", () => {
      const panel = consolePanel();
      const visible = !panel.hidden;
      panel.hidden = visible;
      robPanel().hidden = true;
      results().hidden = !visible;
      toggle.textContent = visible ? "▣ Ver consola" : "▣ Ocultar consola";
      robToggle.textContent = "◔ Robustez";
      stopLiveLog();
      if (!visible) {
        refreshLiveLog(studyId);
        if (["queued", "running"].includes(String(manifest.status || ""))) {
          liveLogTimer = window.setInterval(() => refreshLiveLog(studyId), 3000);
        }
      }
    });

    robToggle.addEventListener("click", () => {
      const panel = robPanel();
      const visible = !panel.hidden;
      if (!visible) panel.innerHTML = robustezPanel(decision);
      panel.hidden = visible;
      consolePanel().hidden = true;
      results().hidden = !visible;
      robToggle.textContent = visible ? "◔ Robustez" : "◔ Ocultar robustez";
      toggle.textContent = "▣ Ver consola";
      stopLiveLog();
    });
  }

  // Panel de robustez del estudio: consolida placebo, bootstrap por bloques, leave-one-year-out,
  // random-portfolio y los estreses de costes y de cartera. Todo viene de decision.json; ninguno
  // de estos resultados selecciona configuración: son evidencia y estrés, se reportan tal cual.
  function robustezPanel(decision) {
    const d = decision || {};
    const rob = d.robustness || {};
    const parts = [];

    const lp = rob.label_permutation;
    if (lp) {
      let body;
      if (lp.status) {
        body = `<p class="muted">${escapeHtml(String(lp.status))}</p>`;
      } else {
        const above = !!lp.signal_above_chance;
        body = global.TFM.metricGrid([
          { value: fmt(lp.rank_ic_real, 3), label: "rank-IC real", hintKey: "rank_ic" },
          { value: `${fmt(lp.placebo_mean, 3)} ± ${fmt(lp.placebo_std, 3)}`, label: "placebo (media ± sd)" },
          { value: fmt(lp.p_value, 3), label: "p-valor", cls: above ? "pos" : "neg" },
          { value: String(lp.n_permutations ?? "—"), label: "permutaciones" },
        ]) + `<p class="muted" style="margin-top:8px">Placebo: se reentrena el finalista con los retornos
          futuros barajados. Si el rank-IC real supera al placebo con p-valor bajo, la señal
          <strong>${above ? "está por encima del azar" : "no se distingue del azar"}</strong>.</p>`;
      }
      parts.push(`<details open><summary>Placebo (permutación de etiquetas)</summary>${body}</details>`);
    }

    const boot = rob.block_bootstrap;
    const loyo = rob.leave_one_year_out || [];
    if (boot || loyo.length) {
      let body = "";
      if (boot) {
        const crossesZero = boot.ci_low <= 0 && boot.ci_high >= 0;
        body += global.TFM.metricGrid([
          { value: fmt(boot.mean, 3), label: "rank-IC medio", hintKey: "rank_ic" },
          { value: `[${fmt(boot.ci_low, 3)}, ${fmt(boot.ci_high, 3)}]`, label: "IC 95 % (bootstrap por bloques)",
            cls: crossesZero ? "neg" : "pos" },
          { value: String(boot.n_cohorts ?? "—"), label: "cohortes" },
          { value: String(boot.block_size ?? "—"), label: "tamaño de bloque" },
        ]);
        if (crossesZero) {
          body += `<p class="muted" style="margin-top:8px">El intervalo de confianza cruza el cero: la señal es
            <strong>estadísticamente indistinguible de cero</strong> (no hay evidencia robusta de aprendizaje).</p>`;
        }
      }
      if (loyo.length) {
        const negatives = loyo.filter((r) => (r.rank_ic_without_it ?? 0) < 0).length;
        body += `<p class="muted" style="margin-top:8px">Leave-one-year-out: el rank-IC es negativo al excluir
          ${negatives} de ${loyo.length} años (mide si el resultado depende de un solo año).</p>` +
          table(loyo, { columns: ["excluded_year", "rank_ic_without_it", "delta_vs_full", "n_cohorts"],
            labels: { excluded_year: "Año excluido", rank_ic_without_it: "rank-IC sin él", delta_vs_full: "Δ vs total", n_cohorts: "Cohortes" },
            decimals: 4, sortable: true });
      }
      parts.push(`<details open><summary>Robustez multi-era (bootstrap + leave-one-year-out)</summary>${body}</details>`);
    }

    const rp = rob.random_portfolio;
    if (rp) {
      let body;
      if (rp.status) {
        body = `<p class="muted">${escapeHtml(String(rp.status))}</p>`;
      } else {
        const beats = !!rp.beats_random_convincingly;
        body = global.TFM.metricGrid([
          { value: pct(rp.model_cagr, 2), label: "CAGR del modelo", hintKey: "cagr" },
          { value: pct(rp.random_cagr_mean, 2), label: "CAGR aleatorio (media)" },
          { value: pct(rp.random_cagr_p95, 2), label: "CAGR aleatorio (p95)" },
          { value: fmt(rp.model_percentile, 3), label: "percentil del modelo", cls: beats ? "pos" : "neg" },
          { value: String(rp.n_simulations ?? "—"), label: "simulaciones" },
        ]) + `<p class="muted" style="margin-top:8px">Compara la cartera del finalista contra carteras aleatorias
          del mismo tamaño. El modelo <strong>${beats ? "bate al azar de forma convincente" : "no bate al azar de forma convincente"}</strong>
          (percentil > 0.95).</p>`;
      }
      parts.push(`<details open><summary>Carteras aleatorias (random-portfolio)</summary>${body}</details>`);
    }

    const econCols = ["cagr_difference", "information_ratio", "beat_rate", "max_drawdown"];
    const econLabels = { cagr_difference: "CAGR vs bench", information_ratio: "Info Ratio",
      beat_rate: "bate SPY", max_drawdown: "Max drawdown" };

    const cost = d.cost_stress || [];
    if (cost.length) {
      const cols = ["commission_bps", "slippage_bps", ...econCols];
      parts.push(`<details><summary>Estrés de costes · ${cost.length} escenarios</summary>
        <p class="muted" style="margin-top:8px">Cada par comisión × slippage se reporta sobre el finalista; los costes
          se <strong>estresan, no se optimizan</strong>: ninguno altera la configuración recomendada.</p>
        ${table(cost, { columns: cols,
          labels: { commission_bps: "Comisión (bps)", slippage_bps: "Slippage (bps)", ...econLabels },
          decimals: 3, sortable: true })}</details>`);
    }

    const port = d.portfolio_stress || [];
    if (port.length) {
      const cols = ["axis", "scenario", ...econCols];
      parts.push(`<details><summary>Estrés de reglas de cartera · ${port.length} escenarios</summary>
        <p class="muted" style="margin-top:8px">Ejes mecánicos (drift, expulsión, rotación) barridos aislados sobre el
          finalista; se <strong>estresan, no se seleccionan</strong>.</p>
        ${table(port, { columns: cols,
          labels: { axis: "Eje", scenario: "Escenario", ...econLabels },
          decimals: 3, sortable: true })}</details>`);
    }

    if (!parts.length) return `<p class="muted">Este estudio no incluyó pruebas de robustez.</p>`;
    return `<h4 style="margin-top:4px">Robustez, placebo y estrés</h4>
      <p class="muted">Evidencia sobre si el modelo aprende de verdad y se sostiene bajo costes y reglas de cartera
        distintas. Nada de esto selecciona configuración.</p>${parts.join("")}`;
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
      <section id="study-robustez-panel" class="panel" hidden style="margin-top:12px"></section>
      <div id="study-results-content">
        ${decisionBlock(data.decision)}
        <h4 style="margin-top:18px">Comparativa por fases</h4>
        <p class="muted">Barra proporcional al rank-IC; a la derecha, CAGR vs benchmark, Information Ratio y % de años que bate al SPY de cada run.</p>
        <div id="study-phases">${phaseComparison(data.comparison)}</div>
        ${performanceBlock(runs)}
      </div>`;
    installLiveActions(studyId, resumable, m, data.decision);
    const resume = document.getElementById("resume-full-study");
    if (resume) resume.addEventListener("click", async () => {
      if (!confirm("Se verificarán y reutilizarán los runs completos; los incompletos se repetirán. ¿Continuar?")) return;
      try {
        const job = await api("/api/optimization", {
          settings: global.TFM.state.defaults,
          study: { name: m.name || "optimization-official", hypothesis: m.hypothesis || "", resume_study_id: studyId },
        });
        resume.disabled = true;
        resume.textContent = `Reanudación iniciada`;
      } catch (error) {
        alert(error.message);
      }
    });
  }

  global.TFM.views.study = { open };
})(window);
