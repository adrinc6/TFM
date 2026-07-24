/* Detalle de un estudio: resumen (hipótesis, métrica de selección, años reservados),
   decisión por fase y comparativa de sus runs. Enlaza a cada run miembro. */
(function (global) {
  "use strict";
  const { api, escapeHtml, fmt, pct, table } = global.TFM;
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

  function installLiveActions(studyId, canResume, manifest, decision, comparison) {
    const actions = document.getElementById("study-live-actions");
    if (!actions) return;
    // Un ÚNICO botón que alterna según el estado: si el study está vivo (queued/running) es
    // "Parar" (rojo, dura e inmediata: mata el proceso y borra el run a medias); si no, es
    // "Reanudar", habilitado solo cuando terminó de forma anómala (failed/interrupted/cancelled).
    const canStop = ["queued", "running"].includes(String(manifest.status || ""));
    const actionButton = canStop
      ? `<button class="button danger" id="stop-full-study" title="${escapeHtml("Detiene el full study al instante: mata el proceso (y el dashboard) y elimina el run en curso. Queda reanudable.")}">Parar full study</button>`
      : `<button class="button primary" id="resume-full-study"${canResume ? "" : " disabled"} title="${escapeHtml(canResume ? "Verifica y reutiliza los runs completos; repite los incompletos y continúa el ciclo." : "Solo disponible si el estudio quedó interrumpido, fallido o cancelado.")}">Reanudar full study</button>`;
    actions.innerHTML = `<button class="button ghost" id="toggle-study-live-log">▣ Ver consola</button>
      <button class="button ghost" id="toggle-robustez">◔ Robustez</button>
      <button class="button ghost" id="toggle-perfiles">☰ Perfiles</button>
      ${actionButton}`;
    const toggle = document.getElementById("toggle-study-live-log");
    const robToggle = document.getElementById("toggle-robustez");
    const perfToggle = document.getElementById("toggle-perfiles");
    const consolePanel = () => document.getElementById("study-live-log");
    const robPanel = () => document.getElementById("study-robustez-panel");
    const perfPanel = () => document.getElementById("study-perfiles-panel");
    const results = () => document.getElementById("study-results-content");
    const selectorCards = () => document.getElementById("study-selector-cards");
    const viewIndicator = () => document.getElementById("study-view-indicator");

    // Los cuatro estados (resultados / consola / robustez / perfiles) son mutuamente excluyentes.
    // Cuando cualquiera de los tres paneles auxiliares está activo, se ocultan las tarjetas de
    // selección superiores y se muestra un indicador textual de en qué vista se está.
    const setSelectorVisibility = (label) => {
      selectorCards().hidden = !!label;
      const indicator = viewIndicator();
      indicator.hidden = !label;
      indicator.textContent = label || "";
    };

    toggle.addEventListener("click", () => {
      const panel = consolePanel();
      const visible = !panel.hidden;
      panel.hidden = visible;
      robPanel().hidden = true;
      perfPanel().hidden = true;
      results().hidden = !visible;
      toggle.textContent = visible ? "▣ Ver consola" : "▣ Ocultar consola";
      robToggle.textContent = "◔ Robustez";
      perfToggle.textContent = "☰ Perfiles";
      setSelectorVisibility(visible ? null : "Consola");
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
      perfPanel().hidden = true;
      results().hidden = !visible;
      robToggle.textContent = visible ? "◔ Robustez" : "◔ Ocultar robustez";
      toggle.textContent = "▣ Ver consola";
      perfToggle.textContent = "☰ Perfiles";
      setSelectorVisibility(visible ? null : "Robustez");
      stopLiveLog();
    });

    perfToggle.addEventListener("click", async () => {
      const panel = perfPanel();
      const visible = !panel.hidden;
      if (!visible) {
        panel.innerHTML = `<p class="muted">Cargando perfiles…</p>`;
        panel.hidden = false;
        consolePanel().hidden = true;
        robPanel().hidden = true;
        results().hidden = true;
        perfToggle.textContent = "☰ Ocultar perfiles";
        toggle.textContent = "▣ Ver consola";
        robToggle.textContent = "◔ Robustez";
        setSelectorVisibility("Perfiles");
        stopLiveLog();
        panel.innerHTML = await perfilesPanel(comparison);
        return;
      }
      panel.hidden = true;
      results().hidden = false;
      perfToggle.textContent = "☰ Perfiles";
      setSelectorVisibility(null);
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
      parts.push(`<details><summary>Placebo (permutación de etiquetas)</summary>${body}</details>`);
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
      parts.push(`<details><summary>Robustez multi-era (bootstrap + leave-one-year-out)</summary>${body}</details>`);
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
        ], "cards-5") + `<p class="muted" style="margin-top:8px">Compara la cartera del finalista contra carteras aleatorias
          del mismo tamaño. El modelo <strong>${beats ? "bate al azar de forma convincente" : "no bate al azar de forma convincente"}</strong>
          (percentil > 0.95).</p>`;
      }
      parts.push(`<details><summary>Carteras aleatorias (random-portfolio)</summary>${body}</details>`);
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

  // Panel de perfiles del estudio: compara los 8 runs de la fase 5_perfiles (uno por perfil de
  // inversor), todos sobre el mismo modelo y cartera ya optimizados. El perfil se reporta,
  // no se selecciona: no hay "ganador" aquí, solo comparación.
  async function perfilesPanel(comparison) {
    const rows = (comparison || []).filter((r) => r && r.run_id && String(r.phase) === "5_perfiles");
    if (!rows.length) return `<p class="muted">Este estudio todavía no tiene runs de perfiles de inversor.</p>`;
    const sorted = rows.slice().sort((a, b) => String(a.scenario || "").localeCompare(String(b.scenario || "")));

    // El rank-IC no se muestra aquí: los 8 perfiles comparten el mismo agente/modelo, así que es
    // idéntico entre filas y no aporta nada a la comparación. En su lugar, rentabilidad anualizada.
    const summaryCols = ["scenario", "cagr_portfolio", "cagr_difference", "information_ratio", "beat_rate"];
    const summaryLabels = { scenario: "Perfil", cagr_portfolio: "Rentabilidad anual", cagr_difference: "CAGR vs bench",
      information_ratio: "Info Ratio", beat_rate: "bate SPY" };
    const summaryHead = `<tr>${summaryCols.map((c) => `<th>${escapeHtml(summaryLabels[c])}</th>`).join("")}</tr>`;
    const summaryBody = sorted.map((r) => {
      const onClick = r.run_id ? ` onclick="TFM.views.results.openRun('${escapeHtml(r.run_id)}')"` : "";
      const cells = summaryCols.map((c) => {
        const v = r[c];
        if (c === "scenario") return `<td>${escapeHtml(String(v || "—"))}</td>`;
        if (c === "beat_rate") return `<td class="mono">${pct(v, 0)}</td>`;
        if (c === "cagr_portfolio" || c === "cagr_difference") return `<td class="mono">${pct(v, 2)}</td>`;
        return `<td class="mono">${fmt(v, 3)}</td>`;
      }).join("");
      return `<tr class="click"${onClick}>${cells}</tr>`;
    }).join("");
    const summaryTable = `<table class="data"><thead>${summaryHead}</thead><tbody>${summaryBody}</tbody></table>`;

    let annualSection = `<p class="muted">No se pudo cargar la rentabilidad anual por perfil.</p>`;
    try {
      const perfData = await Promise.all(sorted.map((r) =>
        api("/api/performance?run_id=" + encodeURIComponent(r.run_id))
          .then((d) => ({ profile: r.scenario, annual: d.annual || [] }))
          .catch(() => ({ profile: r.scenario, annual: [] }))));
      const years = Array.from(new Set(perfData.flatMap((p) => p.annual.map((a) => a.year)))).sort();
      if (years.length) {
        const head = `<tr><th>Año</th>${perfData.map((p) => `<th>${escapeHtml(String(p.profile))}</th>`).join("")}</tr>`;
        const body = years.map((year) => {
          const cells = perfData.map((p) => {
            const row = p.annual.find((a) => a.year === year);
            const alpha = row ? row.alpha : undefined;
            const cls = typeof alpha === "number" ? (alpha >= 0 ? "positive" : "negative") : "";
            return `<td class="mono ${cls}">${pct(alpha, 2)}</td>`;
          }).join("");
          return `<tr><td class="mono">${escapeHtml(String(year))}</td>${cells}</tr>`;
        }).join("");
        annualSection = `<p class="muted">Alfa (rentabilidad de cartera menos benchmark) por año y perfil.</p>
          <div class="table-wrap"><table class="data"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
      } else {
        annualSection = `<p class="muted">Ningún perfil tiene todavía métricas anuales.</p>`;
      }
    } catch (error) {
      annualSection = `<p class="muted">No se pudo cargar la rentabilidad anual por perfil: ${escapeHtml(error.message)}</p>`;
    }

    const runIds = sorted.map((r) => r.run_id).join(",");
    const compareBoxId = "study-perfiles-compare";

    return `<h4 style="margin-top:4px">Perfiles de inversor</h4>
      <p class="muted">Los ${sorted.length} perfiles se ejecutan sobre el mismo modelo y cartera ya optimizados;
        se <strong>reportan, no se seleccionan</strong>. Comparación directa entre ellos.</p>
      <details><summary>Comparativa de métricas</summary>
        <div class="table-wrap" style="margin-top:8px">${summaryTable}</div></details>
      <details style="margin-top:12px"><summary>Rentabilidad anual (alfa) por perfil</summary>
        <div style="margin-top:8px">${annualSection}</div></details>
      <details style="margin-top:12px" id="${compareBoxId}" data-run-ids="${escapeHtml(runIds)}" data-loaded="false"
        ontoggle="if(this.open &amp;&amp; this.dataset.loaded!=='true'){this.dataset.loaded='true';TFM.views.study.refreshPerfilesCompare();}">
        <summary>Composición de cartera por snapshot</summary>
        <div style="margin-top:8px">
          <label class="field">Fecha<select id="perfiles-compare-date" onchange="TFM.views.study.refreshPerfilesCompare()"></select></label>
          <div id="perfiles-compare-table" class="table-wrap" style="margin-top:8px"></div>
        </div>
      </details>`;
  }

  async function refreshPerfilesCompare() {
    const box = document.getElementById("study-perfiles-compare");
    const select = document.getElementById("perfiles-compare-date");
    const target = document.getElementById("perfiles-compare-table");
    if (!box || !select || !target) return;
    const runIds = (box.dataset.runIds || "").split(",").filter(Boolean);
    if (!runIds.length) return;
    let data;
    try {
      data = await api("/api/portfolio/compare?" + global.TFM.qs({ run_ids: runIds.join(","), date: select.value }));
    } catch (error) {
      target.innerHTML = `<div class="notice">${escapeHtml(error.message)}</div>`;
      return;
    }
    if (!select.options.length) {
      select.innerHTML = (data.dates || []).slice().reverse().map((d) => `<option value="${d}">${d}</option>`).join("");
    }
    if (data.selected_date) select.value = data.selected_date;
    const perRun = data.per_run || {};
    const tickers = Array.from(new Set(Object.values(perRun).flatMap((rows) => rows.map((r) => r.ticker)))).sort();
    if (!tickers.length) {
      target.innerHTML = `<p class="muted">Sin composición de cartera para esta fecha.</p>`;
      return;
    }
    const runIdList = Object.keys(perRun);
    const head = `<tr><th>Ticker</th>${runIdList.map((id) => `<th>${escapeHtml(id)}</th>`).join("")}</tr>`;
    const body = tickers.map((ticker) => {
      const cells = runIdList.map((id) => {
        const row = (perRun[id] || []).find((r) => r.ticker === ticker);
        return `<td class="mono">${row ? pct(row.weight, 1) : "—"}</td>`;
      }).join("");
      return `<tr><td>${escapeHtml(ticker)}</td>${cells}</tr>`;
    }).join("");
    target.innerHTML = `<table class="data"><thead>${head}</thead><tbody>${body}</tbody></table>`;
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
    const head = `<tr><th class="col-run">Run</th><th>Total</th>${activeStages.map((s) => `<th>${STAGE_LABEL[s] || s}</th>`).join("")
      }<th>CPU</th><th>Núcleos ef.</th><th>Pico RAM</th><th>Hilos</th></tr>`;
    const body = rows.map((r) => {
      const exec = r.execution, tel = exec.telemetry || {};
      const label = escapeHtml(perfRunLabel(r) || "");
      const onClick = r.run_id ? ` onclick="TFM.views.results.openRun('${escapeHtml(r.run_id)}')"` : "";
      const rss = typeof tel.rss_bytes_at_finish === "number"
        ? `${(tel.rss_bytes_at_finish / 1024 / 1024 / 1024).toFixed(2)} GB` : "—";
      const cores = typeof tel.effective_logical_cores === "number" ? fmt(tel.effective_logical_cores, 2) : "—";
      return `<tr class="click"${onClick}>
        <td class="col-run">${label}</td>
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
      <h3>${escapeHtml(m.name || studyId)} <span id="study-view-indicator" class="tag" hidden></span></h3>
      <p class="muted mono">${escapeHtml(studyId)} · <span class="tag">${escapeHtml(m.kind || "")}</span> <span class="tag">${escapeHtml(m.status || "")}</span></p>
      ${m.description ? `<p class="muted">${escapeHtml(m.description)}</p>` : ""}
      <div id="study-selector-cards">
        <div class="cards" style="margin-top:12px">
          <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(m.selection_metric || "rank-IC")}</div><div class="metric-label">Métrica de selección</div></div>
          <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(String(m.selection_until_year || "—"))}</div><div class="metric-label">Selección hasta año</div></div>
          <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(JSON.stringify(m.reserved_years || "—"))}</div><div class="metric-label">Años reservados</div></div>
          <div class="card"><div class="metric">${runs.length}</div><div class="metric-label">Runs del estudio</div></div>
          <div class="card"><div class="metric" style="font-size:18px">${escapeHtml(String(m.current_phase || "—"))}</div><div class="metric-label">Fase/checkpoint actual</div></div>
        </div>
        ${m.current_scenario ? `<p class="muted">Escenario actual: <code>${escapeHtml(m.current_scenario)}</code></p>` : ""}
      </div>
      <section id="study-live-log" class="notice" hidden style="margin-top:12px">
        <strong>Consola de ejecución (solo lectura)</strong>
        <span id="study-live-log-status" class="muted" style="margin-left:8px">Conectando…</span>
        <pre id="study-live-log-output" class="mono" style="height:26em; overflow:auto; margin:10px 0 0; white-space:pre-wrap">Esperando…</pre>
      </section>
      <section id="study-robustez-panel" class="panel" hidden style="margin-top:12px"></section>
      <section id="study-perfiles-panel" class="panel" hidden style="margin-top:12px"></section>
      <div id="study-results-content">
        ${decisionBlock(data.decision)}
        <h4 style="margin-top:18px">Comparativa por fases</h4>
        <p class="muted">Barra proporcional al rank-IC; a la derecha, CAGR vs benchmark, Information Ratio y % de años que bate al SPY de cada run.</p>
        <div id="study-phases">${phaseComparison(data.comparison)}</div>
        ${performanceBlock(runs)}
      </div>`;
    installLiveActions(studyId, resumable, m, data.decision, data.comparison);
    const stop = document.getElementById("stop-full-study");
    if (stop) stop.addEventListener("click", async () => {
      // El study corre dentro del servidor, así que pararlo TIRA el dashboard: hay que relanzar
      // `python main.py`. El estado 'cancelled' y el borrado del run a medias se persisten antes
      // de matar, de modo que al reabrir el estudio queda reanudable.
      if (!confirm("Se detendrá el full study AL INSTANTE. Esto cierra también el dashboard (comparte proceso), así que tendrás que volver a arrancar 'python main.py'. El run a medias se elimina y el estudio queda reanudable. ¿Continuar?")) return;
      stop.disabled = true;
      stop.textContent = "Parando…";
      try {
        await api("/api/study/" + encodeURIComponent(studyId) + "/stop", {});
        stopLiveLog();
        container.innerHTML = `<div class="notice">Full study detenido y marcado como cancelado. El dashboard se ha cerrado con el proceso: vuelve a arrancar <code>python main.py</code> y recarga esta página para reanudarlo.</div>`;
      } catch (error) {
        alert(error.message);
        stop.disabled = false;
        stop.textContent = "Parar full study";
      }
    });
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

  global.TFM.views.study = { open, refreshPerfilesCompare };
})(window);
