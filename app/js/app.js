(() => {
  const app = document.getElementById("app");
  const navBack = document.getElementById("nav-back");
  const dev = location.pathname.startsWith("/dev");
  const base = dev ? "/dev" : "";
  const state = {
    catalog: null, definition: null, budget: null, studies: [],
    selectedStudy: null, selectedRun: null, section: "summary", eventSequence: 0, timer: null,
    snapshot: null, stockTicker: null, stockView: "portfolio", stockParameter: null,
    retainAllRuns: false,
  };
  const gigabytes = bytes => `${(Number(bytes || 0) / 1024 ** 3).toFixed(1)} GB`;

  const api = async (path, options = {}) => {
    const response = await fetch(base + path, {...options, headers: {"Content-Type": "application/json"}});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  };
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
  const bpsKey = key => /_bps$/.test(key);
  const percentageKey = key => !bpsKey(key) && /(^|_)(alpha|return|cagr|drawdown|rate|fraction|weight|turnover|p_value|pnl_pct)(_|$)|rank_ic/.test(key);
  const integerKey = key => /^(year|runs_completados|runs_restantes|runs_total|observations|training_rows|realized_cohorts|closed_cohorts|attempt|target_size|elapsed_seconds|positive_alpha_years|positive_alpha_eras|months_held)$/.test(key);
  const timestampKey = key => /^(actualizado|updated_at|created_at|heartbeat)$/.test(key);
  const rawPercentileKey = key => /^(current_percentile|percentile)$/.test(key);
  const fmt = (value, key = "") => {
    if (typeof value !== "number" || !Number.isFinite(value)) return value ?? "—";
    if (key === "elapsed_seconds") return formatElapsedSeconds(value);
    if (bpsKey(key)) return `${value.toFixed(0)} pb`;
    if (integerKey(key)) return String(Math.round(value));
    if (rawPercentileKey(key)) return `p${value.toFixed(0)}`;
    if (percentageKey(key)) return `${(value * 100).toFixed(2)} %`;
    return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
  };
  const madridTime = isoTimestamp => {
    const date = new Date(isoTimestamp);
    return Number.isNaN(date.getTime()) ? isoTimestamp : date.toLocaleTimeString("es-ES", {
      timeZone: "Europe/Madrid", hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  };
  const display = (value, key = "") => timestampKey(key) && typeof value === "string"
    ? madridTime(value)
    : fmt(value, key);
  const tone = (value, key = "") => typeof value === "number" && /(alpha|return|rank_ic|information|p_value|pnl_pct)/.test(key) ? (value > 0 ? "positive" : value < 0 ? "negative" : "") : "";
  const notify = (message, error = false) => {
    const toast = document.getElementById("toast");
    toast.textContent = message; toast.className = `show ${error ? "error" : ""}`;
    setTimeout(() => toast.className = "", 3000);
  };

  document.querySelectorAll("nav button").forEach(button => button.onclick = () => {
    if (!button.dataset.view) return;
    document.querySelectorAll("nav button").forEach(item => item.classList.toggle("active", item === button));
    clearInterval(state.timer);
    button.dataset.view === "home" ? renderHome() : renderResults();
  });

  function setBackNavigation(label = "", action = null) {
    navBack.textContent = label;
    navBack.classList.toggle("hidden", !action);
    navBack.onclick = action || null;
  }

  function stageRuns(stage) {
    if (!state.budget) return 0;
    if (stage === "portfolio") {
      return state.catalog.variables
        .filter(variable => variable.stage === stage && !variable.predictive)
        .reduce((sum, variable) => sum + Math.max(0, (state.definition[variable.id]?.values.length || 0) - 1), 0);
    }
    return Object.entries(state.budget.breakdown || {}).filter(([id]) =>
      state.catalog.variables.find(variable => variable.id === id)?.stage === stage
    ).reduce((sum, [, count]) => sum + count, 0);
  }

  function variableRow(variable) {
    const selection = state.definition[variable.id];
    const active = variable.depends_on.every(dep => {
      const controller = state.definition[dep.variable]?.values || [];
      return controller.some(value => dep.values.includes(value));
    });
    if (!active) return "";
    const columnCount = Math.min(4, variable.value_options.length);
    return `<div class="variable-row" data-variable="${esc(variable.id)}">
      <div class="variable-copy"><strong>${esc(variable.label)}</strong><p>${esc(variable.description)}</p>
        <small>${selection.values.length === 1 ? "Fixed" : variable.predictive ? "Optimize" : "Diagnóstico"} · Coste: ${esc(variable.cost)} · Baseline: ${esc(selection.baseline)}</small></div>
      <div class="value-choices columns-${columnCount}">${variable.value_options.map(option => {
        const checked = selection.values.some(value => JSON.stringify(value) === JSON.stringify(option.value));
        const isBaseline = JSON.stringify(selection.baseline) === JSON.stringify(option.value);
        return `<label class="value-choice ${isBaseline ? "is-baseline" : ""}"><input type="checkbox"
          name="value-${esc(variable.id)}" data-value='${esc(JSON.stringify(option.value))}' ${checked ? "checked" : ""}>
          <span><b>${esc(option.label)}</b><small>${esc(option.description)}</small></span>
          ${checked ? `<label class="baseline-pick" title="Usar como baseline"><input type="radio"
            name="baseline-${esc(variable.id)}" data-value='${esc(JSON.stringify(option.value))}' ${isBaseline ? "checked" : ""}>
            <small>Baseline</small></label>` : ""}</label>`;
      }).join("")}</div>
    </div>`;
  }

  function renderHome() {
    setBackNavigation();
    const stages = state.catalog.stage_order.map(stage => {
      const detail = state.catalog.stages.find(item => item.id === stage);
      const variables = state.catalog.variables.filter(variable => variable.stage === stage);
      return `<section class="stage-card"><div class="stage-heading"><div><p class="eyebrow">${esc(detail.label)}</p>
        <h2>${esc(detail.question)}</h2><p>${esc(detail.description)}</p></div>
        <div class="stage-plan"><b>${stageRuns(stage)}</b><span>${stage === "portfolio" ? "comparaciones informativas" : "evaluaciones predictivas"}</span>
        <small>${stage === "portfolio" ? "Comparaciones informativas; nunca eligen modelo." : "Comparación secuencial contra el baseline de cada variable, con las anteriores ya decididas."}</small></div></div>
        <div>${variables.map(variableRow).join("")}</div></section>`;
    }).join("");
    app.innerHTML = `<section class="hero"><div><p class="eyebrow">Configuración cerrada</p><h2>Un Study, una ruta científica</h2>
      <p>Solo Rank-IC robusto decide. Carteras, perfiles y robustez explican el ganador después.</p></div>${budgetMarkup()}</section>
      <div class="study-meta"><label>Nombre<input id="study-name" value="Model Study"></label>
      <label>Nota<input id="study-note" placeholder="Opcional; no afecta a la ciencia"></label></div>
      <label class="study-option ${state.retainAllRuns ? "is-on" : ""}">
        <input type="checkbox" id="retain-all" ${state.retainAllRuns ? "checked" : ""}>
        <span><b>Guardar la evidencia de todos los runs</b>
        <small>Cada run conserva su cartera, órdenes, posiciones, pesos del meta-agente y
        diagnósticos de aprendizaje, no solo el ganador y el baseline. No cambia qué se ejecuta ni
        cómo se elige: solo cuánto se guarda${state.budget?.retained_run_evidence
          ? ` (${state.budget.retained_run_evidence} runs más con evidencia, ${gigabytes(state.budget.estimated_incremental_bytes)} en total)`
          : ""}.</small></span>
      </label>
      ${stages}<div class="actions"><button id="launch" class="primary">Lanzar Study</button></div>`;
    bindConfiguration();
    document.getElementById("launch").onclick = launch;
    document.getElementById("retain-all").onchange = event => {
      state.retainAllRuns = event.target.checked;
      preflight().then(renderHome).catch(error => notify(error.message, true));
    };
  }

  function budgetMarkup() {
    const budget = state.budget || {};
    const items = [
      ["predictive_evaluations", "Predictivas"], ["expensive_fits", "Fits caros"],
      ["meta_recombinations", "Meta"], ["portfolio_diagnostics", "Carteras"],
      ["profiles", "Perfiles"], ["robustness_groups", "Robustez"],
      ["total_runs", "Runs previstos"], ["estimated_minutes", "Minutos estimados"],
    ];
    const cells = items.map(([key, label]) =>
      `<div><b>${Number.isFinite(Number(budget[key])) ? Math.round(Number(budget[key])) : "—"}</b><span>${label}</span></div>`
    );
    if (budget.retain_all_runs) {
      cells.push(`<div class="retained"><b>${Math.round(Number(budget.retained_run_evidence || 0))}</b><span>Runs con evidencia</span></div>`);
    }
    cells.push(`<div><b>${gigabytes(budget.estimated_incremental_bytes)}</b><span>Disco estimado</span></div>`);
    return `<div class="budget">${cells.join("")}</div>`;
  }

  function bindConfiguration() {
    app.querySelectorAll(".variable-row").forEach(row => {
      const id = row.dataset.variable;
      row.querySelectorAll(`input[name="value-${id}"]`).forEach(input => input.onchange = () => {
        const value = JSON.parse(input.dataset.value);
        const selection = state.definition[id];
        if (input.checked) selection.values.push(value);
        else if (selection.values.length === 1) {
          input.checked = true;
          return notify("Cada variable necesita al menos un valor.", true);
        } else {
          selection.values = selection.values.filter(item => JSON.stringify(item) !== JSON.stringify(value));
          if (JSON.stringify(selection.baseline) === JSON.stringify(value)) selection.baseline = selection.values[0];
        }
        const spec = state.catalog.variables.find(item => item.id === id);
        selection.mode = selection.values.length === 1 ? "fixed" : spec.predictive ? "optimize" : "diagnostic";
        preflight().then(renderHome).catch(error => notify(error.message, true));
      });
      row.querySelectorAll(`input[name="baseline-${id}"]`).forEach(input => input.onchange = () => {
        state.definition[id].baseline = JSON.parse(input.dataset.value);
        preflight().then(renderHome).catch(error => notify(error.message, true));
      });
    });
  }

  async function preflight() {
    const response = await api("/api/studies/preflight", {
      method: "POST", body: JSON.stringify({
        definition: state.definition, run_scope: dev ? "dev" : "full",
        retain_all_runs: state.retainAllRuns,
      }),
    });
    state.definition = response.definition; state.budget = response.budget;
  }

  async function launch() {
    try {
      const payload = {
        name: document.getElementById("study-name").value,
        note: document.getElementById("study-note").value,
        definition: state.definition, run_scope: dev ? "dev" : "full",
        retain_all_runs: state.retainAllRuns,
      };
      const result = await api("/api/studies", {method: "POST", body: JSON.stringify(payload)});
      state.selectedStudy = result.study_id;
      notify(`Study ${result.study_id} creado.`);
      document.querySelector('[data-view="results"]').click();
    } catch (error) { notify(error.message, true); }
  }

  async function renderResults() {
    setBackNavigation();
    clearInterval(state.timer);
    state.studies = await api("/api/studies");
    const now = Date.now();
    const rows = state.studies.map(study => ({
      study_id: study.study_id, nombre: study.name, estado: study.status, etapa: study.phase,
      progreso: `${Math.round((study.progress || 0) * 100)} %`,
      runs_completados: study.completed_runs || 0, runs_restantes: study.runs_remaining || 0,
      tiempo: duration(Math.max(0, now - Date.parse(study.created_at || now))),
      rank_ic_max: study.max_rank_ic ?? "—", actualizado: study.updated_at,
    }));
    app.innerHTML = `<section class="page-title"><p class="eyebrow">Ejecuciones persistentes</p><h2>Studies</h2>
      <p>Selecciona un Study para entrar en su página propia.</p></section>
      <div class="study-table">${studySelectionTable(rows)}</div>`;
    app.querySelectorAll("[data-study]").forEach(button => button.onclick = () => {
      state.selectedStudy = button.dataset.study;
      state.selectedRun = null;
      state.section = "runs";
      renderStudyPage();
    });
  }

  async function renderStudyPage() {
    clearInterval(state.timer);
    setBackNavigation("← Studies", renderResults);
    const study = await api(`/api/studies/${state.selectedStudy}`);
    const total = Number(study.budget?.total_runs || 0);
    const best = Math.max(...study.runs.map(run => Number(run.result?.summary?.mean_rank_ic)).filter(Number.isFinite), -Infinity);
    const active = ["queued", "running"].includes(study.status);
    const resumable = ["failed", "cancelled", "interrupted"].includes(study.status);
    const sectionLabel = {runs: "Runs", decisions: "Decisiones", console: "Consola", robustness: "Robustez", attribution: "Atribución", profiles: "Perfiles"};
    app.innerHTML = `<section class="entity-header run-header">
      <div class="entity-top">
        <div class="entity-main"><div><p class="eyebrow">${esc(study.study_id)}${study.retain_all_runs ? " · evidencia de todos los runs" : ""}</p>
          <h2>${esc(study.name)}</h2><p>${esc(study.note || "Sin nota.")}</p></div></div>
        <div class="entity-actions inline">
          ${active ? '<button id="pause-study">Pausar</button>' : ""}
          ${active ? '<button id="cancel-study">Cancelar</button>' : ""}
          ${resumable ? '<button id="resume-study">Reanudar</button>' : ""}
          <button id="refresh-study">Actualizar</button>
        </div>
      </div>
      <div class="entity-cards">${[
        ["Estado", study.status], ["Etapa", study.phase], ["Progreso", `${Math.round((study.progress || 0) * 100)} %`],
        ["Runs", `${study.completed_runs || 0}/${total}`], ["Tiempo", duration(Date.now() - Date.parse(study.created_at))],
        ["Rank-IC máx.", best > -Infinity ? fmt(best, "rank_ic") : "—"],
      ].map(([label, value]) => `<span><b>${esc(value)}</b>${label}</span>`).join("")}</div>
      <div class="entity-actions">${["runs", "decisions", "console", "robustness", "attribution", "profiles"].map(section =>
        `<button data-study-view="${section}" class="${state.section === section ? "active" : ""}">${esc(sectionLabel[section])}</button>`
      ).join("")}</div></section>
      <section id="study-content"></section>`;
    document.getElementById("refresh-study").onclick = renderStudyPage;
    app.querySelectorAll("[data-study-view]").forEach(button => button.onclick = () => {
      state.section = button.dataset.studyView;
      renderStudyPage();
    });
    if (document.getElementById("pause-study")) document.getElementById("pause-study").onclick = () => studyAction("pause");
    if (document.getElementById("cancel-study")) document.getElementById("cancel-study").onclick = () => studyAction("cancel");
    if (document.getElementById("resume-study")) document.getElementById("resume-study").onclick = () => studyAction("resume");
    await renderStudyContent(study);
  }

  async function studyAction(action) {
    try {
      await api(`/api/studies/${state.selectedStudy}/${action}`, {method: "POST", body: "{}"});
      await renderStudyPage();
    } catch (error) { notify(error.message, true); }
  }

  async function renderStudyContent(study) {
    const body = document.getElementById("study-content");
    if (state.section === "runs") {
      const rows = study.runs.map(run => ({
        run_id: run.run_id, phase: run.phase, variable: run.variable_id, value: run.value,
        status: run.status, progress: `${Math.round((run.progress || 0) * 100)} %`,
        rank_ic: run.result?.summary?.mean_rank_ic,
        alpha_anual: run.result?.summary?.geometric_excess_return,
        alpha_confirmacion: run.result?.confirmation_2025_2026?.mean_rank_ic,
        elapsed_seconds: run.elapsed_seconds,
        source: run.result?.source, error: run.error,
      }));
      body.innerHTML = `<h2>Runs</h2><p class="muted">Cada fila es una evaluación persistente.</p>${runSelectionTable(rows)}`;
      body.querySelectorAll("[data-run]").forEach(button => button.onclick = () => {
        state.selectedRun = button.dataset.run;
        state.runView = "summary";
        renderRunPage();
      });
      return;
    }
    if (state.section === "decisions") {
      body.innerHTML = `<h2>Decisiones</h2><p class="muted">Por cada variable predictiva, el candidato ganador y el porqué: solo Rank-IC robusto entre eras decide, nunca alpha ni información económica.</p>${decisionsView(study.decisions || [])}`;
      return;
    }
    if (state.section === "console") {
      const events = await api(`/api/studies/${state.selectedStudy}/events?after=0`);
      body.innerHTML = `<div class="content-heading"><h2>Consola</h2><button id="refresh-console">Actualizar eventos</button></div>
        <pre class="console">${events.map(event => `${event.sequence} ${madridTime(event.timestamp)} [${event.level}] ${event.message}`).join("\n")}</pre>`;
      const consoleElement = body.querySelector(".console");
      consoleElement.scrollTop = consoleElement.scrollHeight;
      document.getElementById("refresh-console").onclick = async () => {
        const latest = await api(`/api/studies/${state.selectedStudy}/events?after=0`);
        consoleElement.textContent = latest.map(event => `${event.sequence} ${madridTime(event.timestamp)} [${event.level}] ${event.message}`).join("\n");
        consoleElement.scrollTop = consoleElement.scrollHeight;
      };
      return;
    }
    const data = await api(`/api/studies/${state.selectedStudy}/analysis/${state.section}`);
    const views = {profiles, attribution: attributionView, robustness: robustnessView};
    body.innerHTML = (views[state.section] || robustnessView)(data);
    bindInteractiveCharts(body);
  }

  async function renderRunPage() {
    clearInterval(state.timer);
    setBackNavigation("← Study", renderStudyPage);
    const run = await api(`/api/studies/${state.selectedStudy}/runs/${state.selectedRun}`);
    const summary = run.result?.summary || {};
    const views = ["summary", "performance", "learning", "portfolio", "stocks"];
    const runViewLabel = view => ({summary: "Resumen", performance: "Rendimiento", learning: "Aprendizaje", portfolio: "Cartera", stocks: "Acciones"}[view] || columnLabel(view));
    app.innerHTML = `<section class="entity-header run-header"><div class="entity-main">
      <div><p class="eyebrow">${esc(run.phase)} · ${esc(run.variable_id)}</p><h2>${esc(run.run_id)}</h2>
      <p>${esc(run.logical_key)}. Evalúa ${esc(run.value)} dentro de la fase ${esc(run.phase)}.</p></div></div>
      <div class="entity-cards">${[
        ["Estado", run.status], ["Intento", fmt(run.attempt, "attempt")], ["Progreso", `${Math.round((run.progress || 0) * 100)} %`],
        ["Duración", fmt(run.elapsed_seconds, "elapsed_seconds")], ["Rank-IC", fmt(summary.mean_rank_ic, "rank_ic")], ["Fuente", run.result?.source || "—"],
      ].map(([label, value]) => `<span><b>${esc(fmt(value))}</b>${label}</span>`).join("")}</div>
      <div class="entity-actions">${views.map(view => `<button data-run-view="${view}" class="${state.runView === view ? "active" : ""}">${esc(runViewLabel(view))}</button>`).join("")}</div>
      </section><section id="run-content"></section>`;
    app.querySelectorAll("[data-run-view]").forEach(button => button.onclick = () => {
      state.runView = button.dataset.runView;
      renderRunPage();
    });
    await renderRunContent(run);
  }

  async function renderRunContent(run) {
    const body = document.getElementById("run-content");
    const summary = run.result?.summary || {};
    if (state.runView === "summary") {
      body.innerHTML = `<h2>Resumen del run</h2>${groupedMetrics(summary)}<h3>Resultados por era</h3>${table(run.result?.eras || [])}
        <h3>Configuración y evidencia</h3>${configurationCards(run)}
        ${run.error ? `<h3>Error</h3><pre class="error-text report">${esc(run.error)}\n${esc(run.traceback || "")}</pre>` : ""}`;
      return;
    }
    const isWinner = run.logical_key === "winner:evidence";
    const isBaseline = run.logical_key === "predictive:baseline";
    // Con la evidencia de todos los runs activada, un candidato cualquiera conserva la suya y las
    // vistas pesadas se sirven desde ella en lugar de estar vacías.
    const retained = !isWinner && !isBaseline && run.evidence_path ? `run:${run.run_id}` : "";
    if (!isWinner && !isBaseline && !retained) {
      body.innerHTML = `<article class="detail"><h2>Vista no materializada para este candidato</h2>
        <p>Los candidatos normales conservan configuración, Rank-IC, eras y decisión. Las vistas pesadas se guardan únicamente en el run de evidencia del ganador y en el baseline, salvo que el Study se lance guardando la evidencia de todos los runs.</p></article>`;
      return;
    }
    const map = {performance: "portfolio", learning: "learning", portfolio: "portfolio", stocks: "stocks"};
    const sourceQuery = isBaseline ? "source=baseline" : retained ? `source=${encodeURIComponent(retained)}` : "";
    if (state.runView === "stocks") {
      return renderStockBrowser(isBaseline, retained);
    }
    const params = [
      state.runView === "portfolio" && state.snapshot ? `snapshot=${encodeURIComponent(state.snapshot)}` : "",
      sourceQuery,
    ].filter(Boolean).join("&");
    const data = await api(`/api/studies/${state.selectedStudy}/analysis/${map[state.runView]}${params ? `?${params}` : ""}`);
    if (state.runView === "performance") {
      body.innerHTML = `${groupedMetrics(data.summary?.summary || data.summary || {})}${equity(data.equity)}${table(data.annual)}`;
      bindInteractiveCharts(body);
    }
    if (state.runView === "learning") body.innerHTML = `
      <h3>Evolución del Rank-IC por agente</h3>
      ${multiLineChart(data.rank_ic, "prediction_date", "agent", "rank_ic", {percent: true, yLabel: "Rank-IC"})}
      <h3>Evolución de pesos del meta-agente</h3>
      ${multiLineChart(data.weights, "snapshot_date", "agent", "weight", {percent: true, domain: "weight", yLabel: "Peso"})}
      <details><summary>Tabla Rank-IC</summary>${table(data.rank_ic)}</details>
      <details><summary>Tabla de pesos</summary>${table(data.weights)}</details>
      <h3>Evidencia de features</h3>${table(data.features)}`;
    if (state.runView === "learning") bindInteractiveCharts(body);
    if (state.runView === "portfolio") renderPortfolioSnapshot(data);
  }

  function renderPortfolioSnapshot(data) {
    state.snapshot = data.selected_snapshot || state.snapshot;
    const body = document.getElementById("run-content");
    body.innerHTML = `<section class="snapshot-controls"><label>Snapshot de cartera
      <select id="portfolio-snapshot">${(data.available_snapshots || []).map(value => `<option value="${esc(value)}" ${value === data.selected_snapshot ? "selected" : ""}>${esc(value)}</option>`).join("")}</select></label>
      <p>La cartera muestra las posiciones valoradas y las órdenes ejecutadas exactamente en la fecha seleccionada.</p></section>
      <h3>Posiciones · ${esc(data.selected_snapshot || "sin fecha")}</h3>${table(data.positions)}
      <h3>Movimientos ejecutados ese día</h3>${table(data.orders)}
      ${cashSection(data.equity)}
      ${alphaCurveSection(data.alpha_curve)}`;
    document.getElementById("portfolio-snapshot").onchange = event => {
      state.snapshot = event.target.value;
      renderRunPage();
    };
    const windowSelect = document.getElementById("alpha-curve-window");
    if (windowSelect) windowSelect.onchange = event => {
      state.alphaCurveWindow = event.target.value;
      renderRunPage();
    };
  }

  const ALPHA_CURVE_WINDOWS = [
    ["horizon", "Horizonte objetivo"], ["era", "Era (16 trimestres)"],
    ["history", "Todo el histórico"], ["fallback", "Salvaguarda ±10 %"],
  ];

  function cashSection(equity) {
    if (!equity?.length) return "";
    return `<h3>Efectivo por snapshot</h3>
      <p class="muted">Peso en efectivo de la cartera en cada fecha; sale de la misma serie que la curva de capital, no de las posiciones (que solo suman el peso invertido).</p>
      ${singleLineChart(equity, "snapshot_date", "cash_weight", {percent: true, domain: "weight", yLabel: "Efectivo"})}`;
  }

  function alphaCurveSection(curve) {
    if (!curve || !curve.windows) return "";
    const selected = state.alphaCurveWindow && curve.windows[state.alphaCurveWindow]
      ? state.alphaCurveWindow : "horizon";
    const current = curve.windows[selected] || {};
    const fallback = curve.windows.fallback;
    const slope = Number(current.slope);
    const verdict = selected === "fallback"
      ? "Recta impuesta a priori, no estimada de los datos: se aplica solo cuando ninguna de las tres ventanas produce pendiente creciente."
      : !Number.isFinite(slope)
        ? "Sin cohortes suficientes para ajustar una recta en esta ventana."
        : slope > 0
          ? `Pendiente <strong>creciente</strong> (${fmt(slope, "rate")} por ventil): en esta ventana, mejor percentil se tradujo en más alfa.`
          : `Pendiente <strong>decreciente</strong> (${fmt(slope, "rate")} por ventil): en esta ventana el ranking no discriminó a favor, así que la cascada pasa a la siguiente.`;
    return `<h3>Alfa real anualizado por percentil</h3>
      <section class="snapshot-controls"><label>Ventana de datos
        <select id="alpha-curve-window">${ALPHA_CURVE_WINDOWS.map(([value, label]) => `<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></label>
        <p>Cada punto es la media del retorno excedente real ya cerrado, anualizado, de las acciones de ese tramo de <code>meta_rank</code> (ventiles de 5 puntos de percentil, para que cada media tenga muestra suficiente). Las cohortes recientes pesan más que las antiguas. <strong>La recta es la que asigna el alfa</strong>, evaluada en el percentil exacto de cada acción: un p99 recibe más alfa esperado que un p88. La cascada usa la primera ventana con pendiente creciente; si ninguna lo es, se impone la salvaguarda.</p></section>
      <p class="muted">Ventana <strong>${esc(selected)}</strong> · ${esc(current.cohorts || 0)} cohortes cerradas · horizonte ${esc(curve.horizon_months)} meses. ${verdict}</p>
      ${alphaCurveChart(current.points, selected === "fallback" ? null : current, {fallbackLine: fallback})}`;
  }

  const STOCK_VIEWS = [
    ["scores", "Puntuaciones"], ["portfolio", "Situación en cartera"], ["ratios", "Ratios"],
  ];
  let tickerInputTimer = null;

  async function renderStockBrowser(isBaseline, retained = "") {
    state.stockSource = isBaseline ? "baseline" : retained || null;
    const query = new URLSearchParams();
    if (state.stockSource) query.set("source", state.stockSource);
    const initial = await api(`/api/studies/${state.selectedStudy}/analysis/stocks${query.toString() ? `?${query}` : ""}`);
    state.stockTicker = state.stockTicker || initial.available_tickers?.[0] || null;
    if (!state.stockTicker) {
      document.getElementById("run-content").innerHTML = "<p class='muted'>No hay acciones disponibles.</p>";
      return;
    }
    await loadStockBrowser();
  }

  async function loadStockBrowser() {
    const params = new URLSearchParams({ticker: state.stockTicker, view: state.stockView});
    if (state.stockSource) params.set("source", state.stockSource);
    if (state.stockView === "ratios" && state.ratioMode === "detail" && state.selectedRatio) {
      params.set("view", "ratio-detail");
      params.set("ratio", state.selectedRatio);
    } else if (state.stockView === "ratios") {
      params.set("view", "ratios-summary");
      if (state.snapshot) params.set("snapshot", state.snapshot);
    }
    const data = await api(`/api/studies/${state.selectedStudy}/analysis/stocks?${params}`);
    if (state.stockView === "ratios" && state.ratioMode !== "detail") state.snapshot = data.selected_snapshot || state.snapshot;
    const body = document.getElementById("run-content");
    body.innerHTML = `<section class="snapshot-controls stock-controls">
      <label>Acción<input list="stock-ticker-list" id="stock-ticker" value="${esc(data.ticker)}" autocomplete="off"></label>
      <datalist id="stock-ticker-list">${data.available_tickers.map(value => `<option value="${esc(value)}"></option>`).join("")}</datalist>
      <label>Vista<select id="stock-view">${STOCK_VIEWS.map(([id, label]) => `<option value="${id}" ${id === state.stockView ? "selected" : ""}>${label}</option>`).join("")}</select></label>
      ${stockViewControls(data)}
    </section><section id="stock-data">${stockContent(data)}</section>`;
    document.getElementById("stock-ticker").oninput = event => {
      const value = event.target.value.trim().toUpperCase();
      clearTimeout(tickerInputTimer);
      tickerInputTimer = setTimeout(() => {
        if (data.available_tickers.includes(value) && value !== state.stockTicker) { state.stockTicker = value; loadStockBrowser(); }
      }, 250);
    };
    document.getElementById("stock-view").onchange = event => { state.stockView = event.target.value; state.ratioMode = "summary"; state.selectedRatio = null; loadStockBrowser(); };
    if (document.getElementById("stock-snapshot")) document.getElementById("stock-snapshot").onchange = event => { state.snapshot = event.target.value; loadStockBrowser(); };
    if (document.getElementById("ratio-mode")) document.getElementById("ratio-mode").onchange = event => { state.ratioMode = event.target.value; loadStockBrowser(); };
    if (document.getElementById("ratio-select")) document.getElementById("ratio-select").onchange = event => { state.selectedRatio = event.target.value; loadStockBrowser(); };
    bindInteractiveCharts(body);
  }

  function stockViewControls(data) {
    if (state.stockView !== "ratios") return "";
    const mode = state.ratioMode || "summary";
    const modeControl = `<label>Modo<select id="ratio-mode">
      <option value="summary" ${mode === "summary" ? "selected" : ""}>Resumen por snapshot</option>
      <option value="detail" ${mode === "detail" ? "selected" : ""}>Un ratio en detalle</option>
    </select></label>`;
    if (mode === "detail") {
      const selected = state.selectedRatio || data.ratio_options[0]?.id || "";
      state.selectedRatio = state.selectedRatio || selected;
      return `${modeControl}<label>Ratio<select id="ratio-select">${data.ratio_options.map(option => `<option value="${esc(option.id)}" ${option.id === selected ? "selected" : ""}>${esc(option.label)}</option>`).join("")}</select></label>`;
    }
    return `${modeControl}<label>Snapshot<select id="stock-snapshot">${(data.available_snapshots || []).map(value => `<option value="${esc(value)}" ${value === data.selected_snapshot ? "selected" : ""}>${esc(value)}</option>`).join("")}</select></label>`;
  }

  function stockContent(data) {
    if (state.stockView === "scores") return stockScoresView(data);
    if (state.stockView === "portfolio") return stockPortfolioView(data);
    return stockRatiosView(data);
  }

  const AGENT_LABELS = {
    quality: "Quality (calidad)", value: "Value (valoración)", growth: "Growth (crecimiento)",
    momentum: "Momentum (tendencia)", risk: "Risk (riesgo y liquidez)",
    meta_score: "Meta-agente · score", meta_rank: "Meta-agente · rank",
  };

  function stockScoresView(data) {
    const current = data.current || {};
    const rows = ["quality", "value", "growth", "momentum", "risk", "meta_score", "meta_rank"]
      .map(key => ({puntuación: AGENT_LABELS[key], valor: current[key]}));
    const chartRows = normalizeSeries(data.history, "snapshot_date", ["quality", "value", "growth", "momentum", "risk", "meta_score", "price"]);
    return `<h3>Puntuaciones actuales · ${esc(current.snapshot_date || "")}</h3>${table(rows)}
      <h3>Evolución histórica (normalizada, precio en discontinuo)</h3>
      ${multiLineChart(chartRows, "snapshot_date", "series", "value", {domain: "weight", integerAxis: false, yLabel: "Escala 0-1", dashedSeries: ["price"]})}`;
  }

  function stockPortfolioView(data) {
    const status = data.status || {};
    const summary = status.in_portfolio
      ? `<p>En cartera desde <b>${esc(status.entry_date)}</b> (${esc(String(Math.round(status.months_held ?? 0)))} meses), peso actual <b>${fmt(status.weight, "weight")}</b>, percentil <b>${fmt(status.percentile, "current_percentile")}</b> a ${esc(status.as_of)}.</p>`
      : `<p>No está en cartera en el último snapshot (${esc(status.as_of || "sin datos")}).</p>`;
    const events = (data.events || []).map(row => ({
      snapshot_date: row.snapshot_date, side: SIDE_LABELS[row.side] || row.side, reason: REASON_LABELS[row.reason] || row.reason,
      weight_before: row.weight_before, weight_after: row.weight_after, percentile: row.percentile,
      buy_price: row.buy_price, sell_price: row.sell_price, realized_pnl_pct: row.realized_pnl_pct,
    }));
    return `<h3>Situación actual</h3>${summary}
      <h3>Histórico de compras y ventas</h3>${table(events)}
      <h3>Percentil mientras estuvo en cartera</h3>${singleLineChart(data.history, "snapshot_date", "current_percentile", {yLabel: "Percentil"})}`;
  }

  function stockRatiosView(data) {
    if (state.ratioMode === "detail") return stockRatioDetailView(data);
    return `<h3>Puntuaciones de parámetros · ${esc(data.selected_snapshot || "")}</h3>${objectTable(data.scores?.[0] || {})}
      <h3>Valores point-in-time · ${esc(data.selected_snapshot || "")}</h3>${objectTable(data.values?.[0] || {})}`;
  }

  function stockRatioDetailView(data) {
    if (!data.ratio) return "<p class='muted'>Selecciona un ratio.</p>";
    const label = data.ratio_options?.find(option => option.id === data.ratio)?.label || data.ratio;
    const series = [
      ...data.history.map(row => ({snapshot_date: row.snapshot_date, series: label, value: row[data.ratio]})),
      ...(data.components || []).flatMap(component => component.history.map(row => ({snapshot_date: row.snapshot_date, series: component.id, value: row[component.id]}))),
      ...data.price_history.map(row => ({snapshot_date: row.snapshot_date, series: "price", value: row.price})),
    ];
    const normalized = normalizeSeries(series, "snapshot_date", null, "series", "value");
    return `<h3>${esc(label)} · valor actual ${fmt(data.current_value)}</h3>
      <p>Evolución normalizada a escala 0-1 junto con ${data.components.length ? "sus componentes y " : ""}el precio (línea discontinua), para comparar patrones.</p>
      ${multiLineChart(normalized, "snapshot_date", "series", "value", {domain: "weight", yLabel: "Escala 0-1", dashedSeries: ["price"]})}`;
  }

  function normalizeSeries(rows, xKey, valueKeys, seriesKey, valueKey) {
    let long = rows || [];
    if (valueKeys) {
      long = (rows || []).flatMap(row => valueKeys.filter(key => row[key] != null).map(key => ({[xKey]: row[xKey], series: key, value: Number(row[key])})));
      seriesKey = "series"; valueKey = "value";
    }
    const bySeries = new Map();
    long.forEach(row => {
      const name = row[seriesKey];
      if (row[valueKey] == null || !Number.isFinite(Number(row[valueKey]))) return;
      if (!bySeries.has(name)) bySeries.set(name, []);
      bySeries.get(name).push(Number(row[valueKey]));
    });
    const ranges = new Map([...bySeries].map(([name, values]) => [name, [Math.min(...values), Math.max(...values)]]));
    return long
      .filter(row => row[valueKey] != null && Number.isFinite(Number(row[valueKey])))
      .map(row => {
        const [minimum, maximum] = ranges.get(row[seriesKey]);
        const span = maximum - minimum;
        return {...row, [xKey]: row[xKey], [seriesKey]: row[seriesKey], [valueKey]: span > 0 ? (Number(row[valueKey]) - minimum) / span : 0.5};
      });
  }

  function profiles(data) {
    const names = data.comparison.map(row => row.profile);
    const byKey = new Map(data.annual.map(row => [`${row.year}:${row.profile}`, row.alpha]));
    const years = [...new Set(data.annual.map(row => row.year))].sort();
    return `<h3>Alfa anual por perfil</h3>
      ${multiLineChart(data.annual, "year", "profile", "alpha", {percent: true, yLabel: "Alfa vs SPY"})}
      <h3>Comparación agregada</h3>${table(data.comparison)}<h3>Alfa anual frente a SPY</h3>
      <div class="table-wrap"><table><thead><tr><th>Año</th>${names.map(name => `<th>${esc(name)}</th>`).join("")}</tr></thead>
      <tbody>${years.map(year => `<tr><th>${year}</th>${names.map(name => { const value = byKey.get(`${year}:${name}`); return `<td class="${tone(value, "alpha")}">${fmt(value, "alpha")}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function robustnessView(data) {
    const observed = data.permutation?.observed_mean_rank_ic;
    const comparisons = [
      {label: "Modelo · seed 42", value: observed},
      ...(data.seeds || []).map(row => ({label: `Seed ${row.seed}`, value: row.summary?.mean_rank_ic})),
      ...(data.label_placebos || []).map(row => ({label: `Placebo ${row.seed}`, value: row.summary?.mean_rank_ic})),
    ];
    const agents = (data.agent_rank_ic || []).map(row => ({label: row.agent, value: row.mean_rank_ic}));
    const dispersion = data.seed_dispersion || {};
    const alpha = dispersion.geometric_excess_return;
    return `<h2>Robustez informativa</h2>
      ${metrics({
        observed_mean_rank_ic: observed,
        permutation_p_value: data.permutation?.p_value,
        rank_ic_bootstrap_90_low: data.bootstrap_and_era_exclusion?.interval_90?.ci_low,
        rank_ic_bootstrap_90_high: data.bootstrap_and_era_exclusion?.interval_90?.ci_high,
        meta_weight_concentration: data.meta_weight_stability?.mean_concentration,
        meta_weight_turnover: data.meta_weight_stability?.mean_turnover,
      })}
      ${alpha ? `<h3>Estabilidad entre semillas</h3>
        <p class="muted">El Rank-IC apenas depende de la semilla, pero una cartera concentrada puede
        amplificar el ruido de inicialización hasta cambiar el signo del alfa. Este rango es la
        prueba directa de que la conclusión económica es reproducible.</p>
        <div class="verdict ${alpha.crosses_zero ? "bad" : "good"}">
          ${alpha.crosses_zero
            ? "El alfa cambia de signo entre semillas: la conclusión económica NO es estable."
            : "El alfa mantiene el signo en todas las semillas evaluadas."}
        </div>
        ${barChart([
          {label: "Mínimo", value: alpha.min}, {label: "Mediana", value: alpha.median},
          {label: "Máximo", value: alpha.max},
        ], {percent: true, yLabel: "Alfa geométrico"})}` : ""}
      <h3>Modelo, semillas y placebos</h3>${barChart(comparisons, {percent: true, yLabel: "Rank-IC"})}
      <h3>Rank-IC medio por agente</h3>${barChart(agents, {percent: true, yLabel: "Rank-IC"})}
      <h3>Nulo de carteras aleatorias</h3>
      <p class="muted">Carteras del mismo tamaño elegidas al azar, con la misma guarda de datos y
      pagando las mismas comisiones que el modelo.</p>
      ${metrics({
        model_cagr: data.random_portfolios?.general?.model_cagr,
        random_median: data.random_portfolios?.general?.random_median,
        random_p95: data.random_portfolios?.general?.random_p95,
        model_percentile: data.random_portfolios?.general?.model_percentile,
      })}
      <details><summary>Resultado completo de robustez</summary>${objectTree(data)}</details>`;
  }
  function attributionView(data) {
    const selection = data.factor_regression?.selection || {};
    const confirmation = data.confirmation_2025_2026 || {};
    const neutral = data.neutralized_rank_ic || {};
    const loadings = Object.entries(selection.loadings || {}).map(([label, value]) => ({label, value}));
    const baselines = (data.baselines?.baselines || []).map(row => ({label: row.baseline, value: row.mean_rank_ic}));
    const alphaT = selection.alpha_t_stat;
    return `<h2>Atribución del alfa</h2>
      <p class="muted">Responde a la pregunta decisiva: ¿el sistema aprende una ordenación propia o
      reproduce un factor de estilo ya conocido? El alfa es el intercepto de la regresión del exceso
      de la cartera sobre réplicas de valor, momentum, baja volatilidad, calidad y crecimiento
      construidas con el mismo panel point-in-time.</p>
      ${metrics({
        alpha_per_period: selection.alpha_per_period, alpha_t_stat: alphaT,
        r_squared: selection.r_squared,
        raw_mean_rank_ic: neutral.raw_mean_rank_ic,
        neutralized_mean_rank_ic: neutral.neutralized_mean_rank_ic,
        retained_fraction: neutral.retained_fraction,
      })}
      ${Number.isFinite(alphaT) ? `<div class="verdict ${Math.abs(alphaT) > 2 ? "good" : "bad"}">
        ${Math.abs(alphaT) > 2
          ? "El alfa sobrevive al control por factores de estilo (|t| > 2): el sistema aporta ordenación propia."
          : "El alfa no es distinguible de la exposición a factores conocidos (|t| ≤ 2)."}
      </div>` : ""}
      <h3>Exposición a factores de estilo</h3>
      ${loadings.length ? barChart(loadings, {yLabel: "Beta"}) : '<p class="muted">Sin réplicas disponibles.</p>'}
      <h3>Confirmación fuera de muestra 2025-2026</h3>
      <p class="muted">Esta era no participó en ninguna decisión del Study: es la única medida
      predictiva libre de sesgo de selección. Con horizonte de 12 meses las cohortes se solapan, así
      que se declara también el número de observaciones realmente independientes.</p>
      ${metrics({
        mean_rank_ic: confirmation.mean_rank_ic,
        rank_ic_positive_fraction: confirmation.rank_ic_positive_fraction,
        n_cohorts: confirmation.n_cohorts,
        effective_independent_observations: data.ic_significance?.confirmation?.effective_independent_observations,
      })}
      <h3>Corrección por multiplicidad</h3>
      <p class="muted">Con muchas configuraciones probadas, el mejor resultado es alto aunque ninguna
      tenga capacidad real. El Deflated Sharpe descuenta ese efecto.</p>
      ${metrics({
        deflated_sharpe_probability: data.deflated_sharpe?.deflated_sharpe_probability,
        n_trials: data.deflated_sharpe?.n_trials,
        observed_sharpe: data.deflated_sharpe?.observed_sharpe_per_period,
        expected_max_under_null: data.deflated_sharpe?.expected_max_sharpe_under_null,
      })}
      <h3>Baselines sin aprendizaje</h3>
      <p class="muted">Reglas deterministas (GARP, momentum puro, calidad, valor). Si el sistema no
      las supera, el aparato de aprendizaje no está justificado.</p>
      ${baselines.length ? barChart(baselines, {percent: true, yLabel: "Rank-IC"}) : '<p class="muted">No disponibles.</p>'}
      <details><summary>Atribución completa</summary>${objectTree(data)}</details>`;
  }
  function metrics(data) {
    const priority = ["mean_rank_ic", "ic_ir", "rank_ic_positive_fraction", "geometric_excess_return", "cagr_portfolio", "cagr_benchmark", "information_ratio", "max_drawdown", "annualized_turnover", "mean_cash_weight"];
    const entries = [
      ...priority.filter(key => data[key] !== undefined).map(key => [key, data[key]]),
      ...Object.entries(data).filter(([key]) => !priority.includes(key)),
    ].slice(0, 10);
    return `<div class="metrics">${entries.map(([key, value]) => `<div><b class="${tone(value, key)}">${esc(fmt(value, key))}</b><span>${esc(metricLabel(key))}</span></div>`).join("")}</div>`;
  }
  const SUMMARY_METRIC_GROUPS = [
    {title: "Predicción", keys: ["mean_rank_ic", "ic_ir", "rank_ic_positive_fraction"]},
    {title: "Rendimiento vs benchmark", keys: ["geometric_excess_return", "cagr_portfolio", "cagr_benchmark", "information_ratio"]},
    {title: "Ejecución de cartera", keys: ["max_drawdown", "annualized_turnover", "mean_cash_weight"]},
  ];
  function groupedMetrics(data) {
    const groups = SUMMARY_METRIC_GROUPS
      .map(group => ({...group, keys: group.keys.filter(key => data[key] !== undefined)}))
      .filter(group => group.keys.length);
    if (!groups.length) return "<p class='muted'>Sin métricas.</p>";
    return `<div class="metric-groups">${groups.map(group => `
      <article class="metric-group" style="flex-grow:${group.keys.length}">
        <h4>${esc(group.title)}</h4>
        <div class="metric-group-items">${group.keys.map(key => `
          <div><b class="${tone(data[key], key)}">${esc(fmt(data[key], key))}</b><span>${esc(metricLabel(key))}</span></div>
        `).join("")}</div>
      </article>`).join("")}</div>`;
  }
  const METRIC_LABELS = {
    geometric_excess_return: "Alfa geométrico vs SPY", cagr_portfolio: "CAGR cartera",
    cagr_benchmark: "CAGR SPY", mean_rank_ic: "Rank-IC medio", ic_ir: "IC-IR",
    rank_ic_positive_fraction: "Cohortes Rank-IC positivas", annualized_turnover: "Turnover anualizado",
    max_drawdown: "Drawdown máximo", information_ratio: "Information Ratio (anualizado)",
    mean_cash_weight: "Efectivo medio", transfer_coefficient: "Coeficiente de transferencia",
    exit_expected_alpha_bps: "Umbral de salida (pb/año)", rotation_edge_bps: "Ventaja de rotación (pb/año)",
    total_cost_drag: "Coste acumulado", n_cohorts: "Cohortes", beat_rate: "Años por encima de SPY",
    tail_spread: "Diferencial decil superior", target_size: "Posiciones",
    observed_mean_rank_ic: "Rank-IC observado", permutation_p_value: "p-valor de permutación",
    rank_ic_bootstrap_90_low: "Bootstrap 90 % inferior", rank_ic_bootstrap_90_high: "Bootstrap 90 % superior",
    meta_weight_concentration: "Concentración de pesos del meta", meta_weight_turnover: "Rotación de pesos del meta",
    deflated_sharpe_probability: "Probabilidad Deflated Sharpe", alpha_t_stat: "t de Newey-West del alfa",
    neutralized_mean_rank_ic: "Rank-IC neutralizado", raw_mean_rank_ic: "Rank-IC bruto",
    retained_fraction: "Fracción de señal retenida", n_trials: "Configuraciones probadas",
    effective_independent_observations: "Observaciones independientes",
  };
  function metricLabel(key) { return METRIC_LABELS[key] || key; }
  const COLUMN_LABELS = {
    nombre: "Nombre", estado: "Estado", etapa: "Etapa", progreso: "Progreso",
    runs_completados: "Runs completados", runs_restantes: "Runs restantes", tiempo: "Tiempo",
    rank_ic_max: "Rank-IC máximo", actualizado: "Actualizado", study_id: "Study",
    run_id: "Run", phase: "Fase", variable: "Variable", value: "Valor", status: "Estado",
    progress: "Progreso", rank_ic: "Rank-IC", alpha_anual: "Alfa geométrico",
    alpha_confirmacion: "Rank-IC 2025-26",
    elapsed_seconds: "Duración", source: "Origen", error: "Error", winner_value: "Valor ganador",
    selection_rule: "Regla de selección", median_era_rank_ic: "Rank-IC mediano por era",
    paired_advantage: "Ventaja pareada",
    mean_rank_ic: "Rank-IC medio", positive_fraction: "Fracción positiva", rank_ic_std: "Desviación Rank-IC",
    observations: "Observaciones", reason: "Motivo", eligible: "Elegible", is_incumbent: "Baseline",
    candidate_id: "Candidato", max_cash_weight: "Efectivo máximo", mean_cash_weight: "Efectivo medio",
    coverage_percentile_floor: "Suelo de cobertura",
    geometric_excess_return: "Alfa geométrico vs SPY", ic_ir: "IC-IR",
    transfer_coefficient: "Coeficiente de transferencia", months_held: "Meses en cartera",
    current_percentile: "Percentil", percentile: "Percentil", meta_rank: "Puntuación meta-rank", weight: "Peso", entry_date: "Fecha de entrada",
    ticker: "Ticker", snapshot_date: "Snapshot", side: "Sentido", weight_before: "Peso anterior",
    weight_after: "Peso nuevo", entry_price: "Precio de compra", valuation_price: "Precio actual",
    buy_price: "Precio de compra", sell_price: "Precio de venta", realized_pnl_pct: "P&L realizado neto",
    unrealized_pnl_pct: "P&L no realizado", notional: "Nocional", commission_amount: "Comisión", slippage_amount: "Slippage",
  };
  const SIDE_LABELS = {buy: "Compra", sell: "Venta"};
  const REASON_LABELS = {
    initial_fill: "Compra inicial", fully_invested_fill: "Relleno (100 % invertido)",
    cash_floor_fill: "Relleno (suelo de diversificación)", expected_alpha_below_exit: "Alfa esperado bajo umbral",
    displaced_by_net_edge: "Desplazada por rotación", net_edge_over_worst: "Rotación (ventaja de alfa)",
    rebalance: "Reequilibrio de peso", missing_current_score: "Sin puntuación actual",
    below_coverage_percentile: "Bajo el suelo de cobertura",
  };
  function columnLabel(key) {
    if (COLUMN_LABELS[key]) return COLUMN_LABELS[key];
    const spaced = String(key).replace(/_/g, " ");
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }
  function decisionsView(decisions) {
    if (!decisions.length) return "<p class='muted'>Todavía no hay decisiones registradas: el Study aún no ha comparado ninguna variable.</p>";
    const specs = new Map((state.catalog?.variables || []).map(spec => [spec.id, spec]));
    return decisions.map(decision => {
      const spec = specs.get(decision.variable_id);
      const label = spec?.label || columnLabel(decision.variable_id);
      const valueLabel = value => spec?.value_options?.find(option => JSON.stringify(option.value) === JSON.stringify(value))?.label ?? String(value);
      const ruleText = {
        robust_rank_ic: "Ganó por tener la mayor ventaja pareada de Rank-IC contra el baseline entre los candidatos elegibles. La diferencia se mide cohorte a cohorte, lo que elimina el factor común de mercado de cada fecha y reduce mucho la varianza frente a comparar medias.",
        tie_simplicity: "Empate técnico: la ventaja pareada frente al baseline queda por debajo de la tolerancia, así que la diferencia no se distingue del ruido y gana la opción más simple.",
        incumbent_no_eligible_challenger: "Ningún candidato alternativo superó los filtros de elegibilidad (observaciones suficientes, suelo de Rank-IC por era y puerta pareada); se mantiene el valor ya vigente.",
      }[decision.selection_rule] || decision.selection_rule;
      const rows = [...decision.candidates].sort((a, b) => (a.candidate_id === decision.winner_candidate_id ? -1 : b.candidate_id === decision.winner_candidate_id ? 1 : 0));
      const gateCell = candidate => {
        const gates = candidate.gates || {};
        if (!gates.paired_applicable && !candidate.is_incumbent) return '<span class="warn">No aplicable</span>';
        if (gates.paired_dominates_incumbent) return '<span class="positive">Domina</span>';
        if (gates.paired_bootstrap_non_inferior) return "No inferior";
        return '<span class="negative">Inferior</span>';
      };
      return `<article class="config-card decision-card">
        <h4>${esc(label)} <span class="decision-winner">→ ${esc(valueLabel(decision.winner_value))}</span></h4>
        <p class="muted">${esc(ruleText)}</p>
        <div class="table-wrap"><table><thead><tr>
          <th>Valor</th><th>Rank-IC medio</th><th>Ventaja pareada</th><th>% fechas mejor</th>
          <th>Puerta pareada</th><th>Observaciones</th><th>Elegible</th><th>Motivo</th>
        </tr></thead><tbody>${rows.map(candidate => `
          <tr class="${candidate.candidate_id === decision.winner_candidate_id ? "selected-row" : ""}">
            <td>${esc(valueLabel(candidate.value))}${candidate.is_incumbent ? " <small>(baseline)</small>" : ""}</td>
            <td class="${tone(candidate.mean_rank_ic, "rank_ic")}">${esc(fmt(candidate.mean_rank_ic, "rank_ic"))}</td>
            <td class="${tone(candidate.paired_advantage, "rank_ic")}">${esc(fmt(candidate.paired_advantage, "rank_ic"))}</td>
            <td>${esc(fmt(candidate.paired_bootstrap_90?.fraction_a_better, "fraction"))}</td>
            <td>${gateCell(candidate)}</td>
            <td>${esc(fmt(candidate.observations))}</td>
            <td>${candidate.eligible ? "Sí" : "No"}</td>
            <td><small>${esc(candidate.reason)}</small></td>
          </tr>`).join("")}</tbody></table></div>
      </article>`;
    }).join("");
  }
  function configurationCards(run) {
    const configuration = run.configuration || {};
    const specs = new Map((state.catalog?.variables || []).map(spec => [spec.id, spec]));
    const groups = [
      ["Temporal", ["snapshot_step_months", "target_horizon_months", "train_lookback_years", "execution_lag_days", "recency_weighting", "objective"]],
      ["Features y representación", [...specs.values()].filter(spec => spec.stage === "representation").map(spec => spec.id)],
      ["Modelo", [...specs.values()].filter(spec => spec.stage === "model").map(spec => spec.id)],
      ["Meta-agente", [...specs.values()].filter(spec => spec.stage === "meta").map(spec => spec.id)],
      ["Cartera base", [...specs.values()].filter(spec => spec.stage === "portfolio").map(spec => spec.id)],
    ];
    const cards = groups.map(([title, keys]) => {
      const rows = keys.filter(key => configuration[key] !== undefined).map(key => {
        const spec = specs.get(key);
        return `<li><span>${esc(spec?.label || key)}</span><b>${esc(fmt(configuration[key], key))}</b></li>`;
      });
      return rows.length ? `<article class="config-card"><h4>${esc(title)}</h4><ul>${rows.join("")}</ul></article>` : "";
    });
    const result = run.result || {};
    cards.splice(3, 0, `<article class="config-card"><h4>Agentes activos</h4><ul>${[
      ["Quality", "Calidad y rentabilidad"], ["Value", "Valoración"], ["Growth", "Crecimiento"],
      ["Momentum", "Tendencia de precios"], ["Risk", "Riesgo y liquidez"],
    ].map(([name, description]) => `<li><span>${esc(name)}</span><b>${esc(description)}</b></li>`).join("")}</ul></article>`);
    const artifacts = [
      ["Fuente", result.source], ["Dataset", result.dataset_hash], ["Clave de evaluación", result.evaluation_key],
      ["Tiempo de cálculo", result.elapsed_seconds == null ? null : fmt(result.elapsed_seconds, "elapsed_seconds")],
    ].filter(([, value]) => value != null);
    if (artifacts.length) cards.push(`<article class="config-card artifact-card"><h4>Artefactos</h4><ul>${artifacts.map(([label, value]) => `<li><span>${esc(label)}</span><b>${esc(String(value))}</b></li>`).join("")}</ul></article>`);
    return `<div class="config-card-grid">${cards.join("") || "<p class='muted'>Sin configuración persistida.</p>"}</div>`;
  }
  function table(rows) {
    if (!rows?.length) return "<p class='muted'>Sin datos.</p>";
    const columns = Object.keys(rows[0]).slice(0, 14);
    return `<div class="table-wrap"><table><thead><tr>${columns.map(key => `<th>${esc(columnLabel(key))}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 500).map(row => `<tr>${columns.map(key => `<td class="${tone(row[key], key)}">${esc(fmt(typeof row[key] === "object" ? JSON.stringify(row[key]) : row[key], key))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function studySelectionTable(rows) {
    if (!rows.length) return "<p class='muted'>No hay Studies.</p>";
    const columns = ["nombre", "estado", "etapa", "progreso", "runs_completados", "runs_restantes", "tiempo", "rank_ic_max", "actualizado"];
    return `<table><thead><tr>${columns.map(key => `<th>${esc(columnLabel(key))}</th>`).join("")}</tr></thead><tbody>${rows.map(row =>
      `<tr class="${row.study_id === state.selectedStudy ? "selected-row" : ""}">${columns.map(key => `<td class="${tone(row[key], key)}">${esc(display(row[key], key))}</td>`).join("")}
      <td class="action-cell"><button data-study="${esc(row.study_id)}">Abrir</button></td></tr>`).join("")}</tbody></table>`;
  }
  function runSelectionTable(rows) {
    if (!rows.length) return "<p class='muted'>Todavía no hay runs.</p>";
    const columns = ["phase", "variable", "value", "status", "progress", "rank_ic", "alpha_anual", "alpha_estres", "elapsed_seconds", "source", "error"];
    return `<table class="runs-table"><thead><tr>${columns.map(key => `<th>${esc(columnLabel(key))}</th>`).join("")}<th></th></tr></thead><tbody>${rows.map(row =>
      `<tr class="${row.run_id === state.selectedRun ? "selected-row" : ""}">${columns.map(key => `<td class="${tone(row[key], key)}">${esc(display(row[key], key))}</td>`).join("")}
      <td class="action-cell"><button data-run="${esc(row.run_id)}">Ver run</button></td></tr>`).join("")}</tbody></table>`;
  }
  function duration(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours} h ${minutes} min`;
  }
  function formatElapsedSeconds(totalSeconds) {
    if (typeof totalSeconds !== "number" || !Number.isFinite(totalSeconds)) return "—";
    const seconds = Math.floor(totalSeconds);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    const parts = [];
    if (hours) parts.push(`${hours} h`);
    if (hours || minutes) parts.push(`${minutes} min`);
    parts.push(`${secs} s`);
    return parts.join(" ");
  }
  function objectTable(data) { return table(Object.entries(data).map(([key, value]) => ({variable: key, value: typeof value === "object" ? JSON.stringify(value) : value}))); }
  function objectTree(data) { return `<pre class="report">${esc(JSON.stringify(data, null, 2))}</pre>`; }
  const chartColors = ["#57a6ff", "#ff8a65", "#ffd166", "#7bd389", "#c792ea", "#55d6be", "#f06292", "#a8dadc", "#ffb74d", "#9fa8da"];

  function multiLineChart(rows, xKey, seriesKey, yKey, options = {}) {
    return lineChart(rows, xKey, seriesKey, yKey, options);
  }

  function singleLineChart(rows, xKey, yKey, options = {}) {
    return lineChart((rows || []).map(row => ({...row, _series: options.yLabel || yKey})), xKey, "_series", yKey, options);
  }

  function lineChart(rows, xKey, seriesKey, yKey, options = {}) {
    const clean = (rows || []).filter(row => row[xKey] != null && row[seriesKey] != null && Number.isFinite(Number(row[yKey])));
    if (!clean.length) return "<p class='muted'>Sin datos suficientes para dibujar el gráfico.</p>";
    const xValues = [...new Set(clean.map(row => String(row[xKey])))].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
    const names = [...new Set(clean.map(row => String(row[seriesKey])))];
    const lookup = new Map(clean.map(row => [`${row[seriesKey]}:${row[xKey]}`, Number(row[yKey])]));
    const width = 920, height = 360, left = 78, right = 26, top = 24, bottom = 58;
    const innerWidth = width - left - right, innerHeight = height - top - bottom;
    const rawDomain = chartDomain(clean.map(row => Number(row[yKey])), options.domain);
    const ticks = niceTicks(rawDomain.minimum, rawDomain.maximum, 6, options.integerAxis);
    const minimum = ticks[0], maximum = ticks[ticks.length - 1];
    const x = index => left + index * innerWidth / Math.max(xValues.length - 1, 1);
    const y = value => top + (maximum - value) * innerHeight / (maximum - minimum);
    const series = names.map((name, index) => ({
      name, color: chartColors[index % chartColors.length], values: xValues.map(value => lookup.get(`${name}:${value}`) ?? null),
    }));
    const dashedSeries = new Set(options.dashedSeries || []);
    const paths = series.map(item => {
      const points = item.values.map((value, index) => value == null ? null : `${x(index)},${y(value)}`);
      const segments = []; let segment = [];
      points.forEach(point => { if (point) segment.push(point); else if (segment.length) { segments.push(segment); segment = []; } });
      if (segment.length) segments.push(segment);
      const dash = dashedSeries.has(item.name) ? ";stroke-dasharray:4 3" : "";
      return segments.map(values => `<polyline points="${values.join(" ")}" style="stroke:${item.color}${dash}"></polyline>`).join("");
    }).join("");
    const years = yearMarkers(xValues);
    const payload = encodeURIComponent(JSON.stringify({xValues, series, percent: Boolean(options.percent), width, left, right, top, bottom}));
    return `<div class="chart-card interactive-card" data-chart="${payload}"><div class="chart-tooltip hidden"></div>
      <svg class="chart analytic-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.yLabel || yKey)}">
      <g class="chart-grid">${ticks.map(value => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}"></line><text x="${left - 10}" y="${y(value) + 4}" text-anchor="end">${esc(axisFormat(value, options))}</text>`).join("")}
      ${years.map(index => `<line class="year-divider" x1="${x(index)}" y1="${top}" x2="${x(index)}" y2="${height - bottom}"></line><text x="${x(index)}" y="${height - bottom + 24}" text-anchor="middle">${esc(yearLabel(xValues[index]))}</text>`).join("")}
      <line class="axis-line" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"></line><line class="axis-line" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line></g>
      ${paths}<line class="chart-cursor hidden" y1="${top}" y2="${height - bottom}"></line>
      <rect class="chart-hit" x="${left}" y="${top}" width="${innerWidth}" height="${innerHeight}"></rect>
      <text class="axis-title" transform="translate(16 ${top + innerHeight / 2}) rotate(-90)" text-anchor="middle">${esc(options.yLabel || yKey)}</text></svg>
      <div class="chart-legend">${series.map(item => `<span><i style="background:${item.color}"></i>${esc(item.name)}</span>`).join("")}</div></div>`;
  }

  // Curva ventil -> alfa real anualizado. A diferencia de `lineChart`, el eje X es NUMÉRICO y hay
  // dos capas: los puntos observados (media por ventil) y la recta ajustada, que es la que de verdad
  // asigna el alfa de cada acción según su percentil exacto. La salvaguarda se dibuja discontinua
  // para poder compararla de un vistazo con los puntos reales.
  const VENTILES = 20;
  function alphaCurveChart(points, line, options = {}) {
    const clean = (points || []).filter(row => Number.isFinite(Number(row.alpha_annual)));
    const fallback = options.fallbackLine;
    if (!clean.length && !fallback) return "<p class='muted'>Sin cohortes cerradas suficientes para dibujar la curva.</p>";
    const last = VENTILES - 1;
    const width = 920, height = 360, left = 78, right = 26, top = 24, bottom = 58;
    const innerWidth = width - left - right, innerHeight = height - top - bottom;
    const lineAt = (fit, ventile) => fit.slope * ventile + fit.intercept;
    const candidates = clean.map(row => Number(row.alpha_annual));
    [line, fallback].forEach(fit => {
      if (fit && Number.isFinite(fit.slope)) candidates.push(lineAt(fit, 0), lineAt(fit, last));
    });
    const ticks = niceTicks(...Object.values(chartDomain(candidates)), 6);
    const minimum = ticks[0], maximum = ticks[ticks.length - 1];
    const x = ventile => left + ventile * innerWidth / last;
    const y = value => top + (maximum - value) * innerHeight / (maximum - minimum);
    // El ventil v agrupa los percentiles [v·5, v·5+5): se etiqueta por su centro.
    const percentileOf = ventile => Math.round(ventile * (100 / VENTILES) + 2.5);
    const segment = (fit, color, dashed) => {
      if (!fit || !Number.isFinite(fit.slope)) return "";
      const dash = dashed ? ";stroke-dasharray:6 4" : "";
      return `<polyline points="${x(0)},${y(lineAt(fit, 0))} ${x(last)},${y(lineAt(fit, last))}" style="stroke:${color}${dash}"></polyline>`;
    };
    const dots = clean.map(row => {
      const value = Number(row.alpha_annual);
      return `<circle cx="${x(Number(row.ventile))}" cy="${y(value)}" r="4.5" style="fill:${chartColors[0]}"><title>~p${esc(percentileOf(Number(row.ventile)))} · ${esc(fmt(value, "rate"))} (n=${esc(row.observations)})</title></circle>`;
    }).join("");
    const zero = minimum <= 0 && maximum >= 0
      ? `<line class="axis-line" x1="${left}" y1="${y(0)}" x2="${width - right}" y2="${y(0)}"></line>` : "";
    const legend = [
      clean.length ? `<span><i style="background:${chartColors[0]}"></i>Alfa real observado</span>` : "",
      line && Number.isFinite(line.slope) ? `<span><i style="background:${chartColors[3]}"></i>Recta ajustada (la que asigna el alfa)</span>` : "",
      fallback ? `<span><i style="background:${chartColors[1]}"></i>Salvaguarda ±10 %</span>` : "",
    ].filter(Boolean).join("");
    return `<div class="chart-card"><svg class="chart analytic-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Alfa anualizado por percentil">
      <g class="chart-grid">${ticks.map(value => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}"></line><text x="${left - 10}" y="${y(value) + 4}" text-anchor="end">${esc(fmt(value, "rate"))}</text>`).join("")}
      ${[0, 4, 8, 12, 16, last].map(ventile => `<text x="${x(ventile)}" y="${height - bottom + 24}" text-anchor="middle">p${percentileOf(ventile)}</text>`).join("")}
      <line class="axis-line" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"></line><line class="axis-line" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line></g>
      ${zero}${segment(fallback, chartColors[1], true)}${segment(line, chartColors[3], false)}${dots}
      <text class="axis-title" transform="translate(16 ${top + innerHeight / 2}) rotate(-90)" text-anchor="middle">Alfa anualizado</text>
      <text class="axis-title" x="${left + innerWidth / 2}" y="${height - 8}" text-anchor="middle">Percentil de meta_rank</text></svg>
      <div class="chart-legend">${legend}</div></div>`;
  }

  function chartDomain(values, kind = "auto") {
    let minimum = Math.min(...values), maximum = Math.max(...values);
    if (minimum === maximum) {
      const delta = kind === "weight" ? 0.05 : Math.max(Math.abs(minimum) * 0.08, 0.01);
      minimum -= delta; maximum += delta;
    }
    const pad = (maximum - minimum) * (kind === "weight" ? 0.10 : 0.08);
    if (kind === "weight") {
      return {
        minimum: Math.max(0, minimum - pad),
        maximum: Math.min(1, maximum + pad),
      };
    }
    return {minimum: minimum - pad, maximum: maximum + pad};
  }

  function niceTicks(minimum, maximum, count, integers = false) {
    const rawStep = (maximum - minimum) / Math.max(count - 1, 1);
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(rawStep, 1e-9)));
    const normalized = rawStep / magnitude;
    const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
    const start = Math.floor(minimum / step) * step;
    const end = Math.ceil(maximum / step) * step;
    const values = [];
    for (let value = start; value <= end + step / 1000; value += step) values.push(integers ? Math.round(value) : Number(value.toPrecision(12)));
    return values;
  }
  function axisFormat(value, options) { return options.percent ? fmt(value, "rate") : options.integerAxis ? String(Math.round(value)) : fmt(value); }
  function yearLabel(value) { return /^\d{4}/.test(String(value)) ? String(value).slice(0, 4) : String(value); }
  function yearMarkers(values) {
    const indexes = [];
    let previous = null;
    values.forEach((value, index) => { const year = yearLabel(value); if (year !== previous) indexes.push(index); previous = year; });
    return indexes;
  }

  function bindInteractiveCharts(root) {
    root.querySelectorAll(".interactive-card[data-chart]").forEach(card => {
      if (card.dataset.bound) return;
      card.dataset.bound = "true";
      const data = JSON.parse(decodeURIComponent(card.dataset.chart));
      const svg = card.querySelector("svg"), hit = card.querySelector(".chart-hit"), cursor = card.querySelector(".chart-cursor"), tooltip = card.querySelector(".chart-tooltip");
      hit.onmousemove = event => {
        // Con `max-height`, el SVG puede quedar centrado dentro de una caja más ancha que su
        // área dibujada. Convertir las coordenadas de pantalla al `viewBox` evita que el cursor
        // avance más despacio que el ratón en pantalla completa.
        const matrix = svg.getScreenCTM();
        if (!matrix) return;
        const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
        const index = Math.max(0, Math.min(data.xValues.length - 1, Math.round((point.x - data.left) / ((data.width - data.left - data.right) / Math.max(data.xValues.length - 1, 1)))));
        const x = data.left + index * (data.width - data.left - data.right) / Math.max(data.xValues.length - 1, 1);
        cursor.setAttribute("x1", x); cursor.setAttribute("x2", x); cursor.classList.remove("hidden");
        tooltip.innerHTML = `<b>${esc(data.xValues[index])}</b>${data.series.map(item => item.values[index] == null ? "" : `<span><i style="background:${item.color}"></i>${esc(item.name)} <strong>${esc(data.percent ? fmt(item.values[index], "rate") : fmt(item.values[index]))}</strong></span>`).join("")}`;
        tooltip.classList.remove("hidden");
        const cardBounds = card.getBoundingClientRect();
        tooltip.style.left = `${Math.min(card.clientWidth - 205, Math.max(8, event.clientX - cardBounds.left + 14))}px`;
        tooltip.style.top = "14px";
      };
      hit.onmouseleave = () => { cursor.classList.add("hidden"); tooltip.classList.add("hidden"); };
    });
  }

  function barChart(rows, options = {}) {
    const clean = (rows || []).filter(row => row.value != null && Number.isFinite(Number(row.value)));
    if (!clean.length) return "<p class='muted'>Sin datos suficientes para dibujar el gráfico.</p>";
    const width = 920, rowHeight = 34, left = 170, right = 70, top = 24, bottom = 42;
    const height = top + bottom + clean.length * rowHeight;
    const values = clean.map(row => Number(row.value));
    let minimum = Math.min(0, ...values), maximum = Math.max(0, ...values);
    if (minimum === maximum) maximum = minimum + 0.01;
    const pad = (maximum - minimum) * 0.08;
    minimum -= pad; maximum += pad;
    const innerWidth = width - left - right;
    const x = value => left + (Number(value) - minimum) * innerWidth / (maximum - minimum);
    const zero = x(0);
    const ticks = Array.from({length: 5}, (_, index) => minimum + index * (maximum - minimum) / 4);
    return `<div class="chart-card"><svg class="chart analytic-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.yLabel || "Valores")}">
      <g class="chart-grid">${ticks.map(value => `<line x1="${x(value)}" y1="${top}" x2="${x(value)}" y2="${height - bottom}"></line>
      <text x="${x(value)}" y="${height - 14}" text-anchor="middle">${esc(options.percent ? fmt(value, "rate") : fmt(value))}</text>`).join("")}
      <line x1="${zero}" y1="${top}" x2="${zero}" y2="${height - bottom}" style="stroke:#aaa"></line></g>
      ${clean.map((row, index) => {
        const value = Number(row.value), start = Math.min(zero, x(value)), size = Math.abs(x(value) - zero);
        const color = value >= 0 ? chartColors[index % chartColors.length] : "#ef7373";
        const y = top + index * rowHeight + 7;
        return `<text x="${left - 10}" y="${y + 13}" text-anchor="end">${esc(row.label)}</text>
          <rect x="${start}" y="${y}" width="${Math.max(size, 1)}" height="18" rx="3" style="fill:${color}"></rect>
          <text x="${value >= 0 ? x(value) + 6 : x(value) - 6}" y="${y + 13}" text-anchor="${value >= 0 ? "start" : "end"}">${esc(options.percent ? fmt(value, "rate") : fmt(value))}</text>`;
      }).join("")}
      <text class="axis-title" x="${left + innerWidth / 2}" y="${height - 1}" text-anchor="middle">${esc(options.yLabel || "Valor")}</text>
      </svg></div>`;
  }

  function equity(rows) {
    if (!rows?.length) return "<p class='muted'>Sin curva de equity.</p>";
    const series = rows.flatMap(row => [
      {snapshot_date: row.snapshot_date, series: "Cartera", value: row.portfolio_value},
      {snapshot_date: row.snapshot_date, series: "SPY", value: row.benchmark_value},
    ]);
    return multiLineChart(series, "snapshot_date", "series", "value", {yLabel: "Valor acumulado", integerAxis: true});
  }

  async function init() {
    try {
      state.catalog = await api("/api/catalog");
      state.definition = structuredClone(state.catalog.recommended_definition);
      await preflight();
      renderHome();
    } catch (error) { app.innerHTML = `<p class="error-text">${esc(error.message)}</p>`; }
  }
  init();
})();
