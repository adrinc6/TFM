(() => {
  const app = document.getElementById("app");
  const navBack = document.getElementById("nav-back");
  const dev = location.pathname.startsWith("/dev");
  const base = dev ? "/dev" : "";
  const state = {
    catalog: null, definition: null, budget: null, studies: [],
    selectedStudy: null, selectedRun: null, section: "summary", eventSequence: 0, timer: null,
    snapshot: null, stockTicker: null, stockView: "portfolio", stockParameter: null,
  };

  const api = async (path, options = {}) => {
    const response = await fetch(base + path, {...options, headers: {"Content-Type": "application/json"}});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  };
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
  const percentageKey = key => /(^|_)(alpha|return|cagr|drawdown|rate|fraction|weight|turnover|p_value)(_|$)|rank_ic/.test(key);
  const integerKey = key => /^(year|runs_completados|runs_restantes|runs_total|observations|training_rows|realized_cohorts|closed_cohorts|attempt|target_size|elapsed_seconds)$/.test(key);
  const timestampKey = key => /^(actualizado|updated_at|created_at|heartbeat)$/.test(key);
  const fmt = (value, key = "") => {
    if (typeof value !== "number" || !Number.isFinite(value)) return value ?? "—";
    if (percentageKey(key)) return `${(value * 100).toFixed(2)} %`;
    if (integerKey(key)) return String(Math.round(value));
    return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
  };
  const display = (value, key = "") => timestampKey(key) && typeof value === "string"
    ? value.replace(/\.\d+(?=\+|Z|$)/, "").replace(/\+00:00$/, " UTC")
    : fmt(value, key);
  const tone = (value, key = "") => typeof value === "number" && /(alpha|return|rank_ic|information|p_value)/.test(key) ? (value > 0 ? "positive" : value < 0 ? "negative" : "") : "";
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
        <small>${selection.values.length === 1 ? "Fixed" : variable.predictive ? "Optimize" : "Diagnóstico"} · Coste: ${esc(variable.cost)} · Recomendado: ${esc(variable.recommended)}</small></div>
      <div class="value-choices columns-${columnCount}">${variable.value_options.map(option => {
        const checked = selection.values.some(value => JSON.stringify(value) === JSON.stringify(option.value));
        return `<label class="value-choice"><input type="checkbox"
          name="value-${esc(variable.id)}" data-value='${esc(JSON.stringify(option.value))}' ${checked ? "checked" : ""}>
          <span><b>${esc(option.label)}</b><small>${esc(option.description)}</small></span></label>`;
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
        <div class="stage-plan"><b>${stageRuns(stage)}</b><span>evaluaciones predictivas</span>
        <small>${stage === "portfolio" ? "Comparaciones informativas; nunca eligen modelo." : "Comparación secuencial contra el incumbent acumulado."}</small></div></div>
        <div>${variables.map(variableRow).join("")}</div></section>`;
    }).join("");
    app.innerHTML = `<section class="hero"><div><p class="eyebrow">Configuración cerrada</p><h2>Un Study, una ruta científica</h2>
      <p>Solo Rank-IC robusto decide. Carteras, perfiles y robustez explican el ganador después.</p></div>${budgetMarkup()}</section>
      <div class="study-meta"><label>Nombre<input id="study-name" value="Model Study"></label>
      <label>Nota<input id="study-note" placeholder="Opcional; no afecta a la ciencia"></label></div>
      ${stages}<div class="actions"><button id="launch" class="primary">Lanzar Study</button></div>`;
    bindConfiguration();
    document.getElementById("launch").onclick = launch;
  }

  function budgetMarkup() {
    const budget = state.budget || {};
    const items = [
      ["predictive_evaluations", "Predictivas"], ["expensive_fits", "Fits caros"],
      ["meta_recombinations", "Meta"], ["portfolio_diagnostics", "Carteras"],
      ["profiles", "Perfiles"], ["robustness_groups", "Robustez"],
      ["total_runs", "Runs previstos"], ["estimated_minutes", "Minutos estimados"],
    ];
    return `<div class="budget">${items.map(([key, label]) => `<div><b>${Number.isFinite(Number(budget[key])) ? Math.round(Number(budget[key])) : "—"}</b><span>${label}</span></div>`).join("")}</div>`;
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
        } else selection.values = selection.values.filter(item => JSON.stringify(item) !== JSON.stringify(value));
        const spec = state.catalog.variables.find(item => item.id === id);
        selection.mode = selection.values.length === 1 ? "fixed" : spec.predictive ? "optimize" : "diagnostic";
        preflight().then(renderHome).catch(error => notify(error.message, true));
      });
    });
  }

  async function preflight() {
    const response = await api("/api/studies/preflight", {
      method: "POST", body: JSON.stringify({definition: state.definition, run_scope: dev ? "dev" : "full"}),
    });
    state.definition = response.definition; state.budget = response.budget;
  }

  async function launch() {
    try {
      const payload = {
        name: document.getElementById("study-name").value,
        note: document.getElementById("study-note").value,
        definition: state.definition, run_scope: dev ? "dev" : "full",
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
    app.innerHTML = `<section class="entity-header"><div class="entity-main">
      <div><p class="eyebrow">${esc(study.study_id)}</p><h2>${esc(study.name)}</h2><p>${esc(study.note || "Sin nota.")}</p></div></div>
      <div class="entity-cards">${[
        ["Estado", study.status], ["Etapa", study.phase], ["Progreso", `${Math.round((study.progress || 0) * 100)} %`],
        ["Runs", `${study.completed_runs || 0}/${total}`], ["Tiempo", duration(Date.now() - Date.parse(study.created_at))],
        ["Rank-IC máx.", best > -Infinity ? fmt(best, "rank_ic") : "—"],
      ].map(([label, value]) => `<span><b>${esc(value)}</b>${label}</span>`).join("")}</div>
      <div class="entity-actions">
        <button data-study-view="runs" class="${state.section === "runs" ? "active" : ""}">Runs</button>
        <button data-study-view="console" class="${state.section === "console" ? "active" : ""}">Consola</button>
        <button data-study-view="robustness" class="${state.section === "robustness" ? "active" : ""}">Robustez</button>
        <button data-study-view="profiles" class="${state.section === "profiles" ? "active" : ""}">Perfiles</button>
        ${active ? '<button id="pause-study">Pausar</button>' : ""}
        ${resumable ? '<button id="resume-study">Reanudar</button>' : ""}
        <button id="refresh-study">Actualizar</button>
      </div></section><section id="study-content"></section>`;
    document.getElementById("refresh-study").onclick = renderStudyPage;
    app.querySelectorAll("[data-study-view]").forEach(button => button.onclick = () => {
      state.section = button.dataset.studyView;
      renderStudyPage();
    });
    if (document.getElementById("pause-study")) document.getElementById("pause-study").onclick = () => studyAction("pause");
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
        alpha_anual: run.result?.summary?.mean_annual_alpha,
        alpha_estres: run.result?.summary?.stress_mean_alpha,
        elapsed_seconds: run.elapsed_seconds,
        source: run.result?.source, error: run.error,
      }));
      body.innerHTML = `<div class="content-heading"><div><h2>Runs</h2><p>Cada fila es una evaluación persistente.</p></div></div>${runSelectionTable(rows)}`;
      body.querySelectorAll("[data-run]").forEach(button => button.onclick = () => {
        state.selectedRun = button.dataset.run;
        state.runView = "summary";
        renderRunPage();
      });
      return;
    }
    if (state.section === "console") {
      const events = await api(`/api/studies/${state.selectedStudy}/events?after=0`);
      body.innerHTML = `<div class="content-heading"><h2>Consola</h2><button id="refresh-console">Actualizar eventos</button></div>
        <pre class="console">${events.map(event => `${event.sequence} ${event.timestamp} [${event.level}] ${event.message}`).join("\n")}</pre>`;
      const consoleElement = body.querySelector(".console");
      consoleElement.scrollTop = consoleElement.scrollHeight;
      document.getElementById("refresh-console").onclick = async () => {
        const latest = await api(`/api/studies/${state.selectedStudy}/events?after=0`);
        consoleElement.textContent = latest.map(event => `${event.sequence} ${event.timestamp} [${event.level}] ${event.message}`).join("\n");
        consoleElement.scrollTop = consoleElement.scrollHeight;
      };
      return;
    }
    const data = await api(`/api/studies/${state.selectedStudy}/analysis/${state.section}`);
    body.innerHTML = state.section === "profiles" ? profiles(data) : robustnessView(data);
    bindInteractiveCharts(body);
  }

  async function renderRunPage() {
    clearInterval(state.timer);
    setBackNavigation("← Study", renderStudyPage);
    const run = await api(`/api/studies/${state.selectedStudy}/runs/${state.selectedRun}`);
    const summary = run.result?.summary || {};
    const views = ["summary", "performance", "learning", "portfolio", "stocks"];
    app.innerHTML = `<section class="entity-header run-header"><div class="entity-main">
      <div><p class="eyebrow">${esc(run.phase)} · ${esc(run.variable_id)}</p><h2>${esc(run.run_id)}</h2>
      <p>${esc(run.logical_key)}. Evalúa ${esc(run.value)} dentro de la fase ${esc(run.phase)}.</p></div></div>
      <div class="entity-cards">${[
        ["Estado", run.status], ["Intento", run.attempt], ["Progreso", `${Math.round((run.progress || 0) * 100)} %`],
        ["Duración", `${fmt(run.elapsed_seconds)} s`], ["Rank-IC", fmt(summary.mean_rank_ic, "rank_ic")], ["Fuente", run.result?.source || "—"],
      ].map(([label, value]) => `<span><b>${esc(fmt(value))}</b>${label}</span>`).join("")}</div>
      <div class="entity-actions">${views.map(view => `<button data-run-view="${view}" class="${state.runView === view ? "active" : ""}">${view}</button>`).join("")}</div>
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
      body.innerHTML = `<h2>Resumen del run</h2>${metrics(summary)}<h3>Resultados por era</h3>${table(run.result?.eras || [])}
        <h3>Configuración y evidencia</h3>${configurationCards(run)}
        ${run.error ? `<h3>Error</h3><pre class="error-text report">${esc(run.error)}\n${esc(run.traceback || "")}</pre>` : ""}`;
      return;
    }
    const isWinner = run.logical_key === "winner:evidence";
    if (!isWinner) {
      body.innerHTML = `<article class="detail"><h2>Vista no materializada para este candidato</h2>
        <p>Los candidatos normales conservan configuración, Rank-IC, eras y decisión. Las vistas pesadas se guardan únicamente en el run de evidencia del ganador.</p></article>`;
      return;
    }
    const map = {performance: "portfolio", learning: "learning", portfolio: "portfolio", stocks: "stocks"};
    if (state.runView === "stocks") {
      return renderStockBrowser();
    }
    const snapshotQuery = state.runView === "portfolio" && state.snapshot ? `?snapshot=${encodeURIComponent(state.snapshot)}` : "";
    const data = await api(`/api/studies/${state.selectedStudy}/analysis/${map[state.runView]}${snapshotQuery}`);
    if (state.runView === "performance") {
      body.innerHTML = `${metrics(data.summary?.summary || data.summary || {})}${equity(data.equity)}${table(data.annual)}`;
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
      <h3>Movimientos ejecutados ese día</h3>${table(data.orders)}`;
    document.getElementById("portfolio-snapshot").onchange = event => {
      state.snapshot = event.target.value;
      renderRunPage();
    };
  }

  async function renderStockBrowser() {
    const initial = await api(`/api/studies/${state.selectedStudy}/analysis/stocks${state.snapshot ? `?snapshot=${encodeURIComponent(state.snapshot)}` : ""}`);
    state.snapshot = state.snapshot || initial.selected_snapshot;
    state.stockTicker = state.stockTicker || initial.available_tickers?.[0] || null;
    if (!state.stockTicker) {
      document.getElementById("run-content").innerHTML = "<p class='muted'>No hay acciones disponibles.</p>";
      return;
    }
    await loadStockBrowser();
  }

  async function loadStockBrowser() {
    const params = new URLSearchParams({ticker: state.stockTicker});
    if (state.snapshot) params.set("snapshot", state.snapshot);
    if (state.stockParameter) params.set("parameter", state.stockParameter);
    const data = await api(`/api/studies/${state.selectedStudy}/analysis/stocks?${params}`);
    state.snapshot = data.selected_snapshot;
    state.stockParameter = data.selected_parameter || state.stockParameter;
    const body = document.getElementById("run-content");
    const views = [
      ["portfolio", "Situación en cartera"], ["agents", "Agentes y meta"],
      ["parameter-scores", "Puntuaciones de parámetros"], ["parameter-values", "Valores PIT"],
      ["evolution", "Evolución de parámetro"],
    ];
    const content = stockContent(data);
    body.innerHTML = `<section class="snapshot-controls stock-controls">
      <label>Acción<select id="stock-ticker">${data.available_tickers.map(value => `<option value="${esc(value)}" ${value === data.ticker ? "selected" : ""}>${esc(value)}</option>`).join("")}</select></label>
      <label>Snapshot<select id="stock-snapshot">${data.available_snapshots.map(value => `<option value="${esc(value)}" ${value === data.selected_snapshot ? "selected" : ""}>${esc(value)}</option>`).join("")}</select></label>
      <label>Vista<select id="stock-view">${views.map(([id, label]) => `<option value="${id}" ${id === state.stockView ? "selected" : ""}>${label}</option>`).join("")}</select></label>
      ${state.stockView === "evolution" ? `<label>Parámetro<select id="stock-parameter">${data.parameter_options.map(option => `<option value="${esc(option.id)}" ${option.id === data.selected_parameter ? "selected" : ""}>${esc(option.kind)} · ${esc(option.label)}</option>`).join("")}</select></label>` : ""}
    </section><section id="stock-data">${content}</section>`;
    document.getElementById("stock-ticker").onchange = event => { state.stockTicker = event.target.value; loadStockBrowser(); };
    document.getElementById("stock-snapshot").onchange = event => { state.snapshot = event.target.value; loadStockBrowser(); };
    document.getElementById("stock-view").onchange = event => { state.stockView = event.target.value; loadStockBrowser(); };
    if (document.getElementById("stock-parameter")) document.getElementById("stock-parameter").onchange = event => { state.stockParameter = event.target.value; loadStockBrowser(); };
    bindInteractiveCharts(body);
  }

  function stockContent(data) {
    if (state.stockView === "portfolio") return `<h3>Situación en cartera · ${esc(data.selected_snapshot)}</h3>${table(data.positions)}<h3>Órdenes del snapshot</h3>${table(data.orders)}`;
    if (state.stockView === "agents") return `<h3>Scores de los cinco agentes y del meta-agente</h3>${objectTable(data.scores[0] || {})}`;
    if (state.stockView === "parameter-scores") return `<h3>Puntuaciones normalizadas de parámetros</h3>${objectTable(data.parameter_scores[0] || {})}`;
    if (state.stockView === "parameter-values") return `<h3>Valores point-in-time de parámetros</h3>${objectTable(data.parameter_values[0] || {})}`;
    return `<h3>Evolución · ${esc(data.selected_parameter || "")}</h3>${singleLineChart(data.parameter_history, "snapshot_date", data.selected_parameter, {yLabel: data.selected_parameter || "Valor"})}`;
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
    return `<h2>Robustez informativa</h2>
      ${metrics({
        observed_mean_rank_ic: observed,
        permutation_p_value: data.permutation?.p_value,
        rank_ic_bootstrap_90_low: data.bootstrap_and_era_exclusion?.interval_90?.ci_low,
        rank_ic_bootstrap_90_high: data.bootstrap_and_era_exclusion?.interval_90?.ci_high,
        meta_weight_concentration: data.meta_weight_stability?.mean_concentration,
        meta_weight_turnover: data.meta_weight_stability?.mean_turnover,
      })}
      <h3>Modelo, semillas y placebos</h3>${barChart(comparisons, {percent: true, yLabel: "Rank-IC"})}
      <h3>Rank-IC medio por agente</h3>${barChart(agents, {percent: true, yLabel: "Rank-IC"})}
      <details><summary>Resultado completo de robustez</summary>${objectTree(data)}</details>`;
  }
  function metrics(data) {
    const priority = ["mean_rank_ic", "rank_ic_positive_fraction", "cagr_portfolio", "cagr_benchmark", "cagr_difference", "information_ratio", "max_drawdown", "annualized_turnover"];
    const entries = [
      ...priority.filter(key => data[key] !== undefined).map(key => [key, data[key]]),
      ...Object.entries(data).filter(([key]) => !priority.includes(key)),
    ].slice(0, 8);
    return `<div class="metrics">${entries.map(([key, value]) => `<div><b class="${tone(value, key)}">${esc(fmt(value, key))}</b><span>${esc(metricLabel(key))}</span></div>`).join("")}</div>`;
  }
  function metricLabel(key) {
    return {cagr_difference: "Alpha anualizado vs SPY", cagr_portfolio: "CAGR cartera", cagr_benchmark: "CAGR SPY", mean_rank_ic: "Rank-IC medio", rank_ic_positive_fraction: "Cohortes Rank-IC positivas", annualized_turnover: "Turnover anualizado", max_drawdown: "Drawdown máximo", information_ratio: "Information Ratio"}[key] || key;
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
      ["Tiempo de cálculo", result.elapsed_seconds == null ? null : `${fmt(result.elapsed_seconds, "elapsed_seconds")} s`],
    ].filter(([, value]) => value != null);
    if (artifacts.length) cards.push(`<article class="config-card artifact-card"><h4>Artefactos</h4><ul>${artifacts.map(([label, value]) => `<li><span>${esc(label)}</span><b>${esc(String(value))}</b></li>`).join("")}</ul></article>`);
    return `<div class="config-card-grid">${cards.join("") || "<p class='muted'>Sin configuración persistida.</p>"}</div>`;
  }
  function table(rows) {
    if (!rows?.length) return "<p class='muted'>Sin datos.</p>";
    const columns = Object.keys(rows[0]).slice(0, 14);
    return `<div class="table-wrap"><table><thead><tr>${columns.map(key => `<th>${esc(key)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 500).map(row => `<tr>${columns.map(key => `<td class="${tone(row[key], key)}">${esc(fmt(typeof row[key] === "object" ? JSON.stringify(row[key]) : row[key], key))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function studySelectionTable(rows) {
    if (!rows.length) return "<p class='muted'>No hay Studies.</p>";
    const columns = ["nombre", "estado", "etapa", "progreso", "runs_completados", "runs_restantes", "tiempo", "rank_ic_max", "actualizado"];
    return `<table><thead><tr>${columns.map(key => `<th>${esc(key)}</th>`).join("")}</tr></thead><tbody>${rows.map(row =>
      `<tr class="${row.study_id === state.selectedStudy ? "selected-row" : ""}">${columns.map(key => `<td class="${tone(row[key], key)}">${esc(display(row[key], key))}</td>`).join("")}
      <td><button data-study="${esc(row.study_id)}">Abrir</button></td></tr>`).join("")}</tbody></table>`;
  }
  function runSelectionTable(rows) {
    if (!rows.length) return "<p class='muted'>Todavía no hay runs.</p>";
    const columns = ["phase", "variable", "value", "status", "progress", "rank_ic", "alpha_anual", "alpha_estres", "elapsed_seconds", "source", "error"];
    return `<table class="runs-table"><thead><tr>${columns.map(key => `<th>${esc(key)}</th>`).join("")}<th></th></tr></thead><tbody>${rows.map(row =>
      `<tr class="${row.run_id === state.selectedRun ? "selected-row" : ""}">${columns.map(key => `<td class="${tone(row[key], key)}">${esc(fmt(row[key], key))}</td>`).join("")}
      <td><button data-run="${esc(row.run_id)}">Ver run</button></td></tr>`).join("")}</tbody></table>`;
  }
  function duration(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours} h ${minutes} min`;
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
    const paths = series.map(item => {
      const points = item.values.map((value, index) => value == null ? null : `${x(index)},${y(value)}`);
      const segments = []; let segment = [];
      points.forEach(point => { if (point) segment.push(point); else if (segment.length) { segments.push(segment); segment = []; } });
      if (segment.length) segments.push(segment);
      return segments.map(values => `<polyline points="${values.join(" ")}" style="stroke:${item.color}"></polyline>`).join("");
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
        const bounds = svg.getBoundingClientRect();
        const point = (event.clientX - bounds.left) / bounds.width * data.width;
        const index = Math.max(0, Math.min(data.xValues.length - 1, Math.round((point - data.left) / ((data.width - data.left - data.right) / Math.max(data.xValues.length - 1, 1)))));
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
