(() => {
  "use strict";

  const state = {
    catalog: null,
    definition: {},
    budget: null,
    studies: [],
    hypotheses: [],
    models: [],
    selectedEntity: "",
    selectedProfile: "",
    analysisView: "performance",
  };
  const app = document.getElementById("app");
  const toast = document.getElementById("toast");

  const api = async (path, options = {}) => {
    const response = await fetch(path, {
      headers: {"Content-Type": "application/json"},
      ...options,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  };
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const fmt = value => typeof value === "number"
    ? new Intl.NumberFormat("es-ES", {maximumFractionDigits: 4}).format(value) : (value ?? "—");
  const notify = (message, error = false) => {
    toast.textContent = message;
    toast.className = error ? "show error" : "show";
    setTimeout(() => { toast.className = ""; }, 4500);
  };

  document.getElementById("nav").addEventListener("click", event => {
    const button = event.target.closest("[data-view]");
    if (!button) return;
    document.querySelectorAll("#nav button").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    render(button.dataset.view);
  });

  function defaultDefinition() {
    return structuredClone(state.catalog.recommended_exploratory_definition);
  }

  function allFixedDefinition() {
    return Object.fromEntries(state.catalog.variables.map(variable => [
      variable.id, {mode: "fixed", values: [variable.recommended]},
    ]));
  }

  function active(variable) {
    return (variable.depends_on || []).every(dependency => {
      const controller = state.definition[dependency.variable];
      const selected = controller?.values || [];
      return controller?.mode === "fixed" && selected.some(value => dependency.values.includes(value));
    });
  }

  function render(view = "new") {
    if (view === "new") renderNew();
    if (view === "studies") renderStudies();
    if (view === "hypotheses") renderHypotheses();
    if (view === "analysis") renderAnalysis();
  }

  function renderNew() {
    const stages = state.catalog.stage_order.map(stage => {
      const variables = state.catalog.variables.filter(item => item.stage === stage && active(item));
      const detail = state.catalog.stages.find(item => item.id === stage);
      const stagePlan = describeStage(stage, variables);
      return `<section class="stage-card">
        <header class="stage-heading">
          <div><p class="eyebrow">Etapa ${state.catalog.stage_order.indexOf(stage) + 1}</p>
          <h3>${esc(detail.label)}</h3><p>${esc(detail.description)}</p>
          <small><b>Pregunta:</b> ${esc(detail.question)}</small></div>
          <div class="stage-plan"><b>+${stagePlan.evaluations}</b><span>evaluaciones de etapa</span>
          <small>${esc(stagePlan.tests)}</small></div>
        </header>
        <div class="variable-rows">${variables.map(variableRow).join("")}</div>
      </section>`;
    }).join("");
    app.innerHTML = `
      <section class="hero">
        <div><p class="eyebrow">Exploratory Study</p><h2>Construye una hipótesis secuencial</h2>
        <p>Cada variable usa exclusivamente valores del catálogo v${state.catalog.version}.</p>
        <p class="recommendation"><b>Configuración inicial recomendada:</b>
        ${esc(state.catalog.recommended_exploratory_rationale)}</p>
        <div class="preset-actions">
          <button id="load-recommended">Restaurar recomendación</button>
          <button id="load-fixed">Dejar todo fijo</button>
        </div></div>
        <div id="budget" class="budget">${budgetMarkup()}</div>
      </section>
      <div class="study-meta">
        <label>Nombre<input id="study-name" value="Exploratory Study"></label>
        <label>Nota opcional<input id="study-note" placeholder="No afecta a la ejecución"></label>
      </div>
      ${stages}
      <div class="actions"><button id="launch-exploratory" class="primary" ${state.budget ? "" : "disabled"}>Lanzar Exploratory</button></div>
      <section class="confirm-card">
        <div><p class="eyebrow">Confirmatory Study</p><h2>Corrobora una hipótesis congelada</h2>
        <p>No admite ningún override científico. Ejecuta siempre 23 evaluaciones.</p></div>
        <select id="confirm-hypothesis"><option value="">Selecciona hipótesis…</option>
          ${state.hypotheses.map(item => `<option value="${esc(item.hypothesis_id)}">${esc(item.hypothesis_id)} · ${esc(item.statement)}</option>`).join("")}
        </select>
        <button id="launch-confirmatory" class="primary">Lanzar Confirmatory</button>
      </section>`;
    app.querySelectorAll("[data-mode]").forEach(control => control.addEventListener("change", onConfigChange));
    app.querySelectorAll("[data-values]").forEach(control => control.addEventListener("change", onConfigChange));
    document.getElementById("launch-exploratory").onclick = launchExploratory;
    document.getElementById("launch-confirmatory").onclick = launchConfirmatory;
    document.getElementById("load-recommended").onclick = () => {
      state.definition = defaultDefinition();
      state.budget = null;
      renderNew();
      preflight();
    };
    document.getElementById("load-fixed").onclick = () => {
      state.definition = allFixedDefinition();
      state.budget = null;
      renderNew();
      preflight();
    };
  }

  function describeStage(stage, variables) {
    const optimized = variables.filter(variable => state.definition[variable.id].mode === "optimize");
    const evaluations = optimized.reduce(
      (total, variable) => total + state.definition[variable.id].values.length, 0,
    );
    const tests = optimized.length
      ? optimized.map(variable => `${variable.label}: ${state.definition[variable.id].values.join(", ")}`).join(" · ")
      : "Sin comparación: todas las variables de esta etapa están fijas.";
    return {stage, evaluations, tests};
  }

  function variableRow(variable) {
    const selection = state.definition[variable.id];
    const isOptimize = selection.mode === "optimize";
    const choiceType = isOptimize ? "checkbox" : "radio";
    return `<article class="variable-row">
      <div class="variable-copy"><strong>${esc(variable.label)}</strong>
        <p>${esc(variable.description)}</p>
        <small>${esc(variable.cost)} · invalida ${esc(variable.invalidates)}</small>
      </div>
      <fieldset class="mode-switch" aria-label="Modo de ${esc(variable.label)}">
        ${variable.modes.map(mode => `<label class="mode-option ${selection.mode === mode ? "selected" : ""}">
          <input type="radio" name="mode-${esc(variable.id)}" value="${mode}" data-mode="${esc(variable.id)}" ${selection.mode === mode ? "checked" : ""}>
          ${mode === "fixed" ? "Fijo" : "Optimizar"}
        </label>`).join("")}
      </fieldset>
      <div class="value-choices ${isOptimize ? "optimizing" : "fixed"}" aria-label="Valores de ${esc(variable.label)}">
        ${variable.value_options.map(option => {
          const value = option.value;
          const serialized = JSON.stringify(value);
          const checked = selection.values.some(item => JSON.stringify(item) === serialized);
          const recommended = serialized === JSON.stringify(variable.recommended);
          return `<label class="value-choice ${checked ? "selected" : ""}">
            <input type="${choiceType}" name="value-${esc(variable.id)}" value="${esc(serialized)}" data-values="${esc(variable.id)}" ${checked ? "checked" : ""}>
            <span class="value-choice-copy"><b>${esc(option.label)}${recommended ? " · recomendado" : ""}</b><small>${esc(option.description)}</small></span>
          </label>`;
        }).join("")}
      </div>
    </article>`;
  }

  function onConfigChange(event) {
    const variableId = event.target.dataset.mode || event.target.dataset.values;
    const selection = state.definition[variableId];
    if (event.target.dataset.mode) {
      selection.mode = event.target.value;
      const variable = state.catalog.variables.find(item => item.id === variableId);
      selection.values = selection.mode === "fixed"
        ? [variable.recommended]
        : [...variable.values];
    } else {
      const value = JSON.parse(event.target.value);
      if (selection.mode === "fixed") {
        selection.values = [value];
      } else {
        const selected = [...document.querySelectorAll(`[data-values="${variableId}"]:checked`)]
          .map(control => JSON.parse(control.value));
        const variable = state.catalog.variables.find(item => item.id === variableId);
        if (selected.length < 2) {
          event.target.checked = true;
          notify("Optimizar requiere al menos dos valores.", true);
          return;
        }
        if (selected.length > variable.max_values) {
          event.target.checked = false;
          notify(`Optimizar admite como máximo ${variable.max_values} valores.`, true);
          return;
        }
        selection.values = selected;
      }
    }
    state.budget = null;
    renderNew();
    preflight();
  }

  let preflightTimer = null;
  function preflight() {
    clearTimeout(preflightTimer);
    preflightTimer = setTimeout(async () => {
      try {
        const result = await api("/api/exploratory/preflight", {
          method: "POST", body: JSON.stringify({definition: state.definition}),
        });
        state.definition = result.definition;
        state.budget = result.budget;
        const budget = document.getElementById("budget");
        if (budget) budget.innerHTML = budgetMarkup();
        const launch = document.getElementById("launch-exploratory");
        if (launch) launch.disabled = false;
      } catch (error) {
        state.budget = null;
        const budget = document.getElementById("budget");
        if (budget) budget.innerHTML = `<strong>Configuración inválida</strong><span>${esc(error.message)}</span>`;
        const launch = document.getElementById("launch-exploratory");
        if (launch) launch.disabled = true;
      }
    }, 180);
  }

  function budgetMarkup() {
    if (!state.budget) return "<strong>Calculando…</strong>";
    const b = state.budget;
    return `
      <div><b>${b.exploratory_evaluations}</b><span>Exploratory</span></div>
      <div><b>${b.expensive_fits}</b><span>Fits caros</span></div>
      <div><b>23</b><span>Confirmatory</span></div>
      <div><b>${b.total_cycle_evaluations}</b><span>Total ciclo</span></div>
      <div><b>${b.meta_recombinations}</b><span>Meta</span></div>
      <div><b>${b.backtests}</b><span>Backtests</span></div>
      <div><b>${(b.estimated_incremental_bytes / 1073741824).toFixed(2)} GiB</b><span>Disco</span></div>
      <div><b>${Math.floor(b.estimated_minutes / 60)}h ${b.estimated_minutes % 60}m</b><span>Tiempo</span></div>`;
  }

  async function launchExploratory() {
    try {
      const payload = {
        name: document.getElementById("study-name").value,
        note: document.getElementById("study-note").value,
        definition: state.definition,
      };
      const job = await api("/api/exploratory", {method: "POST", body: JSON.stringify(payload)});
      notify(`Exploratory lanzado: ${job.job_id}`);
      pollJob(job.job_id, "studies");
    } catch (error) { notify(error.message, true); }
  }

  async function launchConfirmatory() {
    const hypothesisId = document.getElementById("confirm-hypothesis").value;
    if (!hypothesisId) return notify("Selecciona una hipótesis congelada.", true);
    try {
      await api("/api/confirmatory/preflight", {
        method: "POST", body: JSON.stringify({hypothesis_id: hypothesisId}),
      });
      const job = await api("/api/confirmatory", {
        method: "POST", body: JSON.stringify({hypothesis_id: hypothesisId, name: "Confirmatory Study"}),
      });
      notify(`Confirmatory lanzado: ${job.job_id}`);
      pollJob(job.job_id, "studies");
    } catch (error) { notify(error.message, true); }
  }

  async function pollJob(jobId, destination) {
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      const job = await api(`/api/jobs/${jobId}`);
      if (job.status === "succeeded") {
        notify("Trabajo completado.");
        await refresh();
        render(destination);
        return;
      }
      if (job.status === "failed") {
        notify(job.error, true);
        await refresh();
        render(destination);
        return;
      }
    }
  }

  async function renderStudies() {
    await refresh();
    app.innerHTML = `<section class="page-head"><p class="eyebrow">Trazabilidad</p><h2>Estudios</h2></section>
      <div class="list">${state.studies.map(study => `
        <button class="list-row" data-study="${esc(study.study_id)}">
          <span><b>${esc(study.name)}</b><small>${esc(study.study_type)} · ${esc(study.study_id)}</small></span>
          <span class="status ${esc(study.status)}">${esc(study.status)}</span>
        </button>`).join("") || "<p class='muted'>No hay estudios.</p>"}</div>
      <section id="study-detail"></section>`;
    app.querySelectorAll("[data-study]").forEach(button => {
      button.onclick = () => loadStudy(button.dataset.study);
    });
  }

  async function loadStudy(studyId) {
    try {
      const study = await api(`/api/studies/${studyId}`);
      const pending = study.pending_decision;
      const candidates = pending?.candidates || [];
      document.getElementById("study-detail").innerHTML = `
        <article class="detail">
          <h2>${esc(study.name)}</h2>
          <div class="metrics">
            <div><b>${esc(study.status)}</b><span>Estado</span></div>
            <div><b>${study.ledger.length}</b><span>Evaluaciones</span></div>
            <div><b>${study.completed_evaluations ?? "—"}</b><span>Confirmadas</span></div>
            <div><b>${esc(study.verdict || "—")}</b><span>Veredicto</span></div>
          </div>
          ${pending ? `<h3>Decisión: ${esc(pending.variable_id)}</h3>
            <div class="candidates">${candidates.map(candidate => `
              <label class="candidate">
                <input type="radio" name="candidate" value="${esc(candidate.candidate_id)}" ${candidate.candidate_id === pending.automatic_candidate_id ? "checked" : ""}>
                <b>${esc(JSON.stringify(candidate.value))}</b>
                <span>Rank-IC ${fmt(candidate.result.summary.mean_rank_ic)} · IR ${fmt(candidate.result.summary.information_ratio)}</span>
                <small>${esc(candidate.eligibility_reason)}</small>
              </label>`).join("")}</div>
            <select id="decision-reason">${state.catalog.decision_reasons.map(reason => `<option value="${reason}">${reason}</option>`).join("")}</select>
            <button id="advance-study" class="primary">Aceptar y continuar</button>` : ""}
          ${study.status === "awaiting_freeze" ? `<button id="freeze-study" class="primary">Congelar hipótesis</button>` : ""}
          <h3>Ledger</h3>${table(study.ledger, ["evaluation_number", "stage", "variable_id", "candidate_value", "selected", "elapsed_seconds"])}
        </article>`;
      const advance = document.getElementById("advance-study");
      if (advance) advance.onclick = async () => {
        const candidateId = document.querySelector('input[name="candidate"]:checked').value;
        const reason = document.getElementById("decision-reason").value;
        const job = await api(`/api/exploratory/${studyId}/advance`, {
          method: "POST", body: JSON.stringify({candidate_id: candidateId, reason}),
        });
        pollJob(job.job_id, "studies");
      };
      const freeze = document.getElementById("freeze-study");
      if (freeze) freeze.onclick = async () => {
        const job = await api(`/api/exploratory/${studyId}/freeze`, {method: "POST", body: "{}"});
        pollJob(job.job_id, "hypotheses");
      };
    } catch (error) { notify(error.message, true); }
  }

  async function renderHypotheses() {
    await refresh();
    app.innerHTML = `<section class="page-head"><p class="eyebrow">Puntos de partida inmutables</p><h2>Hipótesis</h2></section>
      <div class="cards">${state.hypotheses.map(item => `
        <article class="card"><span class="status succeeded">frozen</span><h3>${esc(item.hypothesis_id)}</h3>
        <p>${esc(item.statement)}</p><small>Dataset ${esc(String(item.dataset_hash).slice(0, 12))}</small>
        <button data-analyse="${esc(item.hypothesis_id)}">Abrir análisis</button></article>`).join("") || "<p class='muted'>No hay hipótesis congeladas.</p>"}</div>`;
    app.querySelectorAll("[data-analyse]").forEach(button => button.onclick = () => {
      state.selectedEntity = button.dataset.analyse;
      document.querySelector('[data-view="analysis"]').click();
    });
  }

  async function renderAnalysis() {
    await refresh();
    const entities = [...state.models, ...state.hypotheses];
    if (!state.selectedEntity && entities.length) {
      state.selectedEntity = entities[0].model_id || entities[0].hypothesis_id;
    }
    app.innerHTML = `<section class="analysis-head">
      <select id="entity-select">${entities.map(item => {
        const id = item.model_id || item.hypothesis_id;
        return `<option value="${esc(id)}" ${id === state.selectedEntity ? "selected" : ""}>${esc(id)}</option>`;
      }).join("")}</select>
      <select id="profile-select"><option value="">Ganador balanced</option>${["growth", "value", "quality", "momentum", "contrarian", "defensive", "garp"].map(profile => `<option value="${profile}" ${profile === state.selectedProfile ? "selected" : ""}>Perfil ${profile}</option>`).join("")}</select>
      <div class="subnav">${["performance", "learning", "rankings", "portfolio", "trades", "stocks"].map(view =>
        `<button data-analysis="${view}" class="${state.analysisView === view ? "active" : ""}">${view}</button>`).join("")}</div>
    </section><section id="analysis-body"><p class="muted">Cargando…</p></section>`;
    document.getElementById("entity-select").onchange = event => {
      state.selectedEntity = event.target.value; loadAnalysis();
    };
    document.getElementById("profile-select").onchange = event => {
      state.selectedProfile = event.target.value; loadAnalysis();
    };
    app.querySelectorAll("[data-analysis]").forEach(button => button.onclick = () => {
      state.analysisView = button.dataset.analysis; renderAnalysis();
    });
    loadAnalysis();
  }

  async function loadAnalysis() {
    const body = document.getElementById("analysis-body");
    if (!state.selectedEntity) {
      body.innerHTML = "<p class='muted'>Congela una hipótesis para habilitar el análisis.</p>";
      return;
    }
    try {
      if (state.analysisView === "stocks") {
        body.innerHTML = `<div class="stock-search"><input id="ticker" value="AAPL"><button id="load-stock">Consultar</button></div><div id="stock-body"></div>`;
        document.getElementById("load-stock").onclick = loadStock;
        return loadStock();
      }
      const profile = state.selectedProfile && ["performance", "portfolio", "trades"].includes(state.analysisView)
        ? `?profile=${encodeURIComponent(state.selectedProfile)}` : "";
      const data = await api(`/api/entities/${state.selectedEntity}/${state.analysisView}${profile}`);
      if (state.analysisView === "performance") body.innerHTML = performanceMarkup(data);
      if (state.analysisView === "learning") body.innerHTML = learningMarkup(data);
      if (state.analysisView === "rankings") body.innerHTML = table(data.rows);
      if (state.analysisView === "portfolio") body.innerHTML = `<h3>Posiciones</h3>${table(data.positions)}`;
      if (state.analysisView === "trades") body.innerHTML = table(data.orders);
    } catch (error) { body.innerHTML = `<p class="error-text">${esc(error.message)}</p>`; }
  }

  async function loadStock() {
    const ticker = document.getElementById("ticker").value;
    const target = document.getElementById("stock-body");
    try {
      const data = await api(`/api/entities/${state.selectedEntity}/stocks/${encodeURIComponent(ticker)}`);
      target.innerHTML = `<h2>${esc(data.ticker)}</h2><h3>Scores</h3>${table(data.scores)}<h3>Fundamentales PIT</h3>${table(data.panel)}<h3>Órdenes</h3>${table(data.orders)}`;
    } catch (error) { target.innerHTML = `<p class="error-text">${esc(error.message)}</p>`; }
  }

  function performanceMarkup(data) {
    const summary = data.summary.summary || data.summary;
    return `<div class="metrics">${["cagr_portfolio", "cagr_benchmark", "mean_annual_alpha", "information_ratio", "max_drawdown", "annualized_turnover"].map(key =>
      `<div><b>${fmt(summary[key])}</b><span>${key}</span></div>`).join("")}</div>
      ${equitySvg(data.equity)}<h3>Por año</h3>${table(data.annual)}`;
  }

  function learningMarkup(data) {
    return `<h3>Rank-IC</h3>${table(data.rank_ic)}<h3>Cola operada</h3>${table(data.tail)}
      <h3>Pesos del meta</h3>${table(data.weights)}<h3>Salud</h3>${table(data.health)}
      <h3>Atribución</h3>${table(data.attribution)}`;
  }

  function equitySvg(rows) {
    if (!rows.length) return "<p class='muted'>Sin curva de equity.</p>";
    const values = rows.flatMap(row => [Number(row.portfolio_value), Number(row.benchmark_value)]).filter(Number.isFinite);
    const min = Math.min(...values), max = Math.max(...values), range = max - min || 1;
    const points = key => rows.map((row, index) => {
      const x = 20 + index * 760 / Math.max(rows.length - 1, 1);
      const y = 230 - (Number(row[key]) - min) * 200 / range;
      return `${x},${y}`;
    }).join(" ");
    return `<svg class="chart" viewBox="0 0 800 250" role="img" aria-label="Curva de rentabilidad">
      <polyline class="portfolio-line" points="${points("portfolio_value")}"></polyline>
      <polyline class="benchmark-line" points="${points("benchmark_value")}"></polyline>
    </svg>`;
  }

  function table(rows, preferred = null) {
    if (!rows?.length) return "<p class='muted'>Sin datos.</p>";
    const columns = preferred || Object.keys(rows[0]).slice(0, 12);
    return `<div class="table-wrap"><table><thead><tr>${columns.map(column => `<th>${esc(column)}</th>`).join("")}</tr></thead>
      <tbody>${rows.slice(0, 500).map(row => `<tr>${columns.map(column => `<td>${esc(fmt(row[column]))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  async function refresh() {
    [state.studies, state.hypotheses, state.models] = await Promise.all([
      api("/api/studies"), api("/api/hypotheses"), api("/api/models"),
    ]);
  }

  async function init() {
    try {
      state.catalog = await api("/api/catalog");
      state.definition = defaultDefinition();
      await refresh();
      await preflight();
      renderNew();
      preflight();
    } catch (error) {
      app.innerHTML = `<p class="error-text">${esc(error.message)}</p>`;
    }
  }

  init();
})();
