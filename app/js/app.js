/* Arranque, estado global compartido y router de vistas. Las vistas se registran en
   TFM.views (cada una expone render(container)) y se activan desde la navegación. */
(function (global) {
  "use strict";
  const { api, el } = global.TFM;

  // Estado compartido entre vistas (defaults del backend, registro de runs, jobs)
  const state = {
    defaults: {},
    groups: {},
    studyOptions: {},
    settingsOptions: {},
    studyModelOptions: {},
    studyPortfolioOptions: {},
    studyPhase3Options: {},
    fullStudyModelOptions: {},
    fullStudyPortfolioOptions: {},
    fullStudyPhase3Options: {},
    fullStudyProfiles: [],
    studyOptionGroups: {},
    fullStudyFixedSettings: {},
    fullStudyStressSettings: {},
    presets: {},
    profileLabels: {},
    runs: [],
    studies: [],
    jobs: [],
  };
  global.TFM.state = state;
  global.TFM.views = global.TFM.views || {};

  // --- Router ---
  function showView(name) {
    document.querySelectorAll("nav.app-nav button").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === name)
    );
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    const view = el(name);
    if (view) view.classList.add("active");
    const renderer = global.TFM.views[name];
    if (renderer) renderer.render(view);
  }
  global.TFM.showView = showView;

  // --- Regla global de desplegables (<details>) ---------------------------------------------
  // Un <details> se abre por defecto SOLO si es el único de nivel superior en todo su panel de
  // vista (no hay nada que elegir). Si el panel tiene varios, se pliegan todos (el usuario decide
  // cuál abrir). El recuento es por PANEL (`.view`), no por contenedor inmediato: dos secciones
  // hermanas con un <details> cada una siguen siendo "varios" y se pliegan. Los <details> anidados
  // dentro de otro no cuentan (los gestiona su padre). `open` explícito en el HTML se respeta.
  function applyDetailsRule(node) {
    // Localiza el panel de vista que contiene el cambio y reevalúa TODOS sus <details> de nivel
    // superior de una vez. Se recalcula en cada pasada (no se memoiza): así, cuando una carga async
    // añade un segundo <details> al panel, el primero —abierto cuando era el único— se vuelve a
    // plegar. No se toca lo que el usuario haya abierto/cerrado a mano (data-user).
    const panel = (node.closest && node.closest(".view")) || node.ownerDocument.querySelector(".view.active") || document.body;
    const all = Array.from(panel.querySelectorAll("details"));
    const topLevel = all.filter((d) => !d.parentElement || !d.parentElement.closest("details"));
    topLevel.forEach((d) => {
      if (d.hasAttribute("data-keep-closed")) return;     // opt-out: nunca auto-abrir (queda plegado)
      if (d.dataset.user === "1") return;                 // toggle manual del usuario: respetar
      if (d.dataset.authorOpen === undefined) {
        d.dataset.authorOpen = d.hasAttribute("open") ? "1" : "0";  // recuerda la intención del HTML
        // Solo un toggle NO provocado por nosotros marca "decisión del usuario".
        d.addEventListener("toggle", () => { if (d.dataset.prog !== "1") d.dataset.user = "1"; });
      }
      if (d.dataset.authorOpen === "1") return;           // `open` explícito del autor: respetar
      const target = topLevel.length === 1;               // único en el panel → abierto; varios → plegado
      if (d.open !== target) {
        // `prog` marca que el cambio es nuestro; el evento `toggle` es asíncrono, así que se limpia
        // en el siguiente microtask (después de que el listener lo haya visto).
        d.dataset.prog = "1";
        d.open = target;
        Promise.resolve().then(() => { delete d.dataset.prog; });
      }
    });
  }
  // Reaplica la regla ante cualquier contenido nuevo (render de vistas, cargas async, sub-vistas).
  const detailsObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType === 1 && (node.querySelector?.("details") || node.matches?.("details"))) {
          applyDetailsRule(node);
        }
      }
    }
  });
  detailsObserver.observe(document.body, { childList: true, subtree: true });
  global.TFM.applyDetailsRule = applyDetailsRule;

  // --- Carga de datos base ---
  async function loadDefaults() {
    const data = await api("/api/defaults");
    state.defaults = data.settings || {};
    state.groups = data.groups || {};
    state.studyOptions = data.study_options || {};
    state.settingsOptions = data.settings_options || {};
    state.studyModelOptions = data.study_model_options || {};
    state.studyPortfolioOptions = data.study_portfolio_options || {};
    state.studyPhase3Options = data.study_phase3_options || {};
    state.fullStudyModelOptions = data.full_study_model_options || {};
    state.fullStudyPortfolioOptions = data.full_study_portfolio_options || {};
    state.fullStudyPhase3Options = data.full_study_phase3_options || {};
    state.fullStudyProfiles = data.full_study_profiles || [];
    state.studyOptionGroups = data.study_option_groups || {};
    state.fullStudyFixedSettings = data.full_study_fixed_settings || {};
    state.fullStudyStressSettings = data.full_study_stress_settings || {};
    state.presets = data.experiment_presets || {};
    state.profileLabels = data.profile_labels || {};
  }

  async function loadJobsAndRuns() {
    try {
      const data = await api("/api/runs");
      state.jobs = data.jobs || [];
      state.runs = (data.runs || []).slice().reverse();
    } catch (e) { /* silencioso: polling */ }
    // Notifica a las vistas interesadas
    if (global.TFM.views.console && el("console").classList.contains("active")) {
      global.TFM.views.console.refreshJobs();
    }
    if (global.TFM.views.results && el("results").classList.contains("active")) {
      global.TFM.views.results.refreshRuns();
    }
  }
  global.TFM.loadJobsAndRuns = loadJobsAndRuns;

  async function loadStudies() {
    try {
      const data = await api("/api/studies");
      state.studies = data.studies || [];
    } catch (e) { state.studies = []; }
  }
  global.TFM.loadStudies = loadStudies;

  // --- Pantalla de carga y errores de arranque ---
  function bootStep(text) {
    const node = el("boot-step");
    if (node) node.textContent = text;
  }
  function bootDone() {
    const boot = el("boot");
    if (boot) boot.classList.add("hidden");
  }
  function bootFail(context, error) {
    bootDone();
    const box = el("boot-error");
    if (!box) return;
    box.classList.remove("hidden");
    const msg = global.TFM && global.TFM.escapeHtml ? global.TFM.escapeHtml(error && error.message || String(error)) : String(error);
    box.innerHTML =
      `<h3>No se pudo cargar la consola</h3>` +
      `<div>Fallo al <strong>${context}</strong>: <code>${msg}</code></div>` +
      `<div class="hint">Comprueba que el servidor está en marcha: ejecuta <code>python main.py</code> ` +
      `(sin RUN_MODE) y abre <code>http://127.0.0.1:8765</code>. No abras el archivo HTML directamente.</div>`;
  }

  // --- Init ---
  async function init() {
    document.querySelectorAll("nav.app-nav button").forEach((b) => {
      b.onclick = () => showView(b.dataset.view);
    });
    try {
      bootStep("cargando configuración…");
      await loadDefaults();
      bootStep("cargando runs y estudios…");
      await loadJobsAndRuns();
      bootDone();
      showView("console");
      setInterval(loadJobsAndRuns, 2500);
    } catch (e) {
      bootFail("contactar con el servidor", e);
    }
  }

  // Red de seguridad: cualquier error no capturado deja de dar pantalla negra.
  global.addEventListener("error", (event) => {
    if (el("boot") && !el("boot").classList.contains("hidden")) {
      bootFail("iniciar la interfaz", event.error || event.message);
    }
  });

  document.addEventListener("DOMContentLoaded", init);
})(window);
