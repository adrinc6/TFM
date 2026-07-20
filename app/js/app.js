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

  // --- Carga de datos base ---
  async function loadDefaults() {
    const data = await api("/api/defaults");
    state.defaults = data.settings || {};
    state.groups = data.groups || {};
    state.studyOptions = data.study_options || {};
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
