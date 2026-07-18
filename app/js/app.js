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

  // --- Init ---
  async function init() {
    document.querySelectorAll("nav.app-nav button").forEach((b) => {
      b.onclick = () => showView(b.dataset.view);
    });
    try {
      await loadDefaults();
    } catch (e) {
      el("console").innerHTML = `<div class="notice">No se pudieron cargar los valores por defecto: ${global.TFM.escapeHtml(e.message)}</div>`;
    }
    await loadJobsAndRuns();
    showView("console");
    setInterval(loadJobsAndRuns, 2500);
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
