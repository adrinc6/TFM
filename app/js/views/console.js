/* Vista Consola: lanzar Experimental / Study / Optimization. Conserva TODOS los controles
   guiados del SPA anterior (presets, opciones por categoría, rejilla de variables de study,
   perfiles, diseño dirigido) y solo mejora su presentación. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, state } = global.TFM;
  const S = () => global.TFM.state;

  const labelFor = (name) => name.replaceAll("_", " ");
  const isCompound = (value) => Array.isArray(value) || (value && typeof value === "object");
  const encodedValue = (value) => isCompound(value) ? JSON.stringify(value) : String(value);
  const displayValue = (value) => Array.isArray(value) ? value.join(" + ") : String(value);
  function decodeValue(raw, compound) {
    if (compound) return JSON.parse(raw);
    return raw === "true" ? true : raw === "false" ? false : Number.isNaN(Number(raw)) ? raw : Number(raw);
  }

  // --- Controles reutilizables ---
  function parameterControl(key, value) {
    const options = S().studyOptions[key] || [value];
    const opts = options
      .map((o) => {
        const compound = isCompound(o);
        const selected = JSON.stringify(o) === JSON.stringify(value);
        return `<option value="${escapeHtml(encodedValue(o))}" data-json="${compound}" ${selected ? "selected" : ""}>${escapeHtml(displayValue(o))}</option>`;
      })
      .join("");
    return `<label class="field">${labelFor(key)}<select data-setting="${key}">${opts}</select></label>`;
  }

  function groupedInputs() {
    return Object.entries(S().groups)
      .map(
        ([title, names]) =>
          `<section class="parameter-group"><h4>${escapeHtml(title)}</h4><div class="formgrid">${names
            .map((k) => parameterControl(k, S().defaults[k]))
            .join("")}</div></section>`
      )
      .join("");
  }

  function presetSelect(kind, title) {
    const items = (S().presets[kind] || [])
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
      .join("");
    return `<label class="field">${title}<select data-preset="${kind}" onchange="TFM.views.console.applyPreset('${kind}', this.value)"><option value="">Seleccionar…</option>${items}</select></label>`;
  }

  function studyControls(lockedFullStudy = false) {
    const rendered = new Set();
    const group = (title, names) => {
      const entries = names.filter((name) => S().studyOptions[name]);
      if (!entries.length) return "";
      entries.forEach((name) => rendered.add(name));
      const body = entries.map((name) => {
        const values = S().studyOptions[name];
        const choices = values
          .map((value, index) => {
            const compound = isCompound(value);
            const checked = lockedFullStudy || index === 0;
            const locked = lockedFullStudy ? "disabled" : "";
            return `<label class="choice"><input type="checkbox" data-study="${name}" value="${escapeHtml(encodedValue(value))}" data-json="${compound}" ${checked ? "checked" : ""} ${locked}>${escapeHtml(displayValue(value))}</label>`;
          })
          .join("");
        return `<div class="study-variable"><strong>${labelFor(name)}</strong><div class="choices">${choices}</div></div>`;
      }).join("");
      return `<details class="parameter-group" open><summary>${escapeHtml(title)}</summary><div class="study-groups">${body}</div></details>`;
    };
    const groups = Object.entries(S().studyOptionGroups || {}).map(([title, names]) => group(title, names)).join("");
    const rest = Object.keys(S().studyOptions).filter((name) => !rendered.has(name));
    return groups + group("Otras variables", rest);
  }

  // --- Formularios ---
  function experimentalForm() {
    const profiles = (S().studyOptions.profile || [])
      .map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(S().profileLabels[p] || p)}</option>`)
      .join("");
    return `<h3>Run experimental</h3>
      <p class="muted">Elige conjuntos coherentes; los controles solo ofrecen valores admitidos por el estudio.</p>
      <div class="formgrid">
        <label class="field">Etiqueta<input id="exp-label" value="Run experimental"></label>
        <label class="field">Modo<select id="exp-mode"><option>full</option><option>dataset</option><option>features</option><option>agents</option><option>backtest</option></select></label>
        <label class="field">Descripción<textarea id="exp-description"></textarea></label>
      </div>
      <section class="parameter-group"><h4>Conjuntos relacionados</h4>
        <div class="formgrid">
          ${presetSelect("training", "Periodo y entrenamiento")}
          ${presetSelect("portfolio", "Cartera")}
          ${presetSelect("features", "Bloque de artefactos")}
          <label class="field">Perfil de inversor<select data-setting="profile">${profiles}</select></label>
        </div>
      </section>
      <details><summary>Opciones permitidas por categoría</summary>${groupedInputs()}</details>
      <div class="actions">
        <button class="button primary" onclick="TFM.views.console.launchExperimental()">Ejecutar</button>
        <button class="button" onclick="TFM.views.console.showForm('experimental')">Restablecer</button>
      </div>`;
  }

  function studyForm() {
    return `<h3>Nuevo study</h3>
      <p class="muted">Selecciona únicamente las variables y valores permitidos. La consola calculará el diseño dirigido antes de ejecutar.</p>
      <div class="formgrid">
        <label class="field">Nombre<input id="study-name" value="study-exploratorio"></label>
        <label class="field">Tipo<select id="study-kind"><option value="exploratory">Exploratorio</option><option value="ablation">Ablation</option><option value="custom_optimization">Optimización personalizada</option></select></label>
        <label class="field">Modo<select id="study-mode"><option>full</option><option>agents</option></select></label>
        <label class="field">Hipótesis<textarea id="study-description"></textarea></label>
      </div>
      <label class="field checkbox"><input type="checkbox" id="study-robustness"> Incluir robustez completa (placebo por permutación + carteras aleatorias; reentrena el finalista, más lento)</label>
      <section class="parameter-group"><h4>Variables y valores a combinar</h4>${studyControls()}</section>
      <div id="study-count" class="notice"></div>
      <div class="actions"><button class="button primary" onclick="TFM.views.console.launchStudy()">Previsualizar y ejecutar</button></div>`;
  }

  function fixedStudySettings() {
    const entries = Object.entries(S().fullStudyFixedSettings || {});
    if (!entries.length) return "";
    return `<details class="parameter-group" open><summary>Parámetros fijos y visibles</summary><div class="study-groups">${entries.map(([key, value]) =>
      `<div class="study-variable"><strong>${escapeHtml(labelFor(key))}</strong><label class="choice"><input type="checkbox" checked disabled>Fijo: ${escapeHtml(displayValue(value))}</label></div>`
    ).join("")}</div></details>`;
  }

  function stressStudySettings() {
    const entries = Object.entries(S().fullStudyStressSettings || {});
    if (!entries.length) return "";
    return `<details class="parameter-group" open><summary>Escenarios de estrés, no optimizables</summary><div class="study-groups">${entries.map(([key, value]) =>
      `<div class="study-variable"><strong>${escapeHtml(labelFor(key))}</strong><label class="choice"><input type="checkbox" checked disabled>Se informan todos: ${escapeHtml(displayValue(value))}</label></div>`
    ).join("")}</div></details>`;
  }

  function fullStudyForm() {
    return `<h3>Full study oficial</h3>
      <div class="formgrid">
        <label class="field">Nombre del estudio<input id="full-study-name" value="optimization-official"></label>
        <label class="field">Hipótesis<textarea id="full-study-hypothesis" placeholder="Ej.: Los bloques de calidad, crecimiento y riesgo mejoran el Rank-IC OOS frente al baseline."></textarea></label>
      </div>
      <p class="notice">Transparencia total: todas las variables barribles están marcadas y bloqueadas. Se ejecutarán todos sus valores permitidos; no se puede desmarcar ninguno en un full study.</p>
      <section class="parameter-group"><h4>Variables que se modifican</h4>${studyControls(true)}</section>
      <section class="parameter-group"><h4>Variables que no se modifican</h4>${fixedStudySettings()}</section>
      <section class="parameter-group"><h4>Costes de ejecución</h4>${stressStudySettings()}</section>
      <div class="actions"><button class="button primary" onclick="TFM.views.console.launchOptimization()">Lanzar full study oficial</button>
      <button class="button" onclick="TFM.views.console.showForm('full-study')">Restablecer vista</button></div>`;
  }

  function showForm(type) {
    const form = el("console-form");
    form.classList.remove("hidden");
    form.innerHTML = type === "experimental" ? experimentalForm() : type === "full-study" ? fullStudyForm() : studyForm();
    if (type === "study") updateStudyCount();
  }

  // --- Presets y recogida de settings ---
  function applyPreset(kind, id) {
    const preset = (S().presets[kind] || []).find((p) => p.id === id);
    if (!preset) return;
    Object.entries(preset.overrides).forEach(([key, value]) => {
      document.querySelectorAll(`[data-setting="${key}"]`).forEach((input) => (input.value = String(value)));
    });
  }

  function collectSettings() {
    const settings = { ...S().defaults };
    document.querySelectorAll("[data-setting]").forEach((i) => {
      const option = i.options && i.options[i.selectedIndex];
      settings[i.dataset.setting] = option && option.dataset.json === "true" ? JSON.parse(i.value) : i.value;
    });
    return settings;
  }

  // --- Diseño dirigido del study (no producto cartesiano) ---
  function selectedStudyVariables() {
    const variables = {};
    document.querySelectorAll("[data-study]").forEach((input) => {
      if (!input.checked) return;
      const key = input.dataset.study;
      const raw = input.value;
      const value = decodeValue(raw, input.dataset.json === "true");
      (variables[key] ??= []).push(value);
    });
    return variables;
  }

  function directedCount(variables) {
    let phase1 = 1;
    let axes = 0;
    Object.entries(variables).forEach(([key, values]) => {
      const changes = values.filter((v) => String(v) !== String(S().defaults[key]));
      phase1 += changes.length;
      if (changes.length) axes++;
    });
    return { phase1, phase2: 1 + axes, total: phase1 + 1 + axes };
  }

  function updateStudyCount() {
    const node = el("study-count");
    if (!node) return;
    const variables = selectedStudyVariables();
    if (!Object.keys(variables).length) {
      node.textContent = "Selecciona al menos una variable.";
      return;
    }
    const n = directedCount(variables);
    node.textContent = `Diseño dirigido (solo fases 1–2): ${n.phase1} runs aislados y hasta ${n.phase2} combinaciones. Después se ejecutan afinado, cartera, perfiles y validación; no es un máximo total ni un producto cartesiano.`;
  }

  // --- Lanzadores ---
  async function launchExperimental() {
    try {
      const job = await api("/api/experimental", {
        settings: collectSettings(),
        mode: el("exp-mode").value,
        label: el("exp-label").value,
        description: el("exp-description").value,
      });
      notify(`Trabajo ${job.job_id} iniciado.`);
      global.TFM.loadJobsAndRuns();
    } catch (e) { notify(e.message, true); }
  }

  async function launchStudy() {
    try {
      const variables = selectedStudyVariables();
      if (!Object.keys(variables).length) throw new Error("Selecciona al menos una variable.");
      const n = directedCount(variables);
      if (!confirm(`Fases 1–2: aproximadamente ${n.total} runs. Después se añaden afinado, cartera, perfiles y validación. ¿Continuar?`)) return;
      const job = await api("/api/study", {
        settings: S().defaults,
        mode: el("study-mode").value,
        variables,
        study: { name: el("study-name").value, kind: el("study-kind").value, description: el("study-description").value, include_robustness: el("study-robustness").checked },
      });
      notify(`Study ${job.job_id} iniciado.`);
      global.TFM.loadJobsAndRuns();
    } catch (e) { notify(e.message, true); }
  }

  async function launchOptimization() {
    if (!confirm("La optimization oficial ejecutará Fase 1, Fase 2 dirigida, afinado y validación reservada. ¿Continuar?")) return;
    try {
      const job = await api("/api/optimization", {
        settings: S().defaults,
        study: {
          name: el("full-study-name")?.value || "optimization-official",
          hypothesis: el("full-study-hypothesis")?.value || "",
        },
      });
      notify(`Optimization ${job.job_id} iniciada.`);
      global.TFM.loadJobsAndRuns();
    } catch (e) { notify(e.message, true); }
  }

  function notify(message, isError) {
    const bar = el("console-notice");
    if (!bar) { alert(message); return; }
    bar.className = isError ? "notice" : "notice";
    bar.textContent = message;
    bar.classList.remove("hidden");
  }

  // --- Historial de jobs ---
  function jobTag(status) {
    const cls = status === "succeeded" ? "pos" : status === "failed" ? "neg" : "warn";
    return `<span class="tag ${cls}">${escapeHtml(status)}</span>`;
  }
  function refreshJobs() {
    const node = el("console-jobs");
    if (!node) return;
    const jobs = S().jobs;
    node.innerHTML = jobs.length
      ? jobs.map((j) => `<p>${jobTag(j.status)} ${escapeHtml(j.name)} ${j.result ? "→ " + escapeHtml(typeof j.result === "string" ? j.result : JSON.stringify(j.result)) : ""}${j.error ? `<br><small class="muted">${escapeHtml(String(j.error).slice(0, 300))}</small>` : ""}</p>`).join("")
      : `<p class="muted">Sin trabajos de esta sesión.</p>`;
  }

  // --- Render principal ---
  function render(container) {
    container.innerHTML = `
      <div class="cards">
        <article class="card">
          <h3>Experimental</h3>
          <p>Ejecuta una configuración libre y trazable.</p>
          <button class="button primary" onclick="TFM.views.console.showForm('experimental')">Nuevo run</button>
        </article>
        <article class="card">
          <h3>Study</h3>
          <p>Compara combinaciones creadas a partir de una hipótesis.</p>
          <button class="button primary" onclick="TFM.views.console.showForm('study')">Crear study</button>
        </article>
        <article class="card">
          <h3>Full study</h3>
          <p>Todos los ejes, fases dirigidas, afinado y validación reservada.</p>
          <button class="button primary" onclick="TFM.views.console.showForm('full-study')">Revisar y lanzar</button>
        </article>
      </div>
      <div id="console-notice" class="notice hidden"></div>
      <div id="console-form" class="panel hidden" style="margin-top:18px"></div>
      <div class="grid">
        <section class="panel"><h3>Historial de trabajos</h3><div id="console-jobs" class="scroll"></div></section>
        <section class="panel">
          <h3>Reglas metodológicas</h3>
          <p class="muted">La selección oficial se basa en rank-IC fuera de muestra y estabilidad. La rentabilidad se presenta como consecuencia, no como criterio de selección.</p>
          <p class="notice">Las ejecuciones <em>experimentales</em> y los studies exploratorios no eligen una configuración oficial.</p>
        </section>
      </div>`;
    // Recalcula el diseño dirigido al marcar/desmarcar variables
    container.addEventListener("change", (event) => {
      if (event.target.matches("[data-study]")) updateStudyCount();
    });
    refreshJobs();
  }

  global.TFM.views.console = {
    render, refreshJobs, showForm, applyPreset, launchExperimental,
    launchStudy, launchOptimization, updateStudyCount,
  };
})(window);
