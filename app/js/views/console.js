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
  function displayValue(value) {
    if (Array.isArray(value)) return value.map(displayValue).join(" + ");
    if (value && typeof value === "object") {
      if ("target_size" in value) {
        return `${value.target_size} posiciones · pesos por meta-rank (máx. 2:1)`;
      }
      return Object.entries(value).map(([key, item]) => `${labelFor(key)}: ${displayValue(item)}`).join(" · ");
    }
    return String(value);
  }
  function decodeValue(raw, compound) {
    if (compound) return JSON.parse(raw);
    return raw === "true" ? true : raw === "false" ? false : Number.isNaN(Number(raw)) ? raw : Number(raw);
  }

  // --- Controles reutilizables ---
  function parameterControl(key, value) {
    const options = S().settingsOptions[key] || [value];
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

  function studyControls(lockedFullStudy = false, optionSource = null) {
    const source = optionSource || S().studyOptions;
    const rendered = new Set();
    const group = (title, names) => {
      const entries = names.filter((name) => source[name]);
      if (!entries.length) return "";
      entries.forEach((name) => rendered.add(name));
      const body = entries.map((name) => {
        const values = source[name];
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
      return `<details class="parameter-group"><summary>${escapeHtml(title)}</summary><div class="study-groups">${body}</div></details>`;
    };
    const groups = Object.entries(S().studyOptionGroups || {}).map(([title, names]) => group(title, names)).join("");
    const rest = Object.keys(source).filter((name) => !rendered.has(name));
    return groups + group("Otras variables", rest);
  }

  // --- Formularios ---
  function experimentalForm() {
    const profiles = (S().settingsOptions.profile || [])
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
      <p class="muted">Selecciona únicamente las variables y valores permitidos. La consola calculará el diseño antes de ejecutar.</p>
      <div class="formgrid">
        <label class="field">Nombre<input id="study-name" value="study-exploratorio"></label>
        <label class="field">Búsqueda de modelo
          <select id="study-search-mode" onchange="TFM.views.console.updateStudyCount()">
            <option value="directed">Dirigida (ejes aislados + greedy, como full study)</option>
            <option value="cartesian">Producto cartesiano (todas las combinaciones marcadas)</option>
          </select>
        </label>
        <label class="field">Hipótesis<textarea id="study-description"></textarea></label>
      </div>
      <label class="field checkbox"><input type="checkbox" id="study-robustness"> Incluir robustez completa (placebo por permutación + carteras aleatorias; reentrena el finalista, más lento)</label>
      <section class="parameter-group"><h4>Fases 1–2 · Datos, factores y modelo</h4>${studyControls(false, S().studyModelOptions)}</section>
      <section class="parameter-group"><h4>Fase 3 · Afinado de hiperparámetros</h4>${studyControls(false, S().studyPhase3Options)}</section>
      <section class="parameter-group"><h4>Fase 4 · Construcción de cartera</h4>${studyControls(false, S().studyPortfolioOptions)}</section>
      ${profileStudySettings()}
      ${stressStudySettings()}
      <div id="study-fixed-container">${unselectedStudyAxesSettings()}</div>
      <div id="study-count" class="notice"></div>
      <div class="actions"><button class="button primary" onclick="TFM.views.console.launchStudy()">Previsualizar y ejecutar</button></div>`;
  }

  function fixedStudySettings() {
    const entries = Object.entries(S().fullStudyFixedSettings || {});
    if (!entries.length) return "";
    return `<details class="parameter-group"><summary>Parámetros fijos y visibles</summary><div class="study-groups">${entries.map(([key, value]) =>
      `<div class="study-variable"><strong>${escapeHtml(labelFor(key))}</strong><label class="choice"><input type="checkbox" checked disabled>Fijo: ${escapeHtml(displayValue(value))}</label></div>`
    ).join("")}</div></details>`;
  }

  // Variables barribles del catálogo que el usuario NO marcó en study: se muestran de solo
  // lectura con su valor por defecto, igual que "Parámetros fijos" en full study.
  function unselectedStudyAxesSettings() {
    const source = S().studyOptions || {};
    const selected = new Set(Object.keys(selectedStudyVariables()));
    const entries = Object.keys(source)
      .filter((name) => !selected.has(name))
      .map((name) => [name, S().defaults[name]]);
    if (!entries.length) return "";
    return `<details class="parameter-group"><summary>Otras variables del catálogo, no barridas (fijas al valor por defecto)</summary><div class="study-groups">${entries.map(([key, value]) =>
      `<div class="study-variable"><strong>${escapeHtml(labelFor(key))}</strong><label class="choice"><input type="checkbox" checked disabled>Fijo: ${escapeHtml(displayValue(value))}</label></div>`
    ).join("")}</div></details>`;
  }

  function stressStudySettings() {
    const entries = Object.entries(S().fullStudyStressSettings || {});
    if (!entries.length) return "";
    return `<details class="parameter-group"><summary>Escenarios de estrés, no optimizables</summary><div class="study-groups">${entries.map(([key, value]) =>
      `<div class="study-variable"><strong>${escapeHtml(labelFor(key))}</strong><label class="choice"><input type="checkbox" checked disabled>Se informan todos: ${escapeHtml(displayValue(value))}</label></div>`
    ).join("")}</div></details>`;
  }

  function profileStudySettings() {
    const profiles = S().fullStudyProfiles || [];
    return `<details class="parameter-group"><summary>Fase 5 · Resultados por perfil, sin ganador</summary>
      <div class="study-groups">${profiles.map((profile) =>
        `<div class="study-variable"><strong>${escapeHtml(profile)}</strong><label class="choice"><input type="checkbox" checked disabled>${escapeHtml(S().profileLabels[profile] || profile)}</label></div>`
      ).join("")}</div></details>`;
  }

  function fullStudyForm() {
    return `<h3>Full study oficial</h3>
      <div class="formgrid">
        <label class="field">Nombre del estudio<input id="full-study-name" value="optimization-official"></label>
        <label class="field">Hipótesis<textarea id="full-study-hypothesis" placeholder="Ej.: Los bloques de calidad, crecimiento y riesgo mejoran el Rank-IC OOS frente al baseline."></textarea></label>
      </div>
      <p class="notice">Transparencia total: todas las variables barribles están marcadas y bloqueadas. Se ejecutarán todos sus valores permitidos; no se puede desmarcar ninguno en un full study.</p>
      <section class="parameter-group"><h4>Fases 1–2 · Datos, factores y modelo</h4>${studyControls(true, S().fullStudyModelOptions)}</section>
      <section class="parameter-group"><h4>Fase 3 · Afinado greedy, sin producto cartesiano</h4>${studyControls(true, S().fullStudyPhase3Options)}</section>
      <section class="parameter-group"><h4>Fase 4 · Construcción de cartera sobre el modelo congelado</h4>${studyControls(true, S().fullStudyPortfolioOptions)}</section>
      ${profileStudySettings()}
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

  function cartesianCount(variables) {
    const modelKeys = new Set(Object.keys(S().studyModelOptions || {}));
    let combinations = 1;
    Object.entries(variables).forEach(([key, values]) => {
      if (modelKeys.has(key)) combinations *= values.length;
    });
    return combinations;
  }

  function searchMode() {
    return el("study-search-mode")?.value || "directed";
  }

  function updateStudyCount() {
    const node = el("study-count");
    if (!node) return;
    const variables = selectedStudyVariables();
    if (!Object.keys(variables).length) {
      node.textContent = "Selecciona al menos una variable.";
      return;
    }
    if (searchMode() === "cartesian") {
      const combinations = cartesianCount(variables);
      const warn = combinations > 500 ? " ⚠️ Por encima del límite (500): el servidor rechazará el envío." : "";
      node.textContent = `Producto cartesiano (Fases 1–2): ${combinations} combinaciones de las variables de modelo marcadas. Después se ejecutan afinado, cartera, perfiles y validación igual que en el modo dirigido.${warn}`;
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
      const mode = searchMode();
      const confirmMsg = mode === "cartesian"
        ? `Producto cartesiano: ${cartesianCount(variables)} combinaciones de modelo. Después se añaden afinado, cartera, perfiles y validación. ¿Continuar?`
        : `Fases 1–2: aproximadamente ${directedCount(variables).total} runs. Después se añaden afinado, cartera, perfiles y validación. ¿Continuar?`;
      if (!confirm(confirmMsg)) return;
      const job = await api("/api/study", {
        settings: S().defaults,
        variables,
        search_mode: mode,
        study: { name: el("study-name").value, description: el("study-description").value, include_robustness: el("study-robustness").checked },
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
    // Recalcula el diseño y los fijos al marcar/desmarcar variables del study
    container.addEventListener("change", (event) => {
      if (event.target.matches("[data-study]")) {
        updateStudyCount();
        const fixedContainer = el("study-fixed-container");
        if (fixedContainer) fixedContainer.innerHTML = unselectedStudyAxesSettings();
      }
    });
    refreshJobs();
  }

  global.TFM.views.console = {
    render, refreshJobs, showForm, applyPreset, launchExperimental,
    launchStudy, launchOptimization, updateStudyCount,
  };
})(window);
