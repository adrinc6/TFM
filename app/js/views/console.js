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
    if (SET_VALUED_AXES[key]) return settingItemPicker(key, value);
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
      .map(([title, names]) => {
        // Los artefactos booleanos se colapsan en un único picker de checks al final del grupo.
        const plain = names.filter((k) => !isArtifactToggle(k));
        const hasArtifacts = names.some(isArtifactToggle);
        const controls = plain.map((k) => parameterControl(k, S().defaults[k])).join("");
        const artifacts = hasArtifacts
          ? artifactTogglesPicker("data-setting-toggle", (n) => Boolean(S().defaults[n]))
          : "";
        return `<section class="parameter-group"><h4>${escapeHtml(title)}</h4><div class="formgrid">${controls}${artifacts}</div></section>`;
      })
      .join("");
  }

  function presetSelect(kind, title) {
    const items = (S().presets[kind] || [])
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
      .join("");
    return `<label class="field">${title}<select data-preset="${kind}" onchange="TFM.views.console.applyPreset('${kind}', this.value)"><option value="">Seleccionar…</option>${items}</select></label>`;
  }

  // Ejes de conjunto: en el study manual se eligen por ITEM atómico (agente/bloque/familia) y el
  // backend aplica la ablación full-minus-one sobre la selección. En el full study bloqueado se
  // conservan las combinaciones fijas del catálogo.
  const SET_VALUED_AXES = {
    enabled_agents: () => S().agentCatalog,
    enabled_feature_blocks: () => S().featureBlockCatalog,
    enabled_model_families: () => S().modelFamilyCatalog,
  };

  function atomicPicker(name) {
    const items = (SET_VALUED_AXES[name]() || []);
    const choices = items
      .map((item) => `<label class="choice"><input type="checkbox" data-study-item="${name}" value="${escapeHtml(item)}" checked>${escapeHtml(labelFor(item))}</label>`)
      .join("");
    return `<div class="study-variable"><strong>${labelFor(name)}</strong>
      <div class="choices">${choices}</div>
      <small class="muted">Se prueban el conjunto marcado y sus ablaciones (quitando cada elemento uno a uno).</small></div>`;
  }

  // Artefactos activables: grupo de booleanos presentado como un único picker de checks (marcar =
  // activar), en vez de un par false/true por cada uno.
  const artifactToggles = () => S().artifactToggles || [];
  const isArtifactToggle = (name) => artifactToggles().includes(name);

  // Choices de artefactos. `attr` distingue el origen (data-setting-toggle en experimental,
  // data-study-toggle en study); `isOn(name)` decide el estado inicial de cada check.
  function artifactChoices(attr, isOn) {
    return artifactToggles()
      .map((name) => `<label class="choice"><input type="checkbox" ${attr}="${name}" ${isOn(name) ? "checked" : ""}>${escapeHtml(labelFor(name))}</label>`)
      .join("");
  }

  // Experimental: cada artefacto marcado => True; desmarcado => False.
  function artifactTogglesPicker(attr, isOn) {
    return `<div class="field set-picker"><strong>Artefactos activables</strong><div class="choices">${artifactChoices(attr, isOn)}</div></div>`;
  }

  // Experimental: un run usa UN conjunto concreto (no barre). Cada elemento es un check; por
  // defecto todos marcados. Se envía como lista plana de los marcados.
  function settingItemPicker(key, value) {
    const items = (SET_VALUED_AXES[key]() || []);
    const selected = new Set(Array.isArray(value) ? value : items);
    const choices = items
      .map((item) => `<label class="choice"><input type="checkbox" data-setting-item="${key}" value="${escapeHtml(item)}" ${selected.has(item) ? "checked" : ""}>${escapeHtml(labelFor(item))}</label>`)
      .join("");
    return `<div class="field set-picker"><strong>${labelFor(key)}</strong><div class="choices">${choices}</div></div>`;
  }

  function studyControls(lockedFullStudy = false, optionSource = null) {
    const source = optionSource || S().studyOptions;
    const rendered = new Set();
    const group = (title, names) => {
      const entries = names.filter((name) => source[name]);
      if (!entries.length) return "";
      entries.forEach((name) => rendered.add(name));
      // Artefactos activables: en el study manual se colapsan en un único picker (marcar = barrer
      // [False, True] en ese eje; desmarcar = fijo en False). En full study van con el resto.
      const artifactEntries = lockedFullStudy ? [] : entries.filter(isArtifactToggle);
      const artifactsPicker = artifactEntries.length
        ? `<div class="study-variable"><strong>Artefactos activables</strong>
             <div class="choices">${artifactChoices("data-study-toggle", () => false)}</div>
             <small class="muted">Marcar = probar con y sin ese artefacto; sin marcar = fijo en False.</small></div>`
        : "";
      const body = entries.filter((name) => !artifactEntries.includes(name)).map((name) => {
        if (!lockedFullStudy && SET_VALUED_AXES[name]) return atomicPicker(name);
        const values = source[name];
        const defaultValue = S().defaults[name];
        const choices = values
          .map((value, index) => {
            const compound = isCompound(value);
            // Por defecto solo el valor baseline queda marcado (un único run esencial); el usuario
            // añade los valores que quiera barrer. En full study se marca y bloquea todo.
            const isDefault = defaultValue !== undefined
              ? JSON.stringify(value) === JSON.stringify(defaultValue)
              : index === 0;
            const checked = lockedFullStudy || isDefault;
            const locked = lockedFullStudy ? "disabled" : "";
            return `<label class="choice"><input type="checkbox" data-study="${name}" value="${escapeHtml(encodedValue(value))}" data-json="${compound}" ${checked ? "checked" : ""} ${locked}>${escapeHtml(displayValue(value))}</label>`;
          })
          .join("");
        return `<div class="study-variable"><strong>${labelFor(name)}</strong><div class="choices">${choices}</div></div>`;
      }).join("");
      return `<details class="parameter-group"><summary>${escapeHtml(title)}</summary><div class="study-groups">${body}${artifactsPicker}</div></details>`;
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
      <div class="formgrid formgrid-3">
        <label class="field">Nombre<input id="study-name" value="study-exploratorio"></label>
        <label class="field">Búsqueda de modelo
          <select id="study-search-mode" onchange="TFM.views.console.updateStudyCount()">
            <option value="directed">Dirigida (ejes aislados + greedy, como full study)</option>
            <option value="cartesian">Producto cartesiano (todas las combinaciones marcadas)</option>
          </select>
        </label>
        <label class="field">Hipótesis<input id="study-description" placeholder="Opcional"></label>
      </div>
      <div id="study-count" class="notice"></div>
      <section class="parameter-group"><h4>Fases 1–2 · Datos, factores y modelo</h4>${studyControls(false, S().studyModelOptions)}</section>
      <section class="parameter-group"><h4>Fase 3 · Afinado de hiperparámetros</h4>${studyControls(false, S().studyPhase3Options)}</section>
      <section class="parameter-group"><h4>Fase 4 · Construcción de cartera</h4>${studyControls(false, S().studyPortfolioOptions)}</section>
      ${robustnessComponentsSettings()}
      ${profileStudySettings()}
      <div class="actions"><button class="button primary" onclick="TFM.views.console.launchStudy()">Previsualizar y ejecutar</button></div>`;
  }

  function fixedStudySettings() {
    const entries = Object.entries(S().fullStudyFixedSettings || {});
    if (!entries.length) return "";
    return `<details class="parameter-group"><summary>Parámetros fijos y visibles</summary><div class="study-groups">${entries.map(([key, value]) =>
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

  // Fase de robustez/estrés: cada componente es un checkbox independiente; solo corre lo marcado.
  // El estrés de cartera y el de costes despliegan además los ejes/valores concretos que se barren.
  function robustnessComponentsSettings() {
    const components = S().robustnessComponents || [];
    if (!components.length) return "";
    const stress = S().fullStudyStressSettings || {};
    const costAxes = ["commission_bps", "slippage_bps"];
    const detailFor = (key) => {
      if (key === "cost_stress") {
        return costAxes.filter((a) => stress[a]).map((a) =>
          `<div class="study-variable"><strong>${labelFor(a)}</strong><label class="choice"><input type="checkbox" checked disabled>Se informan todos: ${escapeHtml(displayValue(stress[a]))}</label></div>`).join("");
      }
      if (key === "portfolio_stress") {
        return Object.keys(stress).filter((a) => !costAxes.includes(a)).map((a) => {
          const choices = (stress[a] || []).map((value, index) =>
            `<label class="choice"><input type="checkbox" data-study="${a}" value="${escapeHtml(encodedValue(value))}" data-json="${isCompound(value)}" ${index === 0 ? "checked" : ""}>${escapeHtml(displayValue(value))}</label>`).join("");
          return `<div class="study-variable"><strong>${labelFor(a)}</strong><div class="choices">${choices}</div></div>`;
        }).join("");
      }
      return "";
    };
    const rows = components.map((c) => {
      const detail = detailFor(c.key);
      const hint = c.cost === "caro" ? " <span class=\"tag warn\">reentrena / lento</span>" : "";
      return `<div class="study-variable">
        <label class="choice"><input type="checkbox" data-robustness="${c.key}"> <strong>${escapeHtml(c.label || c.key)}</strong>${hint}</label>
        ${detail ? `<div class="study-groups">${detail}</div>` : ""}</div>`;
    }).join("");
    return `<details class="parameter-group"><summary>Fase de robustez y estrés (opcional) · marca solo lo que quieras ejecutar</summary>
      <div class="study-groups">${rows}</div></details>`;
  }

  function profileStudySettings(locked = false) {
    const profiles = S().fullStudyProfiles || [];
    // Study manual: el usuario elige qué perfiles ejecutar; por defecto solo el de referencia
    // (`balanced`). Full study: los ocho, bloqueados.
    const choices = profiles.map((profile) => {
      const checked = locked || profile === "balanced";
      return `<label class="choice"><input type="checkbox" data-study="profile" value="${escapeHtml(profile)}" data-json="false" ${checked ? "checked" : ""} ${locked ? "disabled" : ""}>${escapeHtml(S().profileLabels[profile] || profile)}</label>`;
    }).join("");
    return `<details class="parameter-group"><summary>Fase 5 · Perfiles de inversor a ejecutar (salidas, sin ganador)</summary>
      <div class="study-variable"><strong>profile</strong><div class="choices">${choices}</div></div></details>`;
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
      ${profileStudySettings(true)}
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
    // Ejes de conjunto: se envían como la lista plana de elementos marcados. Cada clave presente en
    // el formulario se reinicia a [] (aunque no haya ninguno marcado), no al valor por defecto.
    const setValued = {};
    document.querySelectorAll("[data-setting-item]").forEach((i) => {
      (setValued[i.dataset.settingItem] ??= []);
      if (i.checked) setValued[i.dataset.settingItem].push(i.value);
    });
    Object.assign(settings, setValued);
    // Artefactos activables: marcado => True, presente y desmarcado => False.
    document.querySelectorAll("[data-setting-toggle]").forEach((i) => {
      settings[i.dataset.settingToggle] = i.checked;
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
    // Ejes de conjunto: se recogen como lista PLANA de items atómicos (strings). El backend los
    // expande a la ablación full-minus-one antes de barrerlos.
    document.querySelectorAll("[data-study-item]").forEach((input) => {
      if (!input.checked) return;
      (variables[input.dataset.studyItem] ??= []).push(input.value);
    });
    // Artefactos: marcar uno significa barrer [False, True] en ese eje; sin marcar queda fijo en su
    // valor por defecto (False) y no se añade como variable.
    document.querySelectorAll("[data-study-toggle]").forEach((input) => {
      if (input.checked) variables[input.dataset.studyToggle] = [false, true];
    });
    return variables;
  }

  // Clasifica cada eje marcado en su fase, contando cuántos VALORES no-baseline aporta (los que
  // generan un run adicional). Los ejes de conjunto (agentes/bloques/familias) generan una ablación:
  // sus runs extra = nº de elementos quitables (len - 1).
  function phaseBreakdown(variables) {
    const modelKeys = new Set(Object.keys(S().studyModelOptions || {}));
    const phase3Keys = new Set(Object.keys(S().studyPhase3Options || {}));
    const portfolioKeys = new Set(Object.keys(S().studyPortfolioOptions || {}));
    const stressKeys = new Set(Object.keys(S().fullStudyStressSettings || {}));
    const b = { model: 0, phase3: 0, portfolio: 0, stress: 0, profiles: 0 };
    Object.entries(variables).forEach(([key, values]) => {
      let extra;
      if (SET_VALUED_AXES[key]) extra = values.length > 1 ? values.length : 0;  // ablación
      else extra = values.filter((v) => JSON.stringify(v) !== JSON.stringify(S().defaults[key])).length;
      if (key === "profile") b.profiles = values.length;
      else if (phase3Keys.has(key)) b.phase3 += extra;
      else if (stressKeys.has(key)) b.stress += extra;
      else if (portfolioKeys.has(key)) b.portfolio += extra;
      else if (modelKeys.has(key)) b.model += extra;
    });
    return b;
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
    const b = phaseBreakdown(variables);
    const profiles = b.profiles || 0;
    const stressText = b.stress ? `Fase 4b/c estrés: ${b.stress} runs. ` : "";
    const robustness = Array.from(document.querySelectorAll("[data-robustness]:checked")).length;
    const robText = robustness ? `Robustez: ${robustness} componente(s). ` : "";
    if (searchMode() === "cartesian") {
      const combinations = cartesianCount(variables);
      const warn = combinations > 500 ? " ⚠️ Por encima del límite (500): el servidor rechazará el envío." : "";
      node.textContent = `Fases 1–2 (cartesiano): ${combinations} combinaciones de modelo. Fase 3 afinado: +${b.phase3} runs. Fase 4 cartera: +${b.portfolio} runs. ${stressText}Fase 5 perfiles: ${profiles} runs. ${robText}Más el finalista y la validación reservada.${warn}`;
      return;
    }
    // Dirigido: baseline (1) + un run por cada valor no-baseline de modelo, luego greedy top-2.
    const model = 1 + b.model;
    node.textContent = `Fases 1–2 (dirigido): ${model} runs de modelo (baseline + ${b.model} variantes aisladas) y greedy. Fase 3 afinado: +${b.phase3} runs. Fase 4 cartera: +${b.portfolio} runs. ${stressText}Fase 5 perfiles: ${profiles} runs. ${robText}Más el finalista y la validación reservada.`;
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
      const b = phaseBreakdown(variables);
      const modelRuns = mode === "cartesian" ? cartesianCount(variables) : 1 + b.model;
      const confirmMsg = `Fases 1–2: ${modelRuns} runs de modelo. Fase 3: +${b.phase3}. Fase 4: +${b.portfolio}. Fase 5 perfiles: ${b.profiles}. Más finalista, robustez marcada y validación reservada. ¿Continuar?`;
      if (!confirm(confirmMsg)) return;
      const robustnessComponents = Array.from(
        document.querySelectorAll("[data-robustness]:checked"), (i) => i.dataset.robustness);
      const job = await api("/api/study", {
        settings: S().defaults,
        variables,
        search_mode: mode,
        study: {
          name: el("study-name").value,
          description: el("study-description").value,
          robustness_components: robustnessComponents,
        },
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
      if (event.target.matches("[data-study], [data-study-item], [data-study-toggle], [data-robustness]")) {
        updateStudyCount();
      }
    });
    refreshJobs();
  }

  global.TFM.views.console = {
    render, refreshJobs, showForm, applyPreset, launchExperimental,
    launchStudy, launchOptimization, updateStudyCount,
  };
})(window);
