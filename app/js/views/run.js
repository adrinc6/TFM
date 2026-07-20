/* Detalle de un run: métricas cabecera + pestañas Resumen / Rendimiento / Aprendizaje /
   Cartera / Trades / Stocks / Ticker. Cada pestaña delega en su vista de análisis. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct, metricGrid, table } = global.TFM;

  let current = null; // { runId, manifest }

  const TABS = [
    ["resumen", "Resumen"],
    ["rendimiento", "Rendimiento"],
    ["aprendizaje", "Aprendizaje"],
    ["cartera", "Cartera"],
    ["trades", "Trades"],
    ["stocks", "Stocks"],
  ];

  // `format`: "num" (decimales crudos, p. ej. rank-IC) o "pct" (×100 con " %"). `signed` colorea.
  // `hintKey` añade un tooltip explicativo de la métrica.
  function metricCard(value, label, { decimals = 2, format = "num", signed = false, hintKey = "" } = {}) {
    const cls = signed && typeof value === "number" ? (value >= 0 ? " pos" : " neg") : "";
    const shown = format === "pct" ? pct(value, decimals) : fmt(value, decimals);
    const hint = global.TFM.metricHint(hintKey);
    const labelHtml = hint
      ? `<span title="${escapeHtml(hint)}">${escapeHtml(label)} <span class="hint-dot">?</span></span>`
      : escapeHtml(label);
    return `<div class="card"><div class="metric${cls}">${shown}</div><div class="metric-label">${labelHtml}</div></div>`;
  }

  // Aplana `effective_settings` a tarjetitas. Valores no escalares se muestran compactos.
  function settingsCards(settings) {
    const items = Object.entries(settings || {}).map(([key, raw]) => {
      let value;
      if (raw === null || raw === undefined) value = "—";
      else if (typeof raw === "boolean") value = raw ? "sí" : "no";
      else if (typeof raw === "object") value = JSON.stringify(raw);
      else value = String(raw);
      return { value, label: key };
    });
    return items.length ? metricGrid(items) : `<p class="muted">Sin configuración registrada.</p>`;
  }

  // Robustez multi-era: bootstrap por bloques (IC 95 %) + leave-one-year-out. Es la evidencia
  // honesta sobre si el modelo aprende (una señal cuyo IC cruza el cero es indistinguible de cero).
  function robustnessBlock(robustness) {
    const boot = (robustness || {}).block_bootstrap;
    const loyo = (robustness || {}).leave_one_year_out || [];
    if (!boot && !loyo.length) return "";
    let cards = "";
    if (boot) {
      const crossesZero = boot.ci_low <= 0 && boot.ci_high >= 0;
      cards = global.TFM.metricGrid([
        { value: fmt(boot.mean, 3), label: "rank-IC medio", hintKey: "rank_ic" },
        { value: `[${fmt(boot.ci_low, 3)}, ${fmt(boot.ci_high, 3)}]`, label: "IC 95 % (bootstrap por bloques)",
          cls: crossesZero ? "neg" : "pos" },
        { value: String(boot.n_cohorts ?? "—"), label: "cohortes" },
        { value: String(boot.block_size ?? "—"), label: "tamaño de bloque" },
      ]);
      if (crossesZero) {
        cards += `<p class="muted" style="margin-top:8px">El intervalo de confianza cruza el cero: la señal es <strong>estadísticamente indistinguible de cero</strong> (no hay evidencia robusta de aprendizaje).</p>`;
      }
    }
    let loyoTable = "";
    if (loyo.length) {
      const negatives = loyo.filter((r) => (r.rank_ic_without_it ?? 0) < 0).length;
      loyoTable = `<p class="muted" style="margin-top:8px">Leave-one-year-out: el rank-IC es negativo al excluir ${negatives} de ${loyo.length} años (mide si el resultado depende de un solo año).</p>` +
        table(loyo, { columns: ["excluded_year", "rank_ic_without_it", "delta_vs_full", "n_cohorts"],
          labels: { excluded_year: "Año excluido", rank_ic_without_it: "rank-IC sin él", delta_vs_full: "Δ vs total", n_cohorts: "Cohortes" },
          decimals: 4, sortable: true });
    }
    return `<details style="margin-top:16px"><summary>Robustez multi-era (validación del aprendizaje)</summary>
      ${cards}${loyoTable}</details>`;
  }

  async function open(runId, container) {
    global.TFMCharts.clear();
    container.innerHTML = `<p class="muted">Cargando run…</p>`;
    let manifest;
    try {
      manifest = await api("/api/run/" + encodeURIComponent(runId));
    } catch (e) {
      container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
      return;
    }
    current = { runId, manifest };
    const s = manifest.summary || {};
    const intent = manifest.intent || {};
    container.innerHTML = `
      <h3>${escapeHtml(intent.label || runId)}</h3>
      <p class="muted mono">${escapeHtml(runId)} · <span class="tag">${escapeHtml(manifest.run_kind || "")}</span></p>
      <div class="cards">
        ${metricCard(s.mean_rank_ic, "rank-IC medio (OOS)", { decimals: 2, hintKey: "rank_ic" })}
        ${metricCard(s.cagr_difference, "CAGR vs benchmark", { format: "pct", decimals: 2, signed: true, hintKey: "cagr_difference" })}
        ${metricCard(s.max_drawdown, "Max drawdown", { format: "pct", decimals: 2, signed: true, hintKey: "max_drawdown" })}
        ${metricCard(s.beat_rate, "% años que baten SPY", { format: "pct", decimals: 1, hintKey: "beat_rate" })}
      </div>
      ${intent.description ? `<p class="muted" style="margin-top:12px">${escapeHtml(intent.description)}</p>` : ""}
      ${robustnessBlock(manifest.robustness)}
      <div class="tabs" id="run-tabs">${TABS.map(([id, label], i) => `<button data-tab="${id}" class="${i === 0 ? "active" : ""}" onclick="TFM.views.run.tab('${id}')">${label}</button>`).join("")}</div>
      <div id="run-tab-body"></div>`;
    tab("resumen");
  }

  async function tab(name) {
    document.querySelectorAll("#run-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    const body = el("run-tab-body");
    global.TFMCharts.clear();
    const runId = current.runId;
    const views = global.TFM.views;
    if (name === "resumen") return views.performance.renderSummary(body, runId, current.manifest);
    if (name === "rendimiento") return views.performance.render(body, runId);
    if (name === "aprendizaje") return views.learning.render(body, runId);
    if (name === "cartera") return views.portfolio.render(body, runId);
    if (name === "trades") return views.portfolio.renderTrades(body, runId);
    if (name === "stocks") return views.stocks.render(body, runId);
  }

  global.TFM.views.run = { open, tab, settingsCards };
})(window);
