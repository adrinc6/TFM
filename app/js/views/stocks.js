/* Explorador de stocks: buscador de ticker, selector de ratio (calidad / crecimiento /
   valoración / momentum), rango OOS/completo/manual, tarjetas de ratios con percentiles,
   gráfico de historia vs media S&P 500 e índice de precio, y explicabilidad de agentes.
   Consume /api/stocks, /api/stock/summary, /api/stock/history, /api/stock/agents. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, pct, table } = global.TFM;

  const ctx = {}; // { runId, oosStart, oosEnd }

  function metricOptions(groups) {
    return Object.entries(groups || {})
      .map(([group, items]) =>
        `<optgroup label="${escapeHtml(group)}">${items.map((it) => `<option value="${escapeHtml(it[0])}">${escapeHtml(it[1])}</option>`).join("")}</optgroup>`
      )
      .join("");
  }

  async function render(container, runId) {
    ctx.runId = runId;
    container.innerHTML = `<p class="muted">Cargando explorador…</p>`;
    let meta;
    try { meta = await api("/api/stocks?run_id=" + encodeURIComponent(runId)); }
    catch (e) { container.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    if (!meta.compatible) {
      container.innerHTML = `<div class="notice">${escapeHtml(meta.message || "Este run no conserva el panel de stocks.")}</div>`;
      return;
    }
    ctx.oosStart = meta.oos_start || "";
    ctx.oosEnd = meta.oos_end || "";
    container.innerHTML = `
      <section class="parameter-group"><h4>Explorador de stocks</h4>
        <p class="muted">Datos point-in-time inmutables de este run. El rango inicial es su periodo OOS.</p>
        <div class="formgrid">
          <label class="field">Ticker<input id="stk-ticker" list="stk-list" placeholder="AAPL" oninput="TFM.views.stocks.search()"></label>
          <datalist id="stk-list"></datalist>
          <label class="field">Ratio<select id="stk-metric">${metricOptions(meta.metrics)}</select></label>
          <label class="field">Rango<select id="stk-range" onchange="TFM.views.stocks.setRange()">
            <option value="oos">Periodo OOS del run</option>
            <option value="full">Histórico completo del run</option>
            <option value="custom">Rango manual</option>
          </select></label>
          <label class="field">Inicio<input id="stk-start" type="date" value="${escapeHtml(ctx.oosStart)}"></label>
          <label class="field">Fin<input id="stk-end" type="date" value="${escapeHtml(ctx.oosEnd)}"></label>
        </div>
        <div class="actions"><button class="button primary" onclick="TFM.views.stocks.load()">Analizar acción</button></div>
      </section>
      <div id="stk-output" class="muted">Busca y selecciona una acción.</div>`;
    search();
  }

  async function search() {
    const q = el("stk-ticker").value.trim();
    try {
      const data = await api("/api/stocks?run_id=" + encodeURIComponent(ctx.runId) + "&query=" + encodeURIComponent(q));
      el("stk-list").innerHTML = (data.tickers || []).map((t) => `<option value="${escapeHtml(t)}"></option>`).join("");
    } catch (e) { /* silencioso */ }
  }

  function setRange() {
    const mode = el("stk-range").value;
    if (mode === "oos") { el("stk-start").value = ctx.oosStart || ""; el("stk-end").value = ctx.oosEnd || ""; }
    if (mode === "full") { el("stk-start").value = ""; el("stk-end").value = ""; }
    load();
  }

  function ratiosTable(summary) {
    const rows = (summary.ratios || [])
      .map((r) => `<tr><td>${escapeHtml(r.label)}</td><td>${fmt(r.value, 3)}</td><td>${r.percentile == null ? "—" : pct(r.percentile, 1)}</td><td>${r.observations}</td></tr>`)
      .join("");
    return `<h4>Ratios disponibles <small class="muted">percentil en el universo del snapshot</small></h4>
      <div class="table-wrap scroll"><table class="data"><thead><tr><th>Ratio</th><th>Valor</th><th>Percentil</th><th>N</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function summaryCards(s) {
    const score = s.scores || {};
    const pos = s.position || {};
    return `<div class="cards">
      <article class="card"><h3>${escapeHtml(s.ticker)}</h3>
        <p>Snapshot: ${escapeHtml(s.snapshot_date)}<br>Precio PIT: ${fmt(s.price, 3)} (${escapeHtml(s.price_as_of_date || "—")})</p>
        <p class="muted">Fundamental publicado: ${escapeHtml(s.fundamental_filed_date || "—")} · antigüedad ${fmt(s.fundamental_age_days, 0)} días</p></article>
      <article class="card"><h3>Scores</h3>
        <p>Calidad ${fmt(score.quality, 3)} · p${fmt((score.quality_rank || 0) * 100, 1)}<br>
        Momentum ${fmt(score.momentum, 3)} · p${fmt((score.momentum_rank || 0) * 100, 1)}<br>
        Valor ${fmt(score.value, 3)} · p${fmt((score.value_rank || 0) * 100, 1)}<br>
        Meta ${fmt(score.meta_score, 3)} · p${fmt((score.meta_rank || 0) * 100, 1)}</p></article>
      <article class="card"><h3>Cartera</h3>
        <p>Peso ${fmt((pos.weight || 0) * 100, 2)} %<br>Entrada ${fmt(pos.entry_price, 3)} · valoración ${fmt(pos.valuation_price, 3)}<br>Rentabilidad latente ${fmt(pos.unrealized_return_pct, 2)} %</p></article>
    </div>`;
  }

  function agentExplain(rows) {
    const pos = rows.filter((x) => x.local_contribution > 0).slice(0, 2).map((x) => x.feature.replace("factor_", "")).join(", ");
    const neg = rows.filter((x) => x.local_contribution < 0).slice(0, 2).map((x) => x.feature.replace("factor_", "")).join(", ");
    return `El agente favorece por ${pos || "ninguna contribución positiva destacada"}${neg ? `; penaliza por ${neg}` : ""}.`;
  }

  function agentsBlock(a) {
    const groups = {};
    (a.contributions || []).forEach((c) => (groups[c.agent] ??= []).push(c));
    const cards = Object.entries(groups).map(([agent, rows]) => {
      const t = table(rows.map((r) => ({
        variable: r.feature.replace("factor_", ""),
        valor_factor: r.factor_value,
        contribucion: r.local_contribution,
        direccion: r.direction,
      })), { decimals: 4 });
      return `<article class="card"><h3>${escapeHtml(agent)}</h3><p>${escapeHtml(agentExplain(rows))}</p>${t}</article>`;
    }).join("") || `<p class="muted">Sin contribuciones locales para este run/fecha.</p>`;
    return `<h4>Por qué puntúan los agentes</h4><div class="cards">${cards}</div>
      <h4>Pesos del meta-agente</h4>${table(a.weights || [], { decimals: 4 })}
      <details><summary>Importancia global del modelo de esta fecha</summary>${table(a.global_importance || [], { decimals: 4 })}</details>`;
  }

  async function load() {
    const ticker = el("stk-ticker").value.trim().toUpperCase();
    if (!ticker) return;
    const metric = el("stk-metric").value;
    const start = el("stk-start").value;
    const end = el("stk-end").value;
    const out = el("stk-output");
    out.innerHTML = `<p class="muted">Analizando ${escapeHtml(ticker)}…</p>`;
    try {
      const [summary, history, agents] = await Promise.all([
        api(`/api/stock/summary?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ticker)}`),
        api(`/api/stock/history?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ticker)}&metric=${encodeURIComponent(metric)}&start=${start}&end=${end}`),
        api(`/api/stock/agents?run_id=${encodeURIComponent(ctx.runId)}&ticker=${encodeURIComponent(ticker)}`),
      ]);
      if (summary.found === false) { out.innerHTML = `<div class="notice">Sin datos para ${escapeHtml(ticker)} en este run.</div>`; return; }
      out.innerHTML = summaryCards(summary) + ratiosTable(summary) +
        `<h4>Historia de ${escapeHtml(history.metric_label || metric)}</h4><div id="stk-chart"></div>` +
        `<div id="stk-agents"></div>`;
      global.TFMCharts.clear();
      global.TFMCharts.stockHistory(el("stk-chart"), history);
      el("stk-agents").innerHTML = agentsBlock(agents);
    } catch (e) {
      out.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`;
    }
  }

  global.TFM.views.stocks = { render, search, setRange, load };
})(window);
