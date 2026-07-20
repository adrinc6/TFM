/* Adaptadores de dominio sobre Chart.js. Cada función recibe los JSON de la API y devuelve
   un gráfico con los colores de la paleta (leídos de las variables CSS: el color vive en un
   solo sitio). Todos crean un <canvas> dentro de una .chart-box con título. */
(function (global) {
  "use strict";
  const P = global.TFM.palette;

  // Defaults oscuros globales de Chart.js
  function applyTheme() {
    if (!global.Chart) return;
    Chart.defaults.color = P.muted;
    Chart.defaults.borderColor = P.line;
    Chart.defaults.font.family = getComputedStyle(document.documentElement).getPropertyValue("--font").trim();
    Chart.defaults.plugins.legend.labels.color = P.muted;
    Chart.defaults.plugins.tooltip.backgroundColor = "#000000cc";
    Chart.defaults.plugins.tooltip.borderColor = P.line;
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.maintainAspectRatio = false;
  }

  const registry = [];
  function box(title, height = 260) {
    const wrap = document.createElement("div");
    wrap.className = "chart-box";
    if (title) {
      const h = document.createElement("div");
      h.className = "chart-title";
      h.textContent = title;
      wrap.appendChild(h);
    }
    const holder = document.createElement("div");
    holder.style.height = height + "px";
    const canvas = document.createElement("canvas");
    holder.appendChild(canvas);
    wrap.appendChild(holder);
    return { wrap, canvas };
  }

  function make(container, title, config, height) {
    applyTheme();
    const { wrap, canvas } = box(title, height);
    container.appendChild(wrap);
    const chart = new Chart(canvas.getContext("2d"), config);
    registry.push(chart);
    return chart;
  }

  // Destruye los gráficos previos (evita fugas al recargar una vista)
  function clear() {
    while (registry.length) {
      try { registry.pop().destroy(); } catch (e) { /* noop */ }
    }
  }

  // Sustituye un único gráfico sin borrar los de otras secciones de la vista.
  function clearContainer(container) {
    for (let i = registry.length - 1; i >= 0; i -= 1) {
      const chart = registry[i];
      if (!container.contains(chart.canvas)) continue;
      registry.splice(i, 1);
      try { chart.destroy(); } catch (e) { /* noop */ }
    }
    container.replaceChildren();
  }

  // Redibuja todos los gráficos vivos. Necesario tras desplegar un <details> que los contenía
  // cerrado: dentro de un details cerrado el canvas mide 0 y Chart.js lo renderiza en blanco.
  function resizeAll() {
    for (const chart of registry) {
      try { chart.resize(); } catch (e) { /* noop */ }
    }
  }

  // Color de un agente individual: grises de la paleta de series (el meta se pinta aparte en oro).
  const AGENT_GRAYS = ["--series-1", "--series-2", "--series-3"];
  function agentColor(i) {
    return getComputedStyle(document.documentElement).getPropertyValue(AGENT_GRAYS[i % AGENT_GRAYS.length]).trim();
  }

  const gridScale = (opts = {}) => ({
    grid: { color: P.line, drawBorder: false },
    ticks: { color: P.muted, ...(opts.ticks || {}) },
    ...opts,
  });

  // --- Curva de crecimiento: cartera vs benchmark, base 100 ---
  function equityChart(container, equity) {
    const rows = equity || [];
    if (!rows.length) return;
    const labels = rows.map((r) => String(r.snapshot_date).slice(0, 10));
    const series = P.series();
    make(container, "Crecimiento (base 100): cartera vs benchmark", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Cartera", data: rows.map((r) => r.portfolio_value), borderColor: series[0], backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0.15 },
          { label: "Benchmark (SPY)", data: rows.map((r) => r.benchmark_value), borderColor: series[1], backgroundColor: "transparent", borderWidth: 2, borderDash: [6, 4], pointRadius: 0, tension: 0.15 },
        ],
      },
      options: { scales: { x: gridScale({ ticks: { maxTicksLimit: 8 } }), y: gridScale() }, plugins: { legend: { display: true } } },
    });
  }

  // --- Barras de alfa anual (verde positivo / rojo negativo) ---
  function alphaBars(container, annual) {
    const rows = annual || [];
    if (!rows.length) return;
    make(container, "Alfa anual (cartera − benchmark)", {
      type: "bar",
      data: {
        labels: rows.map((r) => r.year),
        datasets: [{
          label: "Alfa",
          data: rows.map((r) => r.alpha),
          backgroundColor: rows.map((r) => (r.alpha >= 0 ? P.pos : P.neg)),
        }],
      },
      options: { scales: { x: gridScale(), y: gridScale() }, plugins: { legend: { display: false } } },
    });
  }

  // --- Rank-IC en el tiempo (una serie por agente) ---
  function rankIcOverTime(container, series) {
    const rows = series || [];
    if (!rows.length) return;
    const agents = [...new Set(rows.map((r) => r.agent))];
    const dates = [...new Set(rows.map((r) => String(r.prediction_date).slice(0, 10)))].sort();
    const datasets = agents.map((agent, i) => {
      const byDate = new Map(rows.filter((r) => r.agent === agent).map((r) => [String(r.prediction_date).slice(0, 10), r.rank_ic]));
      const meta = agent === "meta_final";
      return {
        label: agent,
        data: dates.map((d) => (byDate.has(d) ? byDate.get(d) : null)),
        // El meta destaca en oro y trazo grueso; los agentes individuales en gris.
        borderColor: meta ? P.accent : agentColor(i),
        backgroundColor: "transparent",
        borderWidth: meta ? 2.8 : 1.5,
        pointRadius: 0,
        tension: 0.1,
        spanGaps: true,
      };
    });
    make(container, "Rank-IC por cohorte (fuera de muestra)", {
      type: "line",
      data: { labels: dates, datasets },
      options: {
        scales: { x: gridScale({ ticks: { maxTicksLimit: 8 } }), y: gridScale() },
        plugins: { legend: { display: true } },
      },
    });
  }

  // --- Evolución de pesos del meta-agente ---
  function metaWeights(container, weights) {
    const rows = weights || [];
    if (!rows.length) return;
    const agents = [...new Set(rows.map((r) => r.agent))];
    const dates = [...new Set(rows.map((r) => String(r.snapshot_date).slice(0, 10)))].sort();
    const colors = P.categorical();
    const datasets = agents.map((agent, i) => {
      const byDate = new Map(rows.filter((r) => r.agent === agent).map((r) => [String(r.snapshot_date).slice(0, 10), r.weight]));
      return {
        label: agent,
        data: dates.map((d) => (byDate.has(d) ? byDate.get(d) : null)),
        // Cada agente es una identidad distinta: paleta categórica en orden fijo.
        borderColor: colors[i % colors.length],
        backgroundColor: "transparent",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        spanGaps: true,
      };
    });
    make(container, "Pesos del meta-agente en el tiempo", {
      type: "line",
      data: { labels: dates, datasets },
      options: { scales: { x: gridScale({ ticks: { maxTicksLimit: 8 } }), y: gridScale() }, plugins: { legend: { display: true } } },
    });
  }

  // --- Composición de cartera (dona) ---
  function portfolioComposition(container, positions) {
    const rows = (positions || []).filter((r) => r.weight);
    if (!rows.length) return;
    // Usa la última fecha disponible
    const lastDate = rows.map((r) => String(r.snapshot_date)).sort().slice(-1)[0];
    const latest = rows.filter((r) => String(r.snapshot_date) === lastDate)
      .sort((a, b) => (b.weight || 0) - (a.weight || 0));
    if (!latest.length) return;
    // Las mayores posiciones reciben un color de la paleta categórica (identidad); el resto,
    // grises decrecientes para no inventar matices más allá de los validados. Anillo de superficie
    // entre porciones (separador de 2px) para que se distingan aunque compartan gama.
    const cat = P.categorical();
    const grays = ["#8a8a92", "#6f6f77", "#5c5c64", "#4a4a51", "#3a3a40", "#2c2c32"];
    const sliceColor = (i) => (i < cat.length ? cat[i] : grays[(i - cat.length) % grays.length]);
    make(container, `Composición de cartera · ${String(lastDate).slice(0, 10)}`, {
      type: "doughnut",
      data: {
        labels: latest.map((r) => r.ticker),
        datasets: [{ data: latest.map((r) => r.weight * 100), backgroundColor: latest.map((_, i) => sliceColor(i)), borderColor: P.surface, borderWidth: 2 }],
      },
      options: { plugins: { legend: { position: "right" }, tooltip: { callbacks: { label: (c) => `${c.label}: ${c.parsed.toFixed(1)} %` } } } },
    }, 300);
  }

  // --- Historia de un ratio de stock vs media S&P 500 e índice de precio ---
  function stockHistory(container, history) {
    const points = (history && history.points) || [];
    if (!points.length) return;
    const labels = points.map((p) => p.snapshot_date);
    const series = P.series();
    const datasets = [
      { label: history.metric_label, data: points.map((p) => p.value), borderColor: series[0], backgroundColor: "transparent", borderWidth: 2.5, pointRadius: 0, yAxisID: "y", tension: 0.1, spanGaps: true },
      { label: "Media S&P 500", data: points.map((p) => p.sp500_mean), borderColor: series[3], backgroundColor: "transparent", borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, yAxisID: "y", tension: 0.1, spanGaps: true },
      { label: "Precio (índice 100)", data: points.map((p) => p.price_index), borderColor: series[2], backgroundColor: "transparent", borderWidth: 1.5, pointRadius: 0, yAxisID: "y1", tension: 0.1, spanGaps: true },
    ];
    if (history.companion) {
      datasets.push({
        label: history.companion_label + (history.companion_indexed ? " (índice 100)" : ""),
        data: points.map((p) => (history.companion_indexed ? p.companion_display : p.companion)),
        borderColor: P.pos, backgroundColor: "transparent", borderWidth: 1.5, pointRadius: 0,
        yAxisID: history.companion_indexed ? "y1" : "y2", tension: 0.1, spanGaps: true,
      });
    }
    const scales = {
      x: gridScale({ ticks: { maxTicksLimit: 8 } }),
      y: gridScale({ position: "left", title: { display: true, text: history.metric_label, color: P.muted } }),
      y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { color: P.muted }, title: { display: true, text: "índice 100", color: P.muted } },
    };
    if (history.companion && !history.companion_indexed) {
      scales.y2 = { position: "right", display: false, grid: { drawOnChartArea: false }, ticks: { color: P.muted } };
    }
    make(container, `Historia de ${history.metric_label} · ${history.ticker}`, {
      type: "line",
      data: { labels, datasets },
      options: { interaction: { mode: "index", intersect: false }, scales, plugins: { legend: { display: true } } },
    }, 320);
  }

  // --- Precio point-in-time de un ticker con marcadores de compra/venta ---
  function tickerPrice(container, prices, orders) {
    const rows = prices || [];
    if (!rows.length) return;
    const labels = rows.map((r) => String(r.snapshot_date).slice(0, 10));
    const byDate = new Map(rows.map((r, i) => [String(r.snapshot_date).slice(0, 10), i]));
    const buys = [];
    const sells = [];
    (orders || []).forEach((o) => {
      const key = String(o.snapshot_date).slice(0, 10);
      if (!byDate.has(key)) return;
      const point = { x: key, y: Number(rows[byDate.get(key)].price) };
      (o.side === "sell" ? sells : buys).push(point);
    });
    make(container, "Precio point-in-time y operaciones", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Precio", data: rows.map((r) => Number(r.price)), borderColor: P.series()[0], backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0.1 },
          { label: "Compras", data: buys, showLine: false, pointBackgroundColor: P.pos, pointBorderColor: P.pos, pointRadius: 5, type: "scatter" },
          { label: "Ventas", data: sells, showLine: false, pointBackgroundColor: P.neg, pointBorderColor: P.neg, pointRadius: 5, type: "scatter" },
        ],
      },
      options: { scales: { x: gridScale({ ticks: { maxTicksLimit: 10 } }), y: gridScale() }, plugins: { legend: { display: true } } },
    });
  }

  // --- Histórico de puntuaciones de una acción por agente (calidad/momentum/valor/meta) ---
  // `rows`: filas de agent_scores del ticker (una por snapshot) con columnas quality/momentum/
  // value/meta_score. Cada agente es una serie con identidad → paleta categórica.
  function stockScoreHistory(container, rows) {
    const list = (rows || []).slice().sort((a, b) => String(a.snapshot_date).localeCompare(String(b.snapshot_date)));
    if (!list.length) return;
    const labels = list.map((r) => String(r.snapshot_date).slice(0, 10));
    const cat = P.categorical();
    const fields = [
      ["quality", "Calidad"], ["momentum", "Momentum"], ["value", "Valor"], ["meta_score", "Meta"],
    ].filter(([key]) => list.some((r) => typeof r[key] === "number" && Number.isFinite(r[key])));
    const datasets = fields.map(([key, label], i) => ({
      label,
      data: list.map((r) => (typeof r[key] === "number" ? r[key] : null)),
      borderColor: cat[i % cat.length],
      backgroundColor: "transparent",
      borderWidth: key === "meta_score" ? 2.6 : 1.6,
      pointRadius: 0,
      tension: 0.1,
      spanGaps: true,
    }));
    make(container, "Histórico de puntuaciones por agente", {
      type: "line",
      data: { labels, datasets },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { x: gridScale({ ticks: { maxTicksLimit: 8 } }), y: gridScale() },
        plugins: { legend: { display: true } },
      },
    });
  }

  global.TFMCharts = {
    clear, clearContainer, resizeAll, equityChart, alphaBars, rankIcOverTime, metaWeights,
    portfolioComposition, stockHistory, tickerPrice, stockScoreHistory,
  };
})(window);
