/* Wrapper de la API JSON del backend y utilidades compartidas de formato.
   El contrato de rutas es el de module/ui/dashboard.py (do_GET / do_POST). */
(function (global) {
  "use strict";

  async function api(path, body) {
    const options = {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    };
    const response = await fetch(path, options);
    let payload;
    try {
      payload = await response.json();
    } catch (err) {
      throw new Error("Respuesta no válida del servidor.");
    }
    if (!response.ok) throw new Error(payload && payload.error ? payload.error : "Error del servidor");
    return payload;
  }

  const qs = (params) =>
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== "")
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join("&");

  // --- Formato ---
  function fmt(value, decimals = 4) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(decimals) : "—";
  }
  function pct(value, decimals = 1) {
    return typeof value === "number" && Number.isFinite(value)
      ? (value * 100).toFixed(decimals) + " %"
      : "—";
  }
  function signClass(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "";
    return value > 0 ? "positive" : value < 0 ? "negative" : "";
  }
  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  // --- Formateadores de dominio (para presentación profesional) ---
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  function year(v) { return isNum(v) ? String(Math.round(v)) : escapeHtml(v ?? "—"); }
  function money(v, decimals = 2) { return isNum(v) ? v.toFixed(decimals) : "—"; }
  function shortDate(v) {
    // Recorta timestamps ISO a YYYY-MM-DD (las fechas llegan como "2020-03-15T00:00:00").
    const s = String(v ?? "");
    return s ? s.slice(0, 10) : "—";
  }
  // Heurística: columnas que no deben mostrarse con 4 decimales.
  const YEAR_COLS = new Set(["year", "anio", "año"]);
  const DATE_COLS = /(date|fecha)/i;
  // Columnas de tipo porcentaje (se muestran ×100 con " %"). Cubre retornos de cartera
  // y benchmark, alfa, exceso, CAGR, drawdown, tasa de aciertos, pesos y latentes.
  // Quedan FUERA a propósito: rank-IC (no es %), information_ratio / _ir (es un ratio),
  // y los ratios fundamentales crudos (pe, pb, roe…).
  const PCT_COLS = /(return|alpha|excess|benchmark|portfolio|cagr|drawdown|beat|weight|unrealized|_rate$|^rate$|percentil|fraction|_pct|pct_|turnover)/i;
  // No convertir a %: ratios (information_ratio/_ir) y columnas de VALOR absoluto o índice base-100
  // (portfolio_value, benchmark_value, price…). Estas últimas matchean PCT_COLS por "portfolio"/
  // "benchmark" pero NO son porcentajes.
  const RATIO_COLS = /(information_ratio|_ratio$|^ir$|_ir$)/i;
  // current_percentile (position_lifecycle) ya viene en escala 0-100, no 0-1: convertirlo a %
  // (×100) lo dejaría en ~9800. Se trata como valor absoluto (número plano), no porcentaje.
  const ABSOLUTE_COLS = /(_value$|^value$|price|shares|quantity|^current_percentile$)/i;
  // Percentiles ya en escala 0-100 (no fracción 0-1): número plano con 1 decimal, sin ×100.
  const SCALED_PERCENTILE_COLS = /^current_percentile$/i;
  function isPctCol(key) { return PCT_COLS.test(key) && !RATIO_COLS.test(key) && !ABSOLUTE_COLS.test(key); }
  function autoFormat(key, v) {
    if (v == null) return "—";
    if (YEAR_COLS.has(key)) return year(v);
    if (DATE_COLS.test(key)) return shortDate(v);
    if (typeof v === "boolean") return v ? "sí" : "no";
    if (!isNum(v)) return escapeHtml(v);
    if (isPctCol(key)) return pct(v, 2);
    if (SCALED_PERCENTILE_COLS.test(key)) return fmt(v, 1);
    // Enteros se muestran sin decimales; el resto con 4.
    return Number.isInteger(v) ? String(v) : fmt(v, 4);
  }

  // --- Lectura de tokens de color de la paleta (para los gráficos) ---
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  const palette = {
    get ink() { return cssVar("--ink"); },
    get muted() { return cssVar("--muted"); },
    get line() { return cssVar("--line"); },
    get surface() { return cssVar("--surface"); },
    get pos() { return cssVar("--series-pos"); },
    get neg() { return cssVar("--series-neg"); },
    series() {
      return [cssVar("--series-1"), cssVar("--series-2"), cssVar("--series-3"), cssVar("--series-4")];
    },
    // Paleta categórica (identidad): orden fijo, validada CVD-safe. Ver tokens.css.
    categorical() {
      return [cssVar("--cat-1"), cssVar("--cat-2"), cssVar("--cat-3"), cssVar("--cat-4")];
    },
    get accent() { return cssVar("--series-4"); },
  };

  // --- Helpers de DOM ---
  const el = (id) => document.getElementById(id);

  // Primera columna de fecha de una fila (para ordenar por defecto lo más reciente arriba).
  function dateKeyOf(row) {
    return Object.keys(row || {}).find((k) => DATE_COLS.test(k)) || null;
  }
  // Ordena descendente por una columna de fecha (copia; no muta el original).
  function sortByDateDesc(rows, key) {
    const k = key || (rows.length ? dateKeyOf(rows[0]) : null);
    if (!k) return rows;
    return rows.slice().sort((a, b) => String(b[k] ?? "").localeCompare(String(a[k] ?? "")));
  }

  function table(rows, opts = {}) {
    let list = Array.isArray(rows) ? rows : [];
    if (!list.length) return `<p class="muted">${escapeHtml(opts.empty || "Sin datos.")}</p>`;
    // Por defecto, las tablas con columna de fecha muestran lo más reciente arriba.
    if (opts.sortDateDesc !== false && dateKeyOf(list[0])) list = sortByDateDesc(list);
    const keys = opts.columns || Object.keys(list[0]);
    const limit = opts.limit || 200;
    // Cabeceras clicables para ordenar (salvo opts.sortable === false). El tooltip explica la
    // métrica cuando la hay. El comportamiento lo añade el módulo de ordenación por delegación.
    const sortable = opts.sortable !== false;
    const head = keys.map((k) => {
      const label = opts.labels ? opts.labels[k] || k : k;
      const hint = metricHint(k);
      const cls = sortable ? ' class="sortable"' : "";
      const title = hint ? ` title="${escapeHtml(hint)}"` : "";
      return `<th${cls}${title} data-key="${escapeHtml(k)}">${escapeHtml(label)}</th>`;
    }).join("");
    const body = list
      .slice(0, limit)
      .map(
        (row) =>
          "<tr>" +
          keys
            .map((k) => {
              const v = row[k];
              const cls = typeof v === "number" ? signClass(v) : "";
              // `decimals` fija los decimales de los números "normales", pero NUNCA anula el
              // formato por-nombre de las columnas especiales: año entero, fecha corta, booleano
              // sí/no y porcentajes ×100. Así evitamos "2014.0000" o "1.0000" en la tabla anual.
              let shown;
              if (typeof v !== "number") shown = autoFormat(k, v);
              else if (YEAR_COLS.has(k)) shown = year(v);
              else if (DATE_COLS.test(k)) shown = shortDate(v);
              // Los porcentajes se muestran con 2 decimales aunque la tabla pida más para los
              // valores crudos (evita "17.2200 %"); `pctDecimals` permite forzar otro valor.
              else if (isPctCol(k)) shown = pct(v, opts.pctDecimals ?? 2);
              else if (SCALED_PERCENTILE_COLS.test(k)) shown = fmt(v, 1);
              // Enteros (meses tenidos, conteos…) nunca llevan decimales, aunque `decimals` los pida.
              else if (Number.isInteger(v)) shown = String(v);
              else shown = fmt(v, opts.decimals ?? 4);
              return `<td class="${cls}">${shown}</td>`;
            })
            .join("") +
          "</tr>"
      )
      .join("");
    return `<div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  // Explicaciones cortas de métricas para tooltips. Clave = concepto (no la columna exacta).
  const METRIC_HINTS = {
    rank_ic: "Correlación de rangos entre la predicción y el retorno futuro. Mide si el modelo ordena bien las acciones (0 = sin capacidad, >0 = acierta el orden).",
    information_ratio: "Rentabilidad extra sobre el benchmark dividida por su volatilidad. Cuánto alfa por unidad de riesgo asumido.",
    cagr: "Tasa de crecimiento anual compuesta. Rentabilidad media anualizada de la cartera.",
    cagr_difference: "Diferencia de CAGR entre la cartera y el benchmark (SPY). Positivo = la cartera crece más rápido.",
    beat_rate: "Porcentaje de años en los que la cartera supera al benchmark. Mide la consistencia, no la magnitud.",
    max_drawdown: "Mayor caída desde un máximo hasta el mínimo posterior. Peor pérdida encadenada.",
    percentile: "Posición relativa dentro del universo en esa fecha (p90 = mejor que el 90 % de las acciones).",
    alpha: "Rentabilidad de la cartera menos la del benchmark en ese periodo.",
    unrealized: "Ganancia o pérdida latente de una posición aún abierta (sobre el precio de entrada).",
    realized: "Ganancia o pérdida realizada al vender (precio de venta frente al de entrada).",
  };
  function metricHint(key) {
    if (!key) return "";
    const k = String(key).toLowerCase();
    const found = Object.keys(METRIC_HINTS).find((h) => k.includes(h));
    return found ? METRIC_HINTS[found] : "";
  }

  // Rejilla de tarjetitas "valor grande + significado pequeño" (config efectiva, resúmenes).
  // Cada item: { value, label, cls?, hint? } — `cls` colorea el valor; `hint` añade tooltip.
  function metricGrid(items) {
    const cards = (items || [])
      .map((it) => {
        const value = it.value == null || it.value === "" ? "—" : escapeHtml(it.value);
        const hint = it.hint || metricHint(it.hintKey);
        const labelAttr = hint ? ` title="${escapeHtml(hint)}"` : "";
        const marker = hint ? ' <span class="hint-dot">?</span>' : "";
        return `<div class="card"><div class="metric${it.cls ? " " + it.cls : ""}">${value}</div>` +
          `<div class="metric-label"${labelAttr}>${escapeHtml(it.label)}${marker}</div></div>`;
      })
      .join("");
    return `<div class="cards">${cards}</div>`;
  }

  // Ordenación de tablas al clicar cabecera: reordena las filas del DOM de esa <table.data>.
  // Por delegación global (un único listener), sin recargar la vista ni estado por tabla.
  function sortTableByColumn(table, colIndex, dir) {
    const tbody = table.tBodies[0];
    if (!tbody) return;
    const rows = Array.from(tbody.rows).filter((r) => !r.classList.contains("empty"));
    const parseCell = (cell) => {
      const raw = (cell.textContent || "").replace(/[%\s]/g, "").replace(",", ".");
      const num = parseFloat(raw);
      return Number.isFinite(num) && /[\d]/.test(raw) ? num : (cell.textContent || "").toLowerCase();
    };
    rows.sort((a, b) => {
      const va = parseCell(a.cells[colIndex]);
      const vb = parseCell(b.cells[colIndex]);
      if (va < vb) return dir === "asc" ? -1 : 1;
      if (va > vb) return dir === "asc" ? 1 : -1;
      return 0;
    });
    rows.forEach((r) => tbody.appendChild(r));
  }
  document.addEventListener("click", (event) => {
    const th = event.target.closest ? event.target.closest("th.sortable") : null;
    if (!th) return;
    const table = th.closest("table.data");
    if (!table) return;
    const index = Array.from(th.parentNode.children).indexOf(th);
    const dir = th.dataset.sortDir === "asc" ? "desc" : "asc";
    table.querySelectorAll("th").forEach((h) => { delete h.dataset.sortDir; h.classList.remove("sort-asc", "sort-desc"); });
    th.dataset.sortDir = dir;
    th.classList.add(dir === "asc" ? "sort-asc" : "sort-desc");
    sortTableByColumn(table, index, dir);
  });

  // Buscador de tabla: filtra las filas de <table.data> por texto. La vista coloca un
  // <input data-filter-table="ID"> y una tabla con ese id; este listener hace el resto.
  document.addEventListener("input", (event) => {
    const input = event.target;
    if (!input.dataset || !input.dataset.filterTable) return;
    const table = document.getElementById(input.dataset.filterTable);
    if (!table || !table.tBodies[0]) return;
    const needle = input.value.trim().toLowerCase();
    for (const row of table.tBodies[0].rows) {
      row.style.display = !needle || row.textContent.toLowerCase().includes(needle) ? "" : "none";
    }
  });

  // Al desplegar un <details> que contenga gráficos, redibujarlos (dentro de un details cerrado
  // el canvas mide 0 y saldría en blanco). `toggle` no burbujea: se escucha en fase de captura.
  document.addEventListener("toggle", (event) => {
    const d = event.target;
    if (d && d.tagName === "DETAILS" && d.open && d.querySelector("canvas") && global.TFMCharts) {
      global.TFMCharts.resizeAll();
    }
  }, true);

  // `views` se inicializa aquí (api.js carga antes que las vistas) para que cada vista pueda
  // registrarse con `TFM.views.<nombre> = {...}` sin depender del orden de <script>.
  global.TFM = { api, qs, fmt, pct, signClass, escapeHtml, palette, el, table, metricGrid, views: {},
                 year, money, shortDate, isNum, sortByDateDesc, metricHint };
})(window);
