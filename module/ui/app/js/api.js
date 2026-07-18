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
  };

  // --- Helpers de DOM ---
  const el = (id) => document.getElementById(id);
  function table(rows, opts = {}) {
    const list = Array.isArray(rows) ? rows : [];
    if (!list.length) return `<p class="muted">Sin datos.</p>`;
    const keys = opts.columns || Object.keys(list[0]);
    const limit = opts.limit || 200;
    const head = keys.map((k) => `<th>${escapeHtml(opts.labels ? opts.labels[k] || k : k)}</th>`).join("");
    const body = list
      .slice(0, limit)
      .map(
        (row) =>
          "<tr>" +
          keys
            .map((k) => {
              const v = row[k];
              const cls = typeof v === "number" ? signClass(v) : "";
              const shown = typeof v === "number" ? fmt(v, opts.decimals ?? 4) : escapeHtml(v ?? "—");
              return `<td class="${cls}">${shown}</td>`;
            })
            .join("") +
          "</tr>"
      )
      .join("");
    return `<div class="table-wrap scroll"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  global.TFM = { api, qs, fmt, pct, signClass, escapeHtml, palette, el, table };
})(window);
