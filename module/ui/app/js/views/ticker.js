/* Ficha por ticker: precio point-in-time con marcadores de compra/venta + tabla de órdenes y
   evolución de agentes. Consume /api/ticker. */
(function (global) {
  "use strict";
  const { api, el, escapeHtml, fmt, table } = global.TFM;

  function render(container, runId) {
    container.innerHTML = `
      <div class="formgrid">
        <label class="field">Buscar ticker<input id="tk-input" placeholder="AAPL" onkeydown="if(event.key==='Enter')TFM.views.ticker.load('${escapeHtml(runId)}')"></label>
      </div>
      <div class="actions"><button class="button primary" onclick="TFM.views.ticker.load('${escapeHtml(runId)}')">Ver ticker</button></div>
      <div id="tk-output" class="muted" style="margin-top:12px">Introduce un ticker del run.</div>`;
  }

  async function load(runId) {
    const ticker = el("tk-input").value.trim().toUpperCase();
    if (!ticker) return;
    const out = el("tk-output");
    out.innerHTML = `<p class="muted">Cargando ${escapeHtml(ticker)}…</p>`;
    let data;
    try { data = await api(`/api/ticker?run_id=${encodeURIComponent(runId)}&ticker=${encodeURIComponent(ticker)}`); }
    catch (e) { out.innerHTML = `<div class="notice">${escapeHtml(e.message)}</div>`; return; }
    const scores = data.scores || [];
    const orders = data.orders || [];
    const prices = data.prices || [];
    out.innerHTML = `
      <h4>${escapeHtml(ticker)}</h4>
      <p class="muted">Precio point-in-time del run; los marcadores indican compras y ventas reales.</p>
      <div id="tk-chart"></div>
      <h4>Operaciones</h4>
      <div class="table-wrap scroll"><table class="data"><thead><tr><th>Fecha</th><th>Operación</th><th>Precio</th><th>Peso antes → después</th><th>Costes</th></tr></thead><tbody>
        ${orders.length ? orders.map((o) => `<tr><td>${escapeHtml(String(o.snapshot_date).slice(0, 10))}</td><td>${escapeHtml(o.side)} · ${escapeHtml(o.reason || "")}</td><td>${fmt(Number(o.price), 3)}</td><td>${fmt(o.weight_before, 3)} → ${fmt(o.weight_after, 3)}</td><td>${fmt((Number(o.commission) || 0) + (Number(o.slippage) || 0), 5)}</td></tr>`).join("") : `<tr><td colspan="5">Sin operaciones.</td></tr>`}
      </tbody></table></div>
      <h4>Evolución de agentes</h4>
      <div>${table(scores.slice(-24), { decimals: 4 })}</div>`;
    global.TFMCharts.clear();
    global.TFMCharts.tickerPrice(el("tk-chart"), prices, orders);
  }

  global.TFM.views.ticker = { render, load };
})(window);
