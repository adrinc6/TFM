# Explicacion de resultados

Este proyecto separa los resultados en dos capas para que sea facil seguir la pista sin ahogarse en datos.

## Estructura recomendada

- `results/<run>/viewer/index.html`: punto de entrada visual.
- `results/<run>/final_report.html`: informe ejecutivo de la ejecucion.
- `results/<run>/result_manifest.json`: inventario automatico de CSV, HTML y PNG, con proposito y prioridad de lectura.
- `results/<run>/logs/pipeline.log`: log completo de la ejecucion.
- `results/<run>/*.csv`: archivos ejecutivos para seguimiento normal.
- `results/<run>/audit/*.csv`: archivos pesados para reconstruir o depurar.
- `results/<run>/viewer/charts/*.png`: graficos principales generados con matplotlib.

## Archivos ejecutivos

- `executive_summary.csv`: una fila con metricas clave de la ejecucion.
- `current_portfolio.csv`: cartera actual, pesos, scores, estado de tesis y tesis de salida.
- `tracking_dashboard.csv`: evolucion mensual compacta de portfolio, benchmark, alpha, compras y ventas.
- `action_journal.csv`: compras y ventas en una tabla, con resultado economico cuando la posicion esta cerrada.
- `position_performance.csv`: retorno total/anualizado por accion y comparacion contra benchmark en el periodo de holding.
- `buy_rationale.csv`: por que se compro cada accion: ranking, scores, alternativa y tesis.
- `sell_reasons_summary.csv`: por que se vendio, agregado por categoria.
- `sector_exposure.csv`: exposicion sectorial mensual.
- `top_opportunities_latest.csv`: mejores oportunidades actuales del universo.
- `strategy_learning_log.csv`: evidencia agregada de patrones que han funcionado o fallado por tipo de entrada/salida.
- `improvement_backlog.csv`: pistas automaticas para futuras mejoras de pesos, umbrales y reglas.
- `watchlist.csv`: candidatos actuales con tesis resumida.

## Archivos de auditoria

- `audit/portfolio_monthly_holdings.csv`: foto completa posicion x mes.
- `audit/rebalance_report.csv`: transacciones y decisiones HOLD/WATCH/REDUCE completas.
- `audit/universe_monthly_scores.csv`: scoring mensual completo del universo.
- `audit/universe_top_candidates.csv`: candidatos historicos por fecha.
- `audit/universe_quarterly_fundamental_review.csv`: revisiones trimestrales de fundamentales.
- `audit/universe_monthly_price_update.csv`: revisiones mensuales intermedias por precio.
- `audit/watchlist_history.csv`: watchlist historica.
- `audit/research_ai_history.csv`: research historico.

## Orden de lectura

1. `viewer/index.html`
2. `viewer/current_portfolio.html`
3. `viewer/action_journal.html`
4. `viewer/position_performance.html`
5. `viewer/buy_rationale.html`
6. `viewer/sell_reasons.html`
7. `viewer/watchlist.html` y `viewer/top_opportunities.html`
8. `viewer/strategy_learning.html`
9. `viewer/audit.html` solo si necesitas tablas completas.

## Filosofia actual

El sistema revisa todo el universo en cada snapshot. Las revisiones trimestrales refrescan fundamentales y los meses intermedios actualizan precio, momentum y valoracion ajustada. La configuracion descarga datos desde `DATA_START_DATE`, pero el dataset maestro solo materializa snapshots desde la primera fecha util para entrenamiento: `PORTFOLIO_START_DATE - MAX_WALK_FORWARD_TRAINING_YEARS`, respetando siempre `DATA_START_DATE`. El scoring ML usa walk-forward con al menos `MIN_WALK_FORWARD_TRAINING_YEARS` anos efectivos de entrenamiento y como maximo una ventana movil de `MAX_WALK_FORWARD_TRAINING_YEARS`: en cada fecha entrena con filas desde `fecha_actual - MAX_WALK_FORWARD_TRAINING_YEARS` hasta la propia fecha actual, usando todo lo disponible hasta ese momento. Para evitar fuga de informacion, las filas cuyo alpha futuro aun no seria observable entran igualmente, pero ese componente se rellena con fallback GARP. La curva principal es neta de costes estimados y conserva tambien la curva bruta.
