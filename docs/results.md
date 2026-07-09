# Cómo leer `results/<run>/viewer/index.html`

El informe es una **única página HTML** en español, autocontenida (sin dependencias externas),
generada por `module/viewer/`. No hay 16 páginas ni un dashboard aparte: cada sección existe
porque responde a una pregunta concreta del TFM o ayuda a depurar el modelo — ese es el criterio
de utilidad aplicado en todo `module/viewer/shared.py`.

Se abre localmente en cualquier navegador: `results/<run>/viewer/index.html`.

## Navegación (7 secciones)

### 1. Resumen

KPIs ejecutivos de un vistazo: Alpha vs SPY, CAGR (cartera y benchmark), Sharpe/Sortino, Max
Drawdown, Information Ratio + Tracking Error, y el t-stat del retorno en exceso con el número de
periodos observados. La frase inicial dice explícitamente si la estrategia batió o no al
benchmark en esa ejecución concreta, seguida del aviso de muestra pequeña
(`SMALL_SAMPLE_CAVEAT`, definido en `module/backtest/artifacts.py`) — el t-stat se reporta como
indicativo, no como prueba de significancia.

### 2. Rendimiento

El gráfico central del trabajo: crecimiento de 1 € en la cartera neta frente a SPY, el alpha
acumulado en el tiempo, y el perfil de drawdown. Debajo, una tabla de **episodios de drawdown**
(pico, valle, fecha de recuperación, profundidad, duración en días, si se recuperó) — útil para
distinguir una caída controlada de una que nunca se recuperó dentro de la ventana simulada.

### 3. Cartera

Dos tablas: la **cartera actual** (ticker, sector, peso `hybrid_weight`, `manager_score`,
`final_score` del modelo, tipo de oportunidad, estado de tesis) y las **últimas 15 operaciones**
del diario de acciones (`action_journal.csv`), cada una con su motivo categorizado, días en
cartera y retorno en exceso resultante — para ver no solo qué se compró/vendió, sino si esa
decisión salió bien.

### 4. Aprendizaje — la sección que demuestra que el sistema aprende

Esta es la sección más importante desde el punto de vista metodológico. Incluye:

- **Nota de la fecha de corte**: explica en una frase que antes del corte el modelo se
  reentrena trimestralmente (aprendizaje real) y después queda **congelado** — lo que se ve
  después del corte mide cómo envejece ese modelo fijo, no aprendizaje nuevo.
- **Evolución de los pesos aprendidos del meta-agente** (gráfico): cómo cambia en el tiempo el
  peso que el sistema da a cada uno de los 4 agentes especializados (Calidad, Crecimiento,
  Infravaloración, Alpha), con una línea vertical marcando el corte.
- **Qué agente domina y por qué** (frase generada desde los datos reales del snapshot más
  reciente): identifica el agente con mayor peso y su rank-IC histórico medio, para no tener que
  interpretar el gráfico a ojo.
- **Rank-IC out-of-sample del agente Alpha por snapshot** (gráfico, con media móvil de 12
  periodos): mide si la capacidad predictiva mejora, se mantiene plana u oscila con el tiempo —
  se reporta tal cual sale, sin forzar una narrativa de mejora.
- **Rank-IC por año** (tabla): media y t-stat aproximado por año calendario, para saber si un
  año concreto domina el promedio.
- **¿Qué horizonte de predicción funciona mejor?**: comparación de rank-IC a 3/6/12 meses del
  agente Alpha, con la conclusión de por qué se mantiene (o no) el horizonte por defecto de 12
  meses — responde con datos, no con supuestos, a si conviene reentrenar a un plazo distinto.

**Cambio 2026-07 relevante para esta sección**: el meta-agente pasó de aprender los pesos
minimizando error cuadrático (NNLS sobre el alpha en bruto) a aprenderlos directamente a partir
del **rank-IC** de cada agente. La versión anterior premiaba a agentes de baja varianza (como
`quality_probability`, con rank-IC cercano a cero) por encima de agentes que sí ordenaban bien las
acciones (`alpha_probability`) — un desajuste entre cómo se mide la calidad del modelo (rank-IC)
y cómo aprendía (error cuadrático). Con el cambio, un agente sin poder de ranking recibe peso ~0.

### 5. Posiciones

Atribución de rendimiento por posición cerrada (lote FIFO): retorno total, anualizado, retorno del
benchmark en el mismo periodo de tenencia, y retorno en exceso — con un gráfico de las mejores y
peores 16 posiciones por retorno en exceso acumulado, y una tabla con las 20 mejores en detalle
(incluyendo el motivo de salida).

### 6. Metodología

Pensada para que cualquiera entienda el diseño sin leer el código:

- Descripción en prosa de los 4 agentes especializados + el meta-agente, con la definición exacta
  de cada target (reutilizada de `model_explainability.json`).
- Mecánica de "entrenar hasta un corte, luego congelar" con las **fechas concretas de esa
  ejecución** (no genéricas): desde cuándo hasta cuándo se entrenó, cuántos años de historia usó
  cada reentrenamiento, y cómo se simula después del corte.
- Tabla de "ficha técnica": fecha de corte, años de entrenamiento previos, cadencia de
  reentrenamiento, horizonte de predicción, ventana móvil de historia, tamaño del universo.

### 7. Debug / TFM

Bloque separado del contenido ejecutivo, para la defensa técnica: diagnóstico walk-forward
(tasa de fallback al GARP determinista, columnas de fase `training`/`frozen`), y enlaces a los
CSV de auditoría pesados que **no** se embeben en el informe (ver más abajo).

## Qué NO está en el informe (y dónde encontrarlo)

Las tablas grandes por ticker/snapshot viven en `results/<run>/audit/` y solo se **enlazan** desde
la sección Debug/TFM, nunca se embeben:

```text
results/<run>/audit/portfolio_transactions.csv       — cada compra/venta individual
results/<run>/audit/portfolio_monthly_holdings.csv   — composición de cartera mes a mes
results/<run>/audit/portfolio_turnover.csv           — rotación de cartera
results/<run>/audit/portfolio_review_diagnostics.csv — por qué cada candidato entró/no entró
results/<run>/audit/universe_monthly_scores.csv      — scores de TODO el universo, cada mes
results/<run>/audit/universe_top_candidates.csv      — ranking completo por snapshot
results/<run>/audit/watchlist_history.csv            — histórico completo de watchlist
results/<run>/audit/research_ai/<ticker>.json        — research generado por empresa
```

Los CSV compactos que el viewer sí consume están en la raíz de `results/<run>/` (uno por sección
anterior: `portfolio_vs_benchmark.csv`, `current_portfolio.csv`, `action_journal.csv`,
`meta_weights_by_snapshot.csv`, `model_walk_forward_diagnostics.csv`,
`label_horizon_comparison.csv`, `position_performance.csv`, `sell_reasons_summary.csv`,
`portfolio_monthly_summary.json`).

## Regenerar solo el informe

Si ya existen los CSV de un run y solo se quiere reconstruir el HTML (p. ej. tras cambiar
`module/viewer/`), basta con:

```python
# environment.py
RUN_MODE = "viewer"   # o "report", que apunta al mismo archivo
```

```bash
python main.py
```

No hace falta re-descargar datos ni reentrenar el modelo — `viewer`/`report` solo leen los CSV ya
escritos por `backtest`.
