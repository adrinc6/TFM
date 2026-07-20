# Informe final: protocolo de evidencia y plantilla de resultados

> Este archivo no inventa cifras. Es la plantilla narrativa y de auditoría que debe completarse únicamente con un full study terminado bajo el protocolo vigente. La documentación técnica está en [doc.md](doc.md) y las decisiones históricas en [bitacora.md](bitacora.md).

## 1. Identificación del estudio

| Campo | Valor a publicar |
|---|---|
| `study_id` | Tomado de `results/studies/<study_id>/study_manifest.json`. |
| Nombre | Nombre introducido en Full study. |
| Hipótesis | Campo `hypothesis` del manifiesto y de `decision.json`. |
| Fecha de ejecución | `created_at_utc` del manifiesto. |
| Universo / benchmark | S&P 500 histórico / SPY. |
| Ventana de selección | Inicio OOS hasta 2024. |
| Reserva final | 2025–2026. |

La hipótesis debe escribirse antes de ver el ganador. Si se modifica después, el cambio debe quedar documentado como un study nuevo, no sustituir el manifiesto previo.

## 2. Método que debe declararse

El sistema parte de datos fundamentales y OHLCV point-in-time. Las features se asignan a bloques económicos; cinco agentes especializados producen scores walk-forward; familias LightGBM, Elastic Net y CatBoost se combinan dentro de cada agente; un meta-agente produce el score final.

La selección del modelo se hace exclusivamente con Rank-IC OOS de `meta_final` hasta 2024. El backtest no elige el modelo. Después de fijarlo, la fase de cartera puede elegir construcción por Information Ratio; costes de ejecución se reportan en nueve stresses y no se usan como variable de optimización.

El informe debe especificar si el finalista utiliza poda, gating, ensemble o stacking, y debe indicar los fallbacks activados cuando no había suficiente historia OOS.

## 3. Tabla de configuración final

Completar desde `decision.json`, `study_manifest.json` y manifiestos del run final:

| Categoría | Parámetro | Valor final | Evidencia de selección |
|---|---|---:|---|
| Calendario | `execution_lag_days`, ventana, cadencias y horizonte |  | Fase 1/2, Rank-IC hasta 2024. |
| Modelos | objetivo y familias |  | Ablación y estabilidad. |
| Meta | tipo, lookback y pesos |  | Diagnósticos de agentes/meta. |
| Factores | bloques, agentes, modo de selección |  | Ablaciones y cobertura. |
| Cartera | tamaño, percentiles, rotación y peso |  | Fase de cartera, Information Ratio. |
| Costes | comisión y slippage base |  | Supuesto base; no criterio de selección. |

No deben publicarse como “ganadores” semilla, comisión o slippage. Si apareciesen en `overrides` de selección, el study corresponde a un protocolo antiguo y debe etiquetarse como histórico.

## 4. Plano de aprendizaje

| Métrica | Valor | Interpretación requerida |
|---|---:|---|
| Rank-IC medio hasta 2024 |  | Señal media usada para seleccionar. |
| Fracción de cohortes positivas |  | Estabilidad direccional. |
| Desviación típica |  | Variabilidad entre snapshots. |
| Bootstrap por bloques |  | Debe indicar intervalo y si cruza cero. |
| Leave-one-year-out |  | Debe indicar si un año domina. |
| Placebo de etiquetas |  | Debe comparar señal real y distribución permutada. |
| Sensibilidad a semilla |  | Debe indicar dispersión y no solo el mejor caso. |

Un Rank-IC positivo con bootstrap que cruza cero se debe presentar como evidencia inconclusa, no como prueba de capacidad predictiva. Un Rank-IC positivo en la reserva con pocas cohortes es informativo, pero no suficiente por sí solo.

## 5. Reserva 2025–2026

| Métrica reservada | Valor | Lectura |
|---|---:|---|
| Cohortes disponibles |  | Tamaño efectivo de la reserva. |
| Rank-IC medio |  | Generalización temporal. |
| Fracción positiva |  | Consistencia de la reserva. |
| Diferencia frente a selección |  | Estabilidad o degradación. |

La reserva se consulta una sola vez tras seleccionar finalista. Si se usa para alterar hiperparámetros o cartera, deja de ser reserva y debe abrirse un nuevo período holdout.

## 6. Ablaciones y explicabilidad

El informe debe incluir:

- baseline completo frente a conjunto básico;
- retirada individual de cada bloque;
- retirada individual de cada agente;
- comparación de familias y ensembles;
- modo de selección de features;
- cobertura, importancia y estabilidad de las métricas principales;
- contribuciones locales para ejemplos de acciones, dejando claro que explican el modelo y no prueban causalidad económica.

Una ablación negativa es un resultado útil: indica que el componente no aporta señal incremental en el período evaluado. No debe ocultarse para simplificar el relato.

## 7. Plano económico

| Métrica | Cartera base | Stress de costes | Benchmark |
|---|---:|---:|---:|
| CAGR |  |  |  |
| Information Ratio |  |  |  |
| Máximo drawdown |  |  |  |
| Alfa anual medio |  |  |  |
| Beat rate |  |  |  |

Los ocho perfiles usan el mismo modelo seleccionado. El perfil recomendado se elige por Information Ratio entre ellos, pero no modifica la evidencia de aprendizaje. La mejora económica que desaparece con costes plausibles debe describirse como frágil.

## 8. Limitaciones obligatorias

1. Múltiples pruebas y riesgo de selección por azar.
2. Universo S&P 500, sesgo de supervivencia residual y cobertura desigual.
3. Dependencia temporal entre snapshots y tamaño reducido de la reserva.
4. Costes simulados frente a ejecución real.
5. Correlación no implica causalidad económica.
6. Posibles diferencias de implementación entre familias de modelo.
7. Métricas o fuentes no incluidas por falta de series PIT verificadas.

## 9. Conclusión permitida

La conclusión debe responder a la hipótesis original y distinguir una de estas categorías:

- **Validación provisional:** mejora estable hasta 2024, reserva no degradada y robustez razonable.
- **Evidencia mixta:** señal media positiva, pero bootstrap, LOYO, placebo o reserva insuficientes.
- **No validado:** no supera baseline o depende de un período, coste o configuración frágil.

No se debe afirmar rentabilidad futura ni superioridad económica permanente. La afirmación máxima defendible es la que soporten los artefactos del study concreto.

## 10. Reproducción y trazabilidad

```powershell
$env:RUN_MODE = "full_study"
$env:RUN_SCOPE = "full"
python -u main.py
```

El informe debe enlazar el `study_id`, `decision.json`, manifiesto y run final exactos. Los studies anteriores al endurecimiento metodológico se mantienen como antecedentes, no como resultado final.
