# Informe acumulativo de resultados

## Estado

Existen **dos Full Studies completos en disco** (`results/studies/`), ejecutados el 2026-07-25 con
la versión del código **anterior** a las correcciones de validez del 2026-07-28. Sus cifras se
conservan y se documentan aquí, pero **no constituyen evidencia válida del TFM**: cuatro de los
defectos corregidos afectan directamente a cómo se eligió el ganador y a cómo se midieron sus
resultados. Este informe explica qué se puede y qué no se puede afirmar con ellas, y qué queda
pendiente.

La ejecución del Study bajo las reglas corregidas está **pendiente de autorización explícita**
(`CLAUDE.md`), porque cambia el ganador con alta probabilidad y consume varias horas.

## Reglas de trazabilidad

Toda cifra debe incluir:

- `study_id`;
- `winner_run_id`;
- hash del catálogo;
- hash del dataset;
- periodo;
- métrica y unidad;
- ruta del artefacto fuente;
- papel de la cifra: selección, confirmación fuera de muestra o diagnóstico.

## Study de referencia (pre-corrección)

Identidad:

- Study: `study-20260725-205429-8a8cbc7f`.
- Run ganador: `run-cec5a3d29e89`.
- Dataset: `41bf819267f732df724ab886e7e8c5196fff24bcbc12f47f380499fa9dfe1902`.
- Catálogo: versión 2 (la versión vigente es 3; **no son comparables**).
- Periodo de selección: 2015–2024, 117 cohortes mensuales.

### Capacidad predictiva (papel: selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0,0737 | `evidence/summary.json` |
| Cohortes positivas | 75,2 % | `evidence/summary.json` |
| Bootstrap por bloques 95 % | [0,0286; 0,1290] | `robustness.json` |
| Permutación transversal | p = 0,0001 | `robustness.json` |
| Placebos de etiqueta (5) | −0,0030 a +0,0037 | `robustness.json` |
| Rank-IC por era | 0,1074 / 0,0831 / 0,0221 | `evidence/summary.json` |

**Lectura.** El intervalo bootstrap excluye cero, la permutación es concluyente y los placebos se
concentran alrededor de cero. La capacidad de ordenación existe y no es un artefacto de
implementación. Es el resultado más sólido del trabajo. Dos matices obligatorios: la degradación
entre eras es fuerte (0,107 → 0,022) y el p-valor **no está corregido por multiplicidad** — con 50
evaluaciones y 17 decisiones, esa corrección es exactamente lo que ahora aporta el Deflated Sharpe
en `attribution.json`.

### Capacidad predictiva por agente (papel: diagnóstico)

| Agente | Rank-IC medio |
|---|---|
| risk | 0,0816 |
| meta_final | 0,0674 |
| meta_equal_weight | 0,0421 |
| value | 0,0212 |
| momentum | 0,0059 |
| growth | 0,0020 |
| quality | −0,0062 |

Concentración de pesos del meta (HHI): 0,642.

**Lectura.** El agente `risk` por sí solo supera al meta apilado. Sin un control de factores, la
interpretación natural de un tribunal es que el sistema redescubrió el efecto de baja volatilidad en
lugar de aprender una ordenación propia. Esa es la razón por la que la regresión con réplicas de
factores y errores Newey-West (`attribution.json`) deja de ser un extra y pasa a ser la pieza que
sostiene —o refuta— la afirmación central. Los cinco agentes se mantienen por decisión de diseño:
aportan cobertura de features y el meta decide a quién atender.

### Traducción a alfa (papel: diagnóstico, **no válido como resultado**)

| Métrica | Valor |
|---|---|
| CAGR cartera | 16,92 % |
| CAGR SPY | 13,81 % |
| Diferencia aritmética de CAGR | +3,11 pp |
| Information Ratio (definición antigua, sin anualizar) | 0,052 |
| Turnover anualizado | 877 % |
| Beat rate | 6/12 años |
| Mediana de alfa anual | −0,05 % |

**Estas cifras no son utilizables**, por cuatro motivos concretos:

1. **Mezclan la era reservada.** El CAGR incluye 2025–2026, que no debía participar en ninguna cifra
   de selección. La versión corregida segmenta selección, confirmación y curva completa.
2. **El IR no era comparable.** Convivían dos fórmulas incompatibles bajo el mismo nombre: 0,052 en
   `winner.json` y 0,098 en `profiles/balanced` para el **mismo** backtest.
3. **La diferencia de CAGR no es el exceso geométrico.** Era una resta, no el cociente de
   acumulados.
4. **El ganador se eligió mal.** Ver más abajo.

**Lectura sustantiva.** Con IC 0,074 sobre ~250 valores, la ley fundamental
(`IR ≈ IC·√BR·TC`) implica un IR teórico en torno a 1,1 frente al ~0,18 realizado: una cartera
long-only de 12 nombres con 877 % de rotación destruía cerca del **85 % de la señal**. El coste
drenaba ~1,3 pp anuales contra una ventaja bruta de ~3,1 pp. Este es el diagnóstico que motiva los
umbrales económicos en puntos básicos, la política de efectivo y la ampliación de `target_size`.

### Estabilidad ante la semilla (papel: diagnóstico)

| Semilla | Rank-IC | Exceso sobre SPY | IR |
|---|---|---|---|
| 42 | 0,0737 | +3,11 pp | 0,052 |
| 2026 | 0,0745 | +2,39 pp | 0,028 |
| 7 | 0,0748 | **−0,51 pp** | **−0,072** |

**Lectura.** El Rank-IC es estable (±0,001) pero **la conclusión económica cambia de signo con la
semilla**. Es el dato más peligroso del trabajo para la palabra «estable» y motiva el ensemble de
cinco semillas por agente y la publicación del rango de alfa entre semillas.

### Defectos que invalidan la selección de este Study

1. **La regla eligió el candidato peor.** Ningún retador de `feature_preset` resultó elegible pese a
   que dos superaban claramente al incumbente: `all` (Rank-IC 0,0958, ventaja pareada +0,0216, mejor
   en el 59,0 % de las cohortes) y el entonces disponible `technical` (0,0994, +0,0265, 64,1 %), que
   falló por **0,00023** frente al margen de −0,01. La causa es que el límite inferior del intervalo
   se ensancha con la diferencia respecto al incumbente, de modo que la prueba castigaba justo a los
   candidatos superiores. Con la puerta corregida y el catálogo de dos presets, **gana `all`**.
2. **Dos decisiones se tomaron sobre ruido.** `market_regime_feature` (ventaja +0,00112) y
   `meta_method` (+0,00033). Con la regla corregida, la primera pasa a `False` por simplicidad y la
   segunda queda registrada como empate técnico.
3. **La puerta era vacía en dos variables.** Al barrer `execution_lag_days` o `snapshot_step_months`
   las rejillas de snapshots son disjuntas y el emparejamiento devolvía `ci_low = 0,0` en silencio,
   lo que hacía pasar automáticamente a todos los candidatos.
4. **Colisión de caché.** `evaluation_key` no incluía el hash del dataset. La misma clave
   `7ec85537…` aparece asociada a dos resultados distintos (CAGR 0,1468 y 0,1692); las decisiones se
   tomaron sobre un dataset y el ganador se recalculó sobre otro.
5. **La ablación por presets no medía lo que decía.** Seis factores derivados de precio se
   inyectaban en el agente momentum fuera de todo condicional, de modo que el preset `fundamental`
   —definido como «nada calculado a partir del precio»— seguía recibiéndolos. Al corregirlo se hizo
   evidente un problema de diseño más profundo: `fundamental` y `technical` dejan agentes enteros sin
   ninguna feature, así que no comparaban qué información necesita cada agente sino qué ocurre al
   amputar parte de la arquitectura. Ambos se retiran del catálogo; quedan `core` y `all`, que
   mantienen los cinco agentes activos.

### Contraste que el modelo aparentemente suspendía

`model_above_p95_both = false`, con percentil del modelo 0,756. Era un **fallo del contraste, no del
modelo**: el nulo de carteras aleatorias daba un percentil 95 de CAGR del **107 % anual**, imposible
para una cartera del S&P 500, porque no exigía cobertura anual completa, no aplicaba la guarda contra
artefactos de datos y no pagaba comisiones mientras el modelo sí las pagaba. Corregido, el contraste
vuelve a ser informativo.

## Qué se puede afirmar hoy

- **Sí:** existe capacidad de ordenación transversal fuera de muestra, estadísticamente
  distinguible de cero y no explicada por la implementación (bootstrap, permutación y cinco placebos
  coherentes).
- **Todavía no:** que esa capacidad se traduzca en alfa neto estable. La evidencia económica
  disponible es frágil ante la semilla, está contaminada por la era reservada y procede de un
  ganador mal seleccionado.
- **Pendiente de medir:** el Rank-IC en 2025–2026, que no se calculaba en ninguna parte del
  proyecto, y la atribución frente a factores de estilo.

## Próximo Study

Con la regla de selección corregida, la identidad de evaluación arreglada, los umbrales en puntos
básicos y el ensemble de semillas. El protocolo de lectura de la era reservada está **pre-registrado**
en `docs/bitacora.md`. Las conclusiones distinguirán capacidad predictiva, estabilidad estadística y
traducción económica, sin seleccionar retrospectivamente por alfa.
