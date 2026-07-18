# Documentación del sistema — TFM: IA aplicada a la selección de acciones

> Guía única del proyecto: qué se quiere conseguir, con qué metodología, con qué arquitectura, y
> qué se ha encontrado. Es la materia prima del TFM (la redacción en LaTeX es la fase de cierre).
> Documento vivo: se actualiza cuando cambia el diseño o aparecen resultados.

---

## 1. Propósito y tesis

Trabajo Fin de Máster sobre **Inteligencia Artificial y Machine Learning**. El objetivo principal
**no** es ganar dinero ni batir a un índice: es **estudiar cómo aprende un sistema de IA**,
evaluar ese aprendizaje con rigor y comprobar si puede tener **utilidad económica**.

La bolsa es el **banco de pruebas**, no el fin. Se elige porque ofrece un entorno difícil,
ruidoso, no estacionario y con una métrica de éxito clara (batir al S&P 500), pero el mérito
académico está en el **método**, no en un número de rentabilidad aislado.

Consecuencia central: **un resultado negativo bien medido es un entregable válido**. Si el sistema
no aprende a ordenar activos fuera de muestra, decirlo con evidencia es tan valioso como una
cartera rentable. Elegir la mejor semilla, confundir suerte con aprendizaje u ocultar un artefacto
de datos invalidaría el trabajo. Principios: **académico, reproducible, explicable y honesto**.

**La pregunta de investigación**: *¿aprende el sistema a ordenar acciones por su retorno futuro,
de forma estable y fuera de muestra, y ese aprendizaje se traduce en utilidad económica neta?*

---

## 2. Métrica de éxito: el aprendizaje, no la rentabilidad

La métrica principal es el **rank-IC out-of-sample**: la correlación de Spearman, en cada corte
transversal, entre el orden que predice el modelo y el orden que de verdad ocurrió. Se calcula
sobre el **meta-score final** (el que opera la cartera), no sobre agentes individuales.

Por qué el rank-IC y no la rentabilidad: con una señal débil, la rentabilidad de una cartera
concentrada la puede dominar un puñado de aciertos afortunados o —como se observó— **un único
artefacto de datos**. El rank-IC mide si el modelo *ordena bien*, que es lo que de verdad se
pregunta. La rentabilidad se reporta como **consecuencia**, con CAGR real (nunca acumulados
compuestos que inflan y que domina un solo año), y solo tras limpiar artefactos.

Referencia de escala: en la industria un factor se considera útil a partir de rank-IC ~0.03-0.05
sostenido. Se acompaña de **significancia estadística** (bootstrap por bloques temporales, porque
las cohortes no son independientes) y de **tests de robustez/placebo** para distinguir señal de
azar.

---

## 3. Arquitectura del sistema (flujo de datos)

```
descarga → dataset point-in-time → features (+ artefactos) → agentes LightGBM → meta-agente
        → cartera (+ perfil de inversor) → backtest → informe HTML
                                                     + barrido de ablations → decisión → run final
```

Cada etapa se ejecuta por separado (`RUN_MODE`) o encadenada. El comando de principio a fin es
`RUN_MODE=full_study` (ver §9).

### 3.1 Datos y universo (point-in-time)
- **Universo dinámico** del S&P 500 por fecha (composición histórica real, no la actual): una
  señal en 2016 solo ve las empresas que estaban en el índice en 2016. Elimina el sesgo de
  inclusión anticipada.
- **Fechas de publicación reales** de SEC EDGAR (`filingDate`): un fundamental solo es observable
  el día en que se publicó, no el día del cierre fiscal. Evita el lookahead.
- **Sesgo de supervivencia medido por año** (no solo declarado): los quebrados no tienen datos en
  fuentes gratuitas; la cobertura pasa de ~50 % en 2000 a ~92 % en 2024. Por eso el sistema se
  centra en **2016+** (más cobertura, menos sesgo).

### 3.2 Dataset point-in-time
Panel `(ticker, snapshot_date)` que reconstruye lo observable en cada fecha: de cada empresa, su
último informe *realmente publicado*. Sin relleno hacia atrás, sin mezclar magnitudes.

### 3.3 Features y artefactos
Factores base (GARP: calidad, crecimiento, valoración; momentum relativo), todos rankeados en el
corte transversal. Encima, un **sistema de artefactos activables** (§4).

### 3.4 Agentes LightGBM
Tres agentes especializados (**calidad, momentum, valor**), cada uno un modelo **LightGBM**
(árboles con gradient boosting, que captura interacciones no lineales que un modelo lineal
promedia a cero). Objetivo `rank_regression`: regresión sobre el percentil transversal del
retorno futuro, alineado con el rank-IC. Entrenamiento **walk-forward**: en cada reentreno solo se
usa historia anterior con etiqueta ya realizada (`label_end_date <= fecha_de_reentreno`).

### 3.5 Meta-agente
Combina los tres agentes en el `meta_score`. Por defecto pondera por el **rank-IC reciente** de
cada agente (`meta_type=rank_ic`), que se midió superior al equiponderado. El `meta_rank` (el
percentil del meta_score) es lo que consume la cartera.

### 3.6 Cartera y backtest
Selección top-N flexible (**8-12 posiciones** por defecto, peso máx 15 %) con reglas de rotación
(umbral de ventaja, expulsión por caída, sin tenencia mínima). Backtest con costes (5 pb comisión
+ 10 pb slippage) y **guarda anti-artefactos**: un retorno mensual imposible de una posición
(>200 %) se trata como dato corrupto y se neutraliza (§4.3). Métricas: rank-IC del meta_final,
CAGR real cartera vs SPY, beat rate, drawdown, turnover.

---

## 4. Sistema de artefactos activables

Un **artefacto** es un bloque de features/contexto que se activa o desactiva por un flag. Todos
son **point-in-time** (verificados con tests de no-lookahead). El barrido los prueba de uno en
uno como *ablations* para medir cuáles suben el rank-IC. Catálogo en `module/artifacts.py`:

| Artefacto | Qué añade |
|---|---|
| `neutralize_by_sector` | Rankea los factores dentro de sector en vez de global. |
| `fundamental_momentum` | Tendencia de ROE/márgenes + descomposición del cambio de P/E en su parte de precio y su parte fundamental. |
| `market_regime_feature` | Régimen bull/bear del SP500 (vs su media) + interacciones factor×régimen. |
| `price_momentum_multi` | Aceleración (r3m−r12m), reversión (−r1m), volatilidad reciente del activo. |
| `moving_averages` | Precio vs SMA6/SMA12, distancia al máximo de 12m (tendencia individual). |
| `regime_extended` | Volatilidad y drawdown del SP500 (contexto macro más rico). |
| `quality_growth_derived` | Tendencia de ROE, estabilidad de márgenes, sorpresa de crecimiento. |

### 4.2 Cadencia como escenario
El reentreno puede ser **trimestral / semestral / anual** (`fundamental_step_months`), barrido como
escenario. La revisión de cartera es mensual.

### 4.3 Guarda anti-artefactos (rentabilidad honesta)
El estudio previo reveló que un CAGR aparente del 18 % se debía casi por completo a un **artefacto
de datos**: julio de 2010, +953 % en un mes en una posición (precio corrupto). La guarda neutraliza
cualquier retorno mensual de una posición mayor que `max_monthly_position_return` (200 %) y lo
registra, para que la rentabilidad reportada sea honesta.

---

## 5. Barrido de ablations y selección automática

El estudio es un **barrido de ablations dirigidas** (no producto cartesiano, que sobreajustaría
por selección): baseline vs baseline + cada artefacto aislado, más hiperparámetros, cadencia y
ventana de entrenamiento (`escenarios/rejilla_base.py`).

**Decisión automática, sin intervención humana** (`decide_accepted_artifacts`): un artefacto se
**acepta** si su rank-IC del meta_final mejora al baseline de forma **estable** —diferencia pareada
por fecha positiva en media y en más de la mitad de las fechas—. La **configuración final** =
baseline + artefactos aceptados + mejores hiperparámetros. Se registra en `artifact_decision.json`
qué entró, qué no, y por qué.

La reutilización por huella SHA-256 evita recomputar etapas compartidas entre escenarios.

---

## 6. Perfiles de inversor (explicabilidad como funcionalidad)

El sistema explica *por qué* cada acción está arriba: cada agente aporta su rango (calidad,
momentum, valor). Sobre eso se construyen **perfiles de inversor** que, entre las **buenas**
acciones del meta (percentil alto), reordenan según estilo — no siempre cogen el top-N puro
(`module/profiles.py`):

`balanced` (referencia), `conservative` (calidad + estabilidad), `aggressive` (momentum),
`value` (barato y bueno), `quality` (mejor negocio), `momentum` (fuerza relativa), `garp`
(equilibrio), `contrarian` (bueno pero castigado, apuesta a reversión).

Cada perfil se mide como un backtest (misma señal, distinta cartera): permite comparar el
trade-off rentabilidad/riesgo de cada estilo usando el mismo modelo.

---

## 7. Robustez / placebo (credibilidad)

Sobre la configuración final se ejecutan tests que demuestran que el resultado no es suerte
(`module/robustness.py`):

- **Permutación de etiquetas** (placebo): se reentrena con los retornos futuros barajados; el
  rank-IC debe **colapsar a ~0**. Si no colapsa, hay fuga de información (también detecta leakage).
- **Carteras aleatorias (Monte Carlo)**: la cartera del modelo se compara con ~1000 aleatorias del
  mismo tamaño; su percentil dice si su rendimiento es distinguible del azar.
- **Bootstrap por bloques** del rank-IC (`module/stats.py`): intervalo de confianza que respeta el
  solapamiento temporal.
- **Leave-one-year-out**: se quita cada año para ver si el resultado depende de uno o dos.

---

## 8. Resultados (estudio full, ancla 2016-2026)

El resultado es **matizado y en dos planos opuestos**, que es lo que hace este TFM interesante y
honesto: el modelo de IA **no aprende** de forma significativa, pero una cartera con sesgo de
estilo defendible **sí bate al mercado** de forma consistente y limpia.

### 8.1 El aprendizaje (rank-IC) no es significativo

El barrido de ablations aceptó automáticamente **un solo artefacto**: la neutralización por
sector (rank-IC del meta_final 0.0036 → 0.0094, mejor que el baseline en el 59 % de las fechas).
Los otros seis empeoran o no aportan:

| artefacto | rank-IC con él | Δ vs baseline | ¿aceptado? |
|---|---|---|---|
| **neutralize_by_sector** | **+0.0094** | +0.0057 | **sí** |
| quality_growth_derived | +0.0041 | +0.0005 | no (mejor solo en 44 %) |
| regime_bull_bear | −0.0003 | −0.0039 | no |
| moving_averages | −0.0009 | −0.0045 | no |
| price_momentum_multi | −0.0009 | −0.0045 | no |
| regime_extended | −0.0021 | −0.0056 | no |
| fundamental_momentum | −0.0024 | −0.0060 | no |

Confirma un patrón consistente en todo el proyecto: **añadir features no crea señal**; solo la
reorganización del ranking dentro de sector aporta algo marginal.

**El sistema final (con sector) alcanza un rank-IC de +0.0036**, y NO es distinguible del azar:
- Intervalo de confianza por bootstrap: **[−0.019, +0.024]** (cruza cero).
- **Test de placebo** (permutación de etiquetas): con retornos barajados el rank-IC colapsa a
  −0.0009 (correcto, no hay fuga), pero el **p-valor es 0.20** — 1 de cada 5 permutaciones
  aleatorias iguala o supera al modelo real. **El aprendizaje no supera al azar.**
- Leave-one-year-out: el rank-IC oscila entre +0.0008 y +0.0085 quitando cada año; ninguno lo
  sostiene en solitario, pero todos son ≈ 0.

**Conclusión del plano de aprendizaje**: con datos gratuitos, universo del S&P 500 y factores
GARP+momentum, el modelo LightGBM **no aprende a ordenar acciones de forma estadísticamente
significativa**. Es el resultado honesto, medido con rigor (placebo + bootstrap + estabilidad).

### 8.2 La rentabilidad: los perfiles de estilo sí baten al SPY (limpio)

Con la guarda anti-artefactos activa (sin el +953 % corrupto de estudios previos), varios perfiles
de inversor baten al SPY de forma consistente en 2016-2026:

| perfil | CAGR | vs SPY (anual) | años que baten | drawdown máx |
|---|---|---|---|---|
| quality | 20.7 % | **+4.5 %** | 45 % | 35 % |
| value | 20.6 % | +4.4 % | 55 % | 30 % |
| conservative | 20.6 % | +4.4 % | 55 % | 32 % |
| garp | 19.9 % | +3.7 % | **64 %** | 41 % |
| contrarian | 18.1 % | +1.9 % | 55 % | 38 % |
| momentum | 15.6 % | −0.6 % | 45 % | 34 % |
| aggressive | 14.9 % | −1.3 % | 55 % | 33 % |
| **balanced** (el meta ML puro) | 14.7 % | **−1.5 %** | 36 % | 48 % |

### 8.3 La lectura clave

El perfil **balanced —el que confía en el meta-score del ML— es el peor** (−1.5 % vs SPY, drawdown
48 %). Los que ganan son los que imponen un **sesgo de estilo humano** (quality, value, GARP)
entre las candidatas. Esto es coherente con los dos planos: como el ML no ordena bien (rank-IC
≈ 0), seguir su ranking puro no bate al mercado; pero inclinar la cartera hacia **calidad y valor**
captura las **primas de factor clásicas**, que sí existen. GARP bate al SPY el 64 % de los años.

**En una frase**: el aprendizaje automático no aporta señal, pero la estructura de factores con un
sesgo de estilo defendible (calidad/valor/GARP) sí bate al mercado de forma consistente y limpia.
El valor del sistema no está en su ML, sino en explotar de forma disciplinada primas de factor
conocidas —y en haberlo **demostrado con honestidad**, separando lo que aprende (poco) de lo que
rinde (los factores).

---

## 9. Cómo ejecutar

Requiere `data/raw` ya descargado (la descarga es un paso aparte, `RUN_MODE=download`).

- **Estudio completo de principio a fin** (recomendado):
  `RUN_MODE=full_study RUN_SCOPE=full python main.py`
  Ejecuta barrido de ablations → decisión automática → run final optimizado → perfiles → robustez
  → informes HTML. Sin decisiones humanas.
- **Etapas sueltas**: `dataset`, `features`, `agents`, `backtest`, `report`, `experiments`.
- **Ver los informes**: `python servir_html.py` y abrir
  `http://localhost:8000/results/escenarios/comparison.html` (un servidor local hace falta porque
  algunas pestañas cargan CSVs grandes por `fetch`).

---

## 10. Limitaciones (medidas, no estimadas)

1. **Sesgo de supervivencia parcialmente irreducible**: los quebrados no tienen datos gratuitos.
   Se mitiga centrándose en 2016+ (cobertura ~65-92 %) y se reporta por año.
2. **Restatements invisibles**: las series de fundamentales son valores actuales; EDGAR fecha
   *cuándo* se publicó, pero el *valor* podría estar reexpresado. Lookahead residual no eliminable
   con datos gratuitos.
3. **Muestra pequeña de eras independientes**: intervalos de confianza anchos; por eso la
   significancia se mide con bootstrap por bloques y se exige estabilidad temporal.
4. **Selección temporal declarada**: centrarse en 2016+ es una decisión basada en cobertura y
   diagnóstico; se reporta también el histórico completo para que el lector vea cuánto cambia.

---

## 11. Reproducibilidad

Todo el estudio es re-ejecutable con un comando. Cada run lleva su `manifest.json` con la
configuración, la semilla, las versiones y los hashes de los inputs. La **bitácora**
(`docs/bitacora.md`) registra el porqué de cada decisión y cada resultado —el hilo narrativo del
desarrollo—. Los tests (`pytest tests/`) cubren la ausencia de lookahead, las reglas de cartera, la
guarda anti-artefactos, la decisión automática, los perfiles y la robustez.
