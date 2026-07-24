# Informe de resultados

## Estado del informe

**Estado actual: pendiente de ejecución empírica del protocolo vigente.**

Este archivo se completará de forma incremental a partir de artefactos reales. Los resultados
anteriores al reinicio metodológico no se incorporan como evidencia del sistema actual. Hasta que
exista una hipótesis congelada y un Confirmatory terminado, no debe afirmarse que la señal está
confirmada ni que produce alfa robusto.

## 1. Identificación de la evidencia

| Campo | Valor |
|---|---|
| Fecha de ejecución | Pendiente |
| Commit Git | Pendiente |
| `catalog_version` | 1 |
| `study_id` exploratorio | Pendiente |
| `hypothesis_id` | Pendiente |
| `study_id` confirmatorio | Pendiente |
| `model_id` | Pendiente o no aplicable |
| `dataset_hash` | Pendiente |
| `evaluation_key` final | Pendiente |
| Veredicto | Pendiente |

## 2. Hipótesis preespecificada

### 2.1 Enunciado congelado

Pendiente de copiar literalmente desde `hypothesis.json`.

### 2.2 Configuración

| Etapa | Variable | Valor final | Origen de la decisión |
|---|---|---|---|
| Temporal |  |  |  |
| Representación |  |  |  |
| Modelo |  |  |  |
| Meta |  |  |  |
| Cartera |  |  |  |

### 2.3 Presupuesto

| Concepto | Previsto | Consumido |
|---|---:|---:|
| Evaluaciones exploratorias | Pendiente | Pendiente |
| Fits caros | Pendiente | Pendiente |
| Recombinaciones meta | Pendiente | Pendiente |
| Backtests | Pendiente | Pendiente |
| Evaluaciones confirmatorias | 23 | Pendiente |
| Tiempo | Pendiente | Pendiente |
| Disco incremental | Pendiente | Pendiente |

## 3. Resultados exploratorios

### 3.1 Secuencia

Para cada paso se informarán todos los valores comparados, no solo el ganador.

| Orden | Variable | Candidatos | Recomendación automática | Elegido | Motivo | Override |
|---:|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

Fuente: `results/studies/<study_id>/evaluation_ledger.parquet`.

### 3.2 Evolución de señal

| Paso | Rank-IC | Cohortes positivas | Spread de cola | Peor era | Bootstrap 90 % |
|---|---:|---:|---:|---:|---|
| Baseline |  |  |  |  |  |
| Final exploratorio |  |  |  |  |  |

### 3.3 Evolución de cartera

| Paso | Alfa anual | IR mediano | Turnover | Eras con alfa positivo | Peor era |
|---|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |
| Final exploratorio |  |  |  |  |  |

### 3.4 Interpretación exploratoria

Pendiente. Debe describir también candidatos rechazados y path dependence. No utilizar expresiones
como «demuestra» o «confirma».

## 4. Evidencia de la hipótesis congelada

### 4.1 Señal por era

| Era | Rank-IC | Fracción positiva | Spread top-universo | Observaciones |
|---|---:|---:|---:|---:|
| 2015–2018 |  |  |  |  |
| 2019–2021 |  |  |  |  |
| 2022–2024 |  |  |  |  |

### 4.2 Traducción económica por era

| Era | Retorno cartera | Retorno SPY | Alfa | IR | Turnover | Drawdown activo |
|---|---:|---:|---:|---:|---:|---:|
| 2015–2018 |  |  |  |  |  |  |
| 2019–2021 |  |  |  |  |  |  |
| 2022–2024 |  |  |  |  |  |  |

### 4.3 Conversión de Rank-IC a alfa

Pendiente de analizar:

- monotonicidad entre rank y retorno;
- comportamiento del top decil y del top operado;
- calibración de retorno excedente esperado;
- exposición activa;
- turnover y costes;
- meses o eras donde existe señal sin alfa;
- meses o eras donde existe alfa sin Rank-IC transversal claro.

## 5. Confirmatory Study

### 5.1 Semillas

| Seed | Rank-IC | Spread de cola | Alfa | IR | Turnover |
|---:|---:|---:|---:|---:|---:|
| 7 |  |  |  |  |  |
| 2026 |  |  |  |  |  |

### 5.2 Perfiles descriptivos

| Perfil | Aplicable | Retorno | Alfa | IR | Turnover | Observación |
|---|---|---:|---:|---:|---:|---|
| balanced |  |  |  |  |  |  |
| growth |  |  |  |  |  |  |
| value |  |  |  |  |  |  |
| quality |  |  |  |  |  |  |
| momentum |  |  |  |  |  |  |
| contrarian |  |  |  |  |  |  |
| defensive |  |  |  |  |  |  |
| garp |  |  |  |  |  |  |

El perfil base elegido en Exploratory forma parte de la hipótesis. Los perfiles de esta tabla son
repeticiones confirmatorias descriptivas: no alteran el perfil congelado ni el veredicto y se
interpretan como sensibilidad de estilo.

### 5.3 Costes

| Comisión | Slippage | Alfa | IR | Turnover | Gate |
|---:|---:|---:|---:|---:|---|
| 0 bps | 5 bps |  |  |  | descriptivo |
| 5 bps | 10 bps |  |  |  | base |
| 10 bps | 20 bps |  |  |  | veredicto |
| 15 bps | 30 bps |  |  |  | severo |

### 5.4 Calendario

| Calendario | Alfa | IR | Turnover | Diferencia frente a base |
|---|---:|---:|---:|---:|
| Base |  |  |  |  |
| Desplazado +1 mes |  |  |  |  |

### 5.5 Placebos reentrenados

| Métrica | Real | Media placebo | Mínimo | Máximo | Real − máximo |
|---|---:|---:|---:|---:|---:|
| Rank-IC |  |  |  |  |  |

No calcular un p-valor inferencial a partir de cinco placebos.

### 5.6 Bootstrap y exclusión de eras

| Prueba | Estimación | IC inferior | IC superior |
|---|---:|---:|---:|
| Rank-IC, bootstrap 95 % |  |  |  |

| Exclusión | Rank-IC restante |
|---|---:|
| Sin 2015–2018 |  |
| Sin 2019–2021 |  |
| Sin 2022–2024 |  |

### 5.7 Permutación transversal

| Permutaciones | Estadístico observado | p-valor add-one | Gate |
|---:|---:|---:|---|
| 9.999 |  |  | `p ≤ 0,10` |

### 5.8 Carteras aleatorias

| Nulo | Simulaciones | CAGR modelo | Media aleatoria | P95 | Percentil modelo |
|---|---:|---:|---:|---:|---:|
| General | 1.000 |  |  |  |  |
| Emparejado por riesgo | 1.000 |  |  |  |  |

Se exige percentil ≥95 en ambos.

## 6. Veredicto predefinido

Marcar exactamente uno:

- [ ] `confirmed`
- [ ] `signal_only`
- [ ] `non_inferior`
- [ ] `rejected`

### Motivos generados por el sistema

Pendiente de copiar desde `decision.json`, sin reescritura favorable.

### Interpretación del autor

Pendiente. Debe respetar el alcance del veredicto y separar significancia estadística, relevancia
económica e implementabilidad.

## 7. Estrés conocido 2025–2026

Esta sección se completa después del veredicto y no puede cambiarlo.

| Año | Retorno cartera | Retorno SPY | Alfa | IR | Drawdown | Comentario |
|---:|---:|---:|---:|---:|---:|---|
| 2025 |  |  |  |  |  |  |
| 2026 |  |  |  |  |  |  |

Etiqueta obligatoria: `known_stress_not_selection`.

## 8. Figuras previstas para el TFM

1. Diagrama Exploratory → congelación → Confirmatory.
2. Rank-IC por cohorte con eras sombreadas.
3. Spread de cola por cohorte.
4. Curva de equity de cartera y SPY.
5. Alfa acumulado y drawdown activo.
6. Turnover y costes por año.
7. Pesos del meta a lo largo del tiempo.
8. Exposición activa y salud de señal.
9. Calibración rank-retorno.
10. Distribuciones nulas con posición del modelo.

Cada figura debe indicar periodo, costes, configuración, fuente y si es exploratoria o
confirmatoria.

## 9. Reglas de redacción

- No elegir retrospectivamente la tabla o periodo más favorable.
- Informar las tres eras y el estrés conocido por separado.
- No confundir Rank-IC con retorno de cartera.
- No llamar «significativo» a un resultado sin prueba y umbral declarado.
- No presentar `non_inferior` como `confirmed`.
- No ocultar perfiles no aplicables ni pruebas fallidas.
- No afirmar causalidad.
- No generalizar a rentabilidad futura sin evidencia forward.
- Toda cifra debe ser rastreable a un archivo de evidencia.
