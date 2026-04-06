# Guia detallada de features por agente

## Objetivo
Este documento detalla exactamente que columnas consume cada agente, cuales se derivan dentro del propio agente, y que significa cada columna en el contexto del agente.

Fuente de verdad usada para esta guia:
- `environment.py` para `*_FEATURE_COLUMNS` (listas autoritativas de entrada por agente).
- `module/agents/*.py` para features derivadas en `_prepare`.

## Convenciones importantes
- `*_FEATURE_COLUMNS`: columnas base que el agente intenta consumir para entrenar/prediccion.
- `*_FEATURE_EXCLUDE`: en el estado actual estan vacias en todos los agentes.
- Una columna puede existir en el dataset y aun asi no entrar al modelo final si el selector de features la descarta en ese fold.
- Algunas columnas derivadas se crean solo si existen sus columnas base.
- Columnas `_zsector`:
  - Se generan en la normalizacion sectorial del pipeline.
  - Se usan principalmente en `FundamentalAgent` cuando estan disponibles.

---

## 1) FundamentalAgent

### 1.1 Columnas base configuradas (`FUNDAMENTAL_FEATURE_COLUMNS`)
- `roe`: rentabilidad sobre patrimonio (beneficio neto / equity). Mayor suele ser mejor.
- `roa`: rentabilidad sobre activos (beneficio neto / activos). Eficiencia global del activo.
- `roi`: retorno sobre inversion. Eficiencia del capital invertido.
- `roic`: retorno sobre capital invertido operativo. Calidad de asignacion de capital.
- `net_margin`: margen neto (beneficio neto / ventas). Rentabilidad final.
- `gross_margin`: margen bruto ((ventas - coste) / ventas). Poder de pricing/producto.
- `fcf_margin`: margen de caja libre (FCF / ventas). Calidad de conversion a caja.
- `ebitda_margin`: margen EBITDA. Rentabilidad operativa pre-amortizacion.
- `operating_margin`: margen operativo (EBIT / ventas). Eficiencia operativa.
- `current_ratio`: liquidez corriente (activo corriente / pasivo corriente).
- `quick_ratio`: liquidez acida (sin inventarios). Liquidez de corto plazo mas estricta.
- `debt_equity`: deuda / patrimonio. Apalancamiento financiero.
- `debt_to_ebitda`: deuda / EBITDA. Capacidad de repago con resultado operativo.
- `interest_coverage`: cobertura de intereses (EBIT / intereses). Riesgo financiero.
- `revenue_yoy_growth`: crecimiento interanual de ventas.
- `net_income_yoy_growth`: crecimiento interanual de beneficio neto.
- `eps_yoy_growth`: crecimiento interanual de EPS.
- `fcf_yoy_growth`: crecimiento interanual de flujo de caja libre.
- `operating_income_yoy_growth`: crecimiento interanual de EBIT/resultado operativo.
- `total_debt_yoy_growth`: crecimiento interanual de deuda total.
- `roa_change_yoy`: cambio YoY de ROA (mejora/deterioro de eficiencia).
- `gross_margin_change_yoy`: cambio YoY de margen bruto.
- `current_ratio_change_yoy`: cambio YoY de liquidez corriente.
- `accruals_ratio`: intensidad de accruals; alto puede sugerir menor calidad contable.
- `capex_to_revenue`: capex / ventas; intensidad de inversion.
- `consecutive_losses`: numero de trimestres consecutivos con perdidas.
- `earnings_quality`: aprox. calidad del beneficio (alineacion beneficio-contable vs caja).
- `piotroski_fscore`: score binario de fortaleza financiera (0-1 en este pipeline normalizado).
- `eps`: beneficio por accion base.
- `roe_trend_2y`: tendencia 2 anos de ROE (pendiente/mejora estructural).
- `roe_trend_3y`: tendencia 3 anos de ROE.
- `net_margin_trend_2y`: tendencia 2 anos de margen neto.
- `net_margin_trend_3y`: tendencia 3 anos de margen neto.
- `gross_margin_trend_3y`: tendencia 3 anos de margen bruto.

### 1.2 Columnas derivadas internas del agente
- No crea nuevas columnas de negocio propias en `_prepare`.
- Anade dinamicamente todas las columnas que terminen en `_zsector` si existen en `X`.

### 1.3 Significado del score del agente
- `fundamental_score` alto: la combinacion de rentabilidad, calidad, crecimiento y solvencia sugiere mayor probabilidad de `Outperform`.

---

## 2) ValuationAgent

### 2.1 Columnas base configuradas (`VALUATION_FEATURE_COLUMNS`)
- `pe_ratio`: precio / beneficio por accion.
- `pb_ratio`: precio / valor en libros por accion.
- `ps_ratio`: precio / ventas por accion.
- `ev_to_ebitda`: valor empresa / EBITDA.
- `fcf_yield`: FCF / market cap (o equivalente en el builder).
- `earnings_yield`: beneficio / precio (inverso de P/E aproximado).
- `pe_vs_5y_median`: posicion de P/E frente a su mediana historica 5y.
- `pb_vs_5y_median`: posicion de P/B frente a mediana historica 5y.
- `ev_ebitda_vs_5y_median`: posicion de EV/EBITDA frente a mediana historica 5y.
- `eps_surprise_pct`: sorpresa de EPS mas reciente (%).
- `eps_revision`: revision reciente de estimaciones EPS.
- `eps_est`: EPS estimado por analistas.
- `eps_reported`: EPS reportado.

### 2.2 Columnas derivadas internas del agente
- No crea columnas derivadas de comparativa sectorial en `_prepare`.
- Opera con las columnas base definidas en `VALUATION_FEATURE_COLUMNS`.

### 2.3 Significado del score del agente
- `valuation_score` alto: la accion aparece atractiva por valoracion absoluta/historica (multiples y yields) junto con apoyo de dinamica EPS.

---

## 3) MomentumAgent

### 3.1 Columnas base configuradas (`MOMENTUM_FEATURE_COLUMNS`)
- `rsi_14`: RSI de 14 periodos (sobrecompra/sobreventa tactica).
- `rsi_28`: RSI de 28 periodos (oscilador mas lento).
- `macd`: componente MACD principal.
- `macd_signal`: linea senal de MACD.
- `macd_hist`: diferencia MACD - signal.
- `sma_20`: posicion/distancia respecto a media movil 20.
- `sma_50`: posicion/distancia respecto a media movil 50.
- `sma_200`: posicion/distancia respecto a media movil 200.
- `bb_pct`: posicion relativa dentro de bandas de Bollinger.
- `price_vs_52w_high`: distancia relativa al maximo 52 semanas.
- `price_vs_52w_low`: distancia relativa al minimo 52 semanas.
- `momentum_1m`: retorno/momentum a 1 mes.
- `momentum_3m`: retorno/momentum a 3 meses.
- `momentum_6m`: retorno/momentum a 6 meses.
- `momentum_12m`: retorno/momentum a 12 meses.
- `volatility_20d`: volatilidad realizada 20 dias.
- `volatility_60d`: volatilidad realizada 60 dias.
- `atr_14`: ATR 14 periodos (rango medio verdadero).
- `vol_ratio_20_50`: ratio de volumen corto vs largo plazo.
- `beat_rate_4q`: proporcion de trimestres con beat de EPS (ultimos 4).
- `eps_surprise_avg_4q`: sorpresa media EPS en ultimos 4 trimestres.
- `eps_revision`: revision reciente de estimaciones EPS.

### 3.2 Columnas derivadas internas del agente
Se crean condicionalmente en `_prepare`:
- `rsi_overbought`: 1 si `rsi_14 > 70`, si no 0.
- `rsi_oversold`: 1 si `rsi_14 < 30`, si no 0.
- `above_sma200`: 1 si condicion de tendencia de largo plazo es alcista segun implementacion (`sma_200 > 0` en este pipeline), si no 0.
- `macd_bullish`: 1 si `macd > macd_signal`.
- `cross_sma_20_50`: 1 si `sma_20 > sma_50`.
- `momentum_quality`: `momentum_12m - momentum_1m` (persistencia vs ruido reciente).
- `vol_expansion`: 1 si `vol_ratio_20_50 > 1.5`.
- `consistent_beater`: 1 si `beat_rate_4q >= 0.75`.
- `earnings_momentum`: suma binaria de:
  - `eps_surprise_avg_4q > 0`
  - `eps_revision > 0`

### 3.3 Significado del score del agente
- `momentum_score` alto: hay senal de continuidad alcista en precio/volumen, reforzada por momentum de beneficios.

---

## 4) BearAgent

### 4.1 Columnas base configuradas (`BEAR_FEATURE_COLUMNS`)
- `total_debt_yoy_growth`: crecimiento YoY de deuda; aceleracion de apalancamiento.
- `debt_equity`: deuda/patrimonio.
- `debt_to_ebitda`: deuda/EBITDA.
- `fcf_margin`: margen de caja libre; negativo sugiere tension.
- `current_ratio`: liquidez de corto plazo.
- `consecutive_losses`: racha de trimestres en perdida.
- `revenue_decline`: bandera de caida YoY de ingresos.
- `interest_coverage`: cobertura de intereses.
- `insider_net_ratio_90d`: balance neto insider ultimos 90 dias.
- `insider_sell_ratio`: proporcion de ventas insider.
- `eps_surprise_pct`: sorpresa EPS reciente.
- `eps_revision`: revision de EPS.

### 4.2 Columnas derivadas internas del agente (flags de riesgo)
Se crean siempre en `_add_flag_cols` (si falta base, se rellena 0.0):
- `debt_growth_high`: 1 si `total_debt_yoy_growth > 0.20`.
- `debt_equity_high`: 1 si `debt_equity > 3.00`.
- `debt_ebitda_high`: 1 si `debt_to_ebitda > 6.00`.
- `fcf_negative`: 1 si `fcf_margin < 0.00`.
- `consecutive_losses` (flag): 1 si `consecutive_losses >= 2.00`.
- `revenue_decline` (flag): 1 si `revenue_decline == 1.00`.
- `low_coverage`: 1 si `interest_coverage < 1.50`.
- `liquidity_risk`: 1 si `current_ratio < 1.00`.
- `insider_selling`: 1 si `insider_sell_ratio > 0.70`.
- `eps_miss`: 1 si `eps_surprise_pct < -5.00`.

### 4.3 Significado del score del agente
- `bear_score` es score de riesgo combinado (reglas + ML) en [0,1].
- Alto `bear_score` = mayor riesgo de deterioro/underperform.
- En el ensamblado final se transforma a seguridad (o se usa su complementario) segun la logica del meta.

---

## 5) SentimentAgent

### 5.1 Columnas base configuradas (`SENTIMENT_FEATURE_COLUMNS`)
- `analyst_buy_ratio`: proporcion de recomendaciones de compra.
- `analyst_bearish_score`: intensidad agregada de recomendaciones bajistas.
- `analyst_consensus`: consenso agregado de analistas.
- `analyst_dispersion`: dispersion/desacuerdo entre analistas.
- `analyst_strong_buy_pct`: porcentaje de recomendaciones strong buy.
- `analyst_consensus_change`: cambio reciente del consenso.
- `mspr_3m`: metrica de sentimiento insider a 3 meses.
- `mspr_trend`: tendencia de `mspr`.
- `insider_net_ratio_90d`: balance neto insider 90 dias.
- `insider_sell_ratio`: proporcion de ventas insider.
- `beat_rate_4q`: tasa de beats EPS (4 trimestres).
- `eps_surprise_avg_4q`: sorpresa media EPS (4 trimestres).
- `eps_surprise_pct`: sorpresa EPS reciente.

### 5.2 Columnas derivadas internas del agente
Se crean condicionalmente en `_prepare`:
- `analyst_net_bullish`: `analyst_buy_ratio - analyst_bearish_score`.
- `insider_net_zscore`: z-score de `insider_net_ratio_90d` (o fallback historico `insider_net_shares_90d`).
- `mspr_positive`: 1 si `mspr_3m > 20`.
- `mspr_negative`: 1 si `mspr_3m < -20`.
- `consistent_beater`: 1 si `beat_rate_4q >= 0.75`.

### 5.3 Significado del score del agente
- `sentiment_score` alto: consenso analista/insider/EPS alineado en sesgo alcista.

---

## 6) SectorRotationAgent

### 6.1 Columnas base configuradas (`SECTOR_ROTATION_FEATURE_COLUMNS`)
Estas columnas se agregan por sector (media de tickers del sector) para construir la observacion sectorial:
- `roe`: rentabilidad sectorial media sobre equity.
- `roa`: rentabilidad sectorial media sobre activos.
- `net_margin`: margen neto medio del sector.
- `gross_margin`: margen bruto medio del sector.
- `fcf_margin`: margen de caja libre medio del sector.
- `ebitda_margin`: margen EBITDA medio del sector.
- `revenue_yoy_growth`: crecimiento YoY medio de ventas del sector.
- `net_income_yoy_growth`: crecimiento YoY medio de beneficio neto sectorial.
- `eps_yoy_growth`: crecimiento YoY medio de EPS sectorial.
- `debt_to_ebitda`: apalancamiento operativo medio del sector.
- `debt_equity`: apalancamiento financiero medio del sector.
- `interest_coverage`: cobertura media de intereses sectorial.
- `current_ratio`: liquidez media de corto plazo sectorial.
- `pe_ratio`: P/E medio del sector.
- `pb_ratio`: P/B medio del sector.
- `ev_to_ebitda`: EV/EBITDA medio del sector.
- `fcf_yield`: FCF yield medio del sector.
- `earnings_yield`: earnings yield medio del sector.
- `pe_vs_5y_median`: desviacion media de P/E vs historia 5y del sector.
- `pb_vs_5y_median`: desviacion media de P/B vs historia 5y del sector.
- `momentum_1m`: momentum medio 1m del sector.
- `momentum_3m`: momentum medio 3m del sector.
- `momentum_6m`: momentum medio 6m del sector.
- `momentum_12m`: momentum medio 12m del sector.
- `volatility_20d`: volatilidad media 20d del sector.
- `volatility_60d`: volatilidad media 60d del sector.
- `rsi_14`: RSI medio 14 del sector.
- `analyst_buy_ratio`: sesgo comprador medio de analistas del sector.
- `analyst_consensus`: consenso medio de analistas del sector.
- `analyst_dispersion`: dispersion media entre analistas del sector.
- `analyst_bearish_score`: sesgo bajista medio de analistas del sector.
- `insider_net_ratio_90d`: balance neto insider medio del sector (90d).
- `insider_sell_ratio`: proporcion media de ventas insider del sector.
- `beat_rate_4q`: tasa media de beats EPS del sector.
- `eps_surprise_pct`: sorpresa EPS media reciente del sector.
- `eps_surprise_avg_4q`: sorpresa media 4Q del sector.
- `eps_revision`: revision media reciente de EPS en el sector.
- `mspr_3m`: sentimiento insider medio 3m del sector.

### 6.2 Columnas derivadas internas del agente
- No genera nuevas columnas de negocio complejas.
- Su transformacion principal es el agregado sectorial (groupby media).

### 6.3 Significado del score del agente
- `sector_score` alto: mayor probabilidad de que el sector supere al benchmark en el siguiente periodo.

---

## 7) MetaLearner

### 7.1 Columnas base configuradas (`META_FEATURE_COLUMNS`)
- `fundamental_score`: score del FundamentalAgent.
- `valuation_score`: score del ValuationAgent.
- `momentum_score`: score del MomentumAgent.
- `bear_score`: score del BearAgent.
- `sentiment_score`: score del SentimentAgent.
- `sector_score`: score del SectorRotationAgent.

### 7.2 Columnas derivadas internas del meta

#### 7.2.1 Dummies de sector
Si existe columna `sector` y `use_sector_features=True`, se anaden:
- `sector_*`: one-hot del sector (nombres reales dependen de sectores presentes en train).

#### 7.2.2 Rankings intra-sector por score
Si existe `sector`, se anaden:
- `fundamental_score_sector_rank`: percentil del score fundamental dentro del sector.
- `valuation_score_sector_rank`: percentil del score valuation dentro del sector.
- `momentum_score_sector_rank`: percentil del score momentum dentro del sector.
- `sentiment_score_sector_rank`: percentil del score sentiment dentro del sector.

#### 7.2.3 Interacciones entre agentes
- `fund_x_val`: `fundamental_score * valuation_score`.
- `mom_x_safety`: `momentum_score * bear_score`.
- `fund_x_sentiment`: `fundamental_score * sentiment_score`.
- `mom_x_sentiment`: `momentum_score * sentiment_score`.
- `sector_x_fundamental`: `sector_score * fundamental_score`.
- `sector_x_momentum`: `sector_score * momentum_score`.

#### 7.2.4 Features de consenso (si `META_ENABLE_CONSENSUS_FEATURES=True`)
Calculadas sobre scores disponibles en `META_AGENT_SCORE_COLUMNS` (definida en `environment.py`):
- `agent_score_mean`: media de scores de agentes.
- `agent_score_std`: desviacion estandar de scores (desacuerdo).
- `agent_score_max`: maximo score entre agentes.
- `agent_score_min`: minimo score entre agentes.
- `agent_score_range`: rango max-min de scores.
- `bullish_agent_score_count`: numero de agentes con score >= `META_BULLISH_SCORE_THRESHOLD`.
- `consensus_score_strength`: `abs(agent_score_mean - 0.5) * 2`; 0 neutro, 1 consenso fuerte.
- `confidence_weighted_score_mean`: media ponderada por conviccion `abs(score - 0.5)`.

### 7.3 Significado del score del meta
- `final_score` alto: mayor probabilidad final de `Outperform` combinando evidencia multi-agente.
- Adicionalmente aplica reglas de riesgo duro/blando relacionadas con `bear_score`/`bear_risk_score` en inferencia.

---

## 8) Resumen rapido: columnas exactas por agente

- FundamentalAgent:
  - Base exacta: `FUNDAMENTAL_FEATURE_COLUMNS` (36 columnas en estado actual).
  - Derivadas: anade todas las disponibles con sufijo `_zsector`.

- ValuationAgent:
  - Base exacta: `VALUATION_FEATURE_COLUMNS` (13 columnas).
  - Derivadas: no anade columnas de comparativa sectorial internas.

- MomentumAgent:
  - Base exacta: `MOMENTUM_FEATURE_COLUMNS` (23 columnas).
  - Derivadas: `rsi_overbought`, `rsi_oversold`, `above_sma200`, `macd_bullish`, `cross_sma_20_50`, `momentum_quality`, `vol_expansion`, `consistent_beater`, `earnings_momentum`.

- BearAgent:
  - Base exacta: `BEAR_FEATURE_COLUMNS` (12 columnas).
  - Derivadas (flags): `debt_growth_high`, `debt_equity_high`, `debt_ebitda_high`, `fcf_negative`, `consecutive_losses` (flag), `revenue_decline` (flag), `low_coverage`, `liquidity_risk`, `insider_selling`, `eps_miss`.

- SentimentAgent:
  - Base exacta: `SENTIMENT_FEATURE_COLUMNS` (13 columnas).
  - Derivadas: `analyst_net_bullish`, `insider_net_zscore`, `mspr_positive`, `mspr_negative`, `consistent_beater`.

- SectorRotationAgent:
  - Base exacta: `SECTOR_ROTATION_FEATURE_COLUMNS` (40 columnas), agregadas por media a nivel sector.
  - Derivadas: no anade features de negocio adicionales; transforma por agregacion sectorial.

- MetaLearner:
  - Base exacta: `META_FEATURE_COLUMNS` (6 scores).
  - Derivadas: `sector_*`, `*_sector_rank`, interacciones (`fund_x_val`, `mom_x_safety`, `fund_x_sentiment`, `mom_x_sentiment`, `sector_x_fundamental`, `sector_x_momentum`) y consenso (`agent_score_mean`, `agent_score_std`, `agent_score_max`, `agent_score_min`, `agent_score_range`, `bullish_agent_score_count`, `consensus_score_strength`, `confidence_weighted_score_mean`).

---

## 9) Nota sobre uso real en entrenamiento
- Que una columna este en esta guia significa que puede entrar al pipeline del agente.
- La seleccion final por fold puede reducir columnas efectivas (FeatureSelector).
- Regla actual del selector: conserva features con importancia >= 50% de la importancia top y luego acota entre 4 y 10 features por fold.
- Las features finalmente seleccionadas se reponderan para que sus pesos sumen 100 y esos pesos se aplican en train/predict.
- Para validar exactamente que se uso en una corrida concreta, revisar:
  - `results/agents/quarter_*_feature_usage_report.csv`
  - `results/agents/quarter_*_feature_usage_report.json`
