# Informe del Portfolio Study

- Study: `study-20260817-212856-f86ca822`
- Model Study de origen: `study-20260817-094411-568bd37e`
- Combinaciones evaluadas: 1440
- Criterio: `information_ratio` sobre la ventana de selección hasta 2024.
- 2025–2026: confirmación fuera de muestra. La rejilla ni siquiera la simuló.

## 1. Cartera ganadora

- `coverage_percentile_floor`: `0.0`
- `max_cash_weight`: `0.1`
- `minimum_holding_period`: `full_horizon`
- `rebalance_drift_tolerance`: `0.4`
- `sizing_mode`: `alpha_proportional`
- `target_size`: `8`

| Métrica | Selección | Era reservada | Artefacto |
|---|---|---|---|
| Information Ratio | 0.8405 | 0.1086 | `portfolio_winner.json` |
| Exceso geométrico | 5.91 % | 0.24 % | `portfolio_winner.json` |
| CAGR cartera | 19.86 % | 19.46 % | `portfolio_winner.json` |
| CAGR benchmark | 13.17 % | 19.18 % | `portfolio_winner.json` |
| Máxima caída | 22.83 % | 16.08 % | `portfolio_winner.json` |
| Años que baten | 100.00 % | 50.00 % | `portfolio_winner.json` |
| Turnover anualizado | 134.53 % | 35.01 % | `portfolio_winner.json` |

## 2. Qué aporta frente a la cartera del modelo

| Métrica | Cartera del modelo | Cartera optimizada | Diferencia |
|---|---|---|---|
| Turnover anualizado | 2.4196 | 1.3453 | -1.0743 |
| Años que baten | 0.7000 | 1.0000 | 0.3000 |
| Exceso geométrico | 0.0212 | 0.0591 | 0.0379 |
| Information Ratio | 0.3125 | 0.8405 | 0.5281 |
| Máxima caída | 0.2521 | 0.2283 | -0.0238 |

## 3. ¿Aguanta el supuesto de coste?

- Coste adoptado: 30.0 pb por operación.
- Equilibrio de ruta congelada (conservador): 446.9 pb por operación (4.47 %, 893.7 pb ida y vuelta).
- Equilibrio resimulado: 294.8 pb por operación (2.95 %, 589.6 pb ida y vuelta).

El equilibrio se define **contra el índice**: es el coste al que el exceso geométrico se anula. La familia congelada mantiene las decisiones ya tomadas, así que subestima el margen; la resimulada deja que la cartera opere menos al encarecerse. Artefacto: `cost_sensitivity.json`, con sus salvedades dentro.

## 4. ¿Hasta qué patrimonio es ejecutable?

- Con órdenes por debajo del 10% del volumen diario habitual: hasta 40579145 USD.
- Con órdenes por debajo del 5% del volumen diario habitual: hasta 20289572 USD.
- Cobertura de volumen de las órdenes: 100.00 %.

Se mide participación sobre el volumen habitual, no impacto de mercado. Artefacto: `capacity.json`.

## 5. Qué hizo la cartera

- Acciones distintas que llegó a tener: 42.
- Posiciones simultáneas (media): 8.0.
- Permanencia mediana de una posición: 15.0 meses.
- Episodios cerrados: 49.

Los nombres más presentes, las mayores y menores contribuciones, las mejores y peores operaciones cerradas y las ventas que luego subieron están en `portfolio_narrative.json` y `portfolio_narrative_holdings.parquet`. El mapa de posiciones se lee de `evidence_best_full/positions.parquet`.

## Interpretación

Los tres diagnósticos de este informe —costes, capacidad y narrativa— se calculan con la cartera **ya elegida** y no intervinieron en la elección. En particular, el coste nunca selecciona: optimizarlo sería escoger el mundo en el que la estrategia luce mejor.
