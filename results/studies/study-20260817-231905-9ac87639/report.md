# Informe del Portfolio Study

- Study: `study-20260817-231905-9ac87639`
- Model Study de origen: `study-20260817-094411-568bd37e`
- Combinaciones evaluadas: 640
- Criterio: `information_ratio` sobre la ventana de selección hasta 2024.
- 2025–2026: confirmación fuera de muestra. La rejilla ni siquiera la simuló.

## 1. Cartera ganadora

- `coverage_percentile_floor`: `60.0`
- `max_cash_weight`: `0.1`
- `minimum_holding_period`: `full_horizon`
- `rebalance_drift_tolerance`: `0.4`
- `sizing_mode`: `equal`
- `target_size`: `8`

| Métrica | Selección | Era reservada | Artefacto |
|---|---|---|---|
| Information Ratio | 0.7494 | 0.2172 | `portfolio_winner.json` |
| Exceso geométrico | 5.49 % | 1.08 % | `portfolio_winner.json` |
| CAGR cartera | 19.38 % | 20.47 % | `portfolio_winner.json` |
| CAGR benchmark | 13.17 % | 19.18 % | `portfolio_winner.json` |
| Máxima caída | 21.17 % | 13.93 % | `portfolio_winner.json` |
| Años que baten | 90.00 % | 50.00 % | `portfolio_winner.json` |
| Turnover anualizado | 137.36 % | 113.73 % | `portfolio_winner.json` |

## 2. Qué aporta frente a la cartera del modelo

| Métrica | Cartera del modelo | Cartera optimizada | Diferencia |
|---|---|---|---|
| Turnover anualizado | 2.4196 | 1.3736 | -1.0460 |
| Años que baten | 0.7000 | 0.9000 | 0.2000 |
| Exceso geométrico | 0.0212 | 0.0549 | 0.0337 |
| Information Ratio | 0.3125 | 0.7494 | 0.4369 |
| Máxima caída | 0.2521 | 0.2117 | -0.0403 |

## 3. ¿Aguanta el supuesto de coste?

- Coste adoptado: 30.0 pb por operación.
- Equilibrio de ruta congelada (conservador): 411.9 pb por operación (4.12 %, 823.8 pb ida y vuelta).
- Equilibrio resimulado: 219.3 pb por operación (2.19 %, 438.6 pb ida y vuelta).

El equilibrio se define **contra el índice**: es el coste al que el exceso geométrico se anula. La familia congelada mantiene las decisiones ya tomadas, así que subestima el margen; la resimulada deja que la cartera opere menos al encarecerse. Artefacto: `cost_sensitivity.json`, con sus salvedades dentro.

## 4. ¿Hasta qué patrimonio es ejecutable?

- Con órdenes por debajo del 10% del volumen diario habitual: hasta 41567322 USD.
- Con órdenes por debajo del 5% del volumen diario habitual: hasta 20783661 USD.
- Cobertura de volumen de las órdenes: 100.00 %.

Se mide participación sobre el volumen habitual, no impacto de mercado. Artefacto: `capacity.json`.

## 5. Qué hizo la cartera

- Acciones distintas que llegó a tener: 50.
- Posiciones simultáneas (media): 8.0.
- Permanencia mediana de una posición: 13.0 meses.
- Episodios cerrados: 57.

Los nombres más presentes, las mayores y menores contribuciones, las mejores y peores operaciones cerradas y las ventas que luego subieron están en `portfolio_narrative.json` y `portfolio_narrative_holdings.parquet`. El mapa de posiciones se lee de `evidence_best_full/positions.parquet`.

## Interpretación

Los tres diagnósticos de este informe —costes, capacidad y narrativa— se calculan con la cartera **ya elegida** y no intervinieron en la elección. En particular, el coste nunca selecciona: optimizarlo sería escoger el mundo en el que la estrategia luce mejor.
