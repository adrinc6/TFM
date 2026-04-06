# TFM - Multi-Agent ML Stock Picker

## Supuestos de Backtest y Baselines

- Shares fraccionales: permitidas siempre (float), sin redondeo a enteros.
- Fee por transaccion: 1 USD fijo por operacion y por ticker, tanto en BUY como en SELL.
- Slippage: se modela con `SLIPPAGE_PCT` y por defecto es 0.0.
- Reglas de entrada/salida sin precio exacto:
  - Si no existe precio en la fecha solicitada, se usa el primer precio disponible `>= fecha_solicitada`.
- Equity diaria:
  - Se calcula `equity = cash + valor_mark_to_market`.
  - Si falta precio en un dia, se forward-fill con el ultimo precio conocido y se contabiliza `n_ffill_days`.
- Tickers elegibles por fold:
  - Ticker presente/valido en dataset del fold.
  - Ticker con cobertura de precios para poder resolver entry/exit del fold.
- Baseline `value_combined` (fija, no parametrizable):
  - Usa simultaneamente P/E y EV/EBITDA.
  - Limpieza: nulos fuera, valores `<= 0` fuera, outliers fuera (`P/E > 300`, `EV/EBITDA > 200`).
  - Ranking combinado: promedio de ranks ascendentes de ambos multiplos.
  - Seleccion: Top-N por menor rank combinado con pesos equal-weight.

## Artefactos del run

Todos los artefactos del pipeline se exportan en `results/` y subcarpetas (`backtest/`, `plots/`, etc.).
