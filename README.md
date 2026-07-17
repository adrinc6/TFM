# TFM — Sistema multi-agente de IA aplicado a la selección de acciones

Trabajo Fin de Máster sobre **Inteligencia Artificial y Machine Learning**. El
núcleo del proyecto es estudiar cómo aprende un sistema de IA y evaluar ese
aprendizaje con rigor; la bolsa (intentar batir al S&P 500 con una estrategia
GARP / momentum) es el **banco de pruebas**, no el objetivo final.

> El proyecto está en un **reinicio en limpio**. Ahora mismo el repositorio solo
> contiene la **descarga de datos**. Todo el diseño (estrategia, agentes,
> backtest, experimentos, informes) y el plan de reconstrucción están descritos
> en detalle en [`docs/doc.md`](docs/doc.md). Las reglas de trabajo para
> contribuir están en [`CLAUDE.md`](CLAUDE.md).

## Requisitos

- Python 3.10+
- Instalar dependencias:

  ```bash
  pip install -r requirements.txt
  ```

## Configuración

Crea un archivo `.env` en la raíz con tu clave de Finnhub (los precios de Yahoo
Finance no necesitan clave):

```
FINNHUB_API_KEY=tu_clave
```

El resto de parámetros de la descarga (rango de fechas, universo de tickers,
modo desarrollo) son editables en [`environment.py`](environment.py).

## Uso

```bash
python main.py
```

Descarga, valida y almacena los datos crudos de todos los tickers configurados.

## Qué produce

- Cache de respuestas JSON por ticker en `data/raw/json/<fuente>/<ticker>/`.
- Parquet agregados en `data/raw/`: `profiles.parquet`, `finnhub_metrics.parquet`,
  `prices.parquet`, `news.parquet`, `report_dates.parquet`.
- Metadatos de la ejecución: `download_coverage.json`, `download_failures.csv`.

Los datos descargados no se versionan (ver `.gitignore`).

## Documentación

- [`docs/doc.md`](docs/doc.md) — plan maestro: propósito, metodología, estrategia
  y arquitectura objetivo por etapas.
- [`CLAUDE.md`](CLAUDE.md) — reglas de estilo, prioridades metodológicas y
  cambios que requieren aprobación.
