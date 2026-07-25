# Model Study · TFM

Sistema point-in-time para estudiar si cinco agentes especializados aprenden a ordenar acciones
fuera de muestra. El proyecto tiene una sola operación científica: `Model Study`.

## Funcionamiento

Un Study recorre secuencialmente las variables predictivas elegidas desde un catálogo cerrado.
Cada comparación mantiene fijo el ganador acumulado y modifica una sola variable. El ganador se
elige exclusivamente por Rank-IC robusto entre 2015 y 2024. Después se calculan robustez, reglas
de cartera y ocho perfiles como evidencia informativa.

```text
Temporal → Representación → Modelo → Meta → Ganador predictivo
                                           ├─ Robustez
                                           ├─ Carteras informativas
                                           ├─ Ocho perfiles
                                           └─ Informe
```

2025–2026 se muestra como estrés histórico conocido y nunca interviene en la selección.

## Dashboard

Instalar dependencias y arrancar:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Abrir `http://127.0.0.1:8765/`.

- **Inicio:** marca uno o varios valores cerrados por variable. Un valor significa configuración
  fija; dos o más valores significan optimización predictiva o comparación informativa.
- **Resultados:** primero muestra la tabla de Studies. Cada Study abre una página propia con
  cabecera y accesos a runs, consola, robustez y perfiles; cada run abre después su propia página
  con el detalle disponible y, si es el ganador, toda la evidencia analítica.
- **Modo visual:** `http://127.0.0.1:8765/dev` utiliza fixtures y no entrena.

Detener con `Ctrl+C`. El servidor termina los workers que haya creado; no deben quedar procesos
Python huérfanos.

## Datos

- `data/raw/`: fuentes descargadas y muestra dev.
- `data/prepared/<hash>/`: materialización PIT compartida e inmutable.
- `data/cache/`: fits y resúmenes content-addressable.
- `results/studies/<study_id>/`: configuración, runs, eventos y evidencia del Study.

Los datasets no se copian dentro de resultados.

## Catálogo y recomendación

Todas las opciones proceden de `module/studies/catalog.py`. La recomendación inicial compara
horizonte, lookback, lag 45/60, recencia, preset de información, momentum fundamental y las tres
combinaciones meta. No existe límite artificial de runs: el dashboard informa de coste, tiempo y
disco antes de lanzar.

## Validación

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

La metodología completa está en [docs/metodologia.md](docs/metodologia.md), las decisiones en
[docs/bitacora.md](docs/bitacora.md) y los resultados trazables en
[docs/informe_resultados.md](docs/informe_resultados.md).
