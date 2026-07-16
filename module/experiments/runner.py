"""Ejecuta una batería de escenarios: aísla cada uno, reutiliza el scoring caro cuando puede, y
recoge una fila de métricas por escenario.

Flujo de un experimento:

1. Se garantiza un escenario ``baseline`` (sin overrides) como referencia.
2. Para cada escenario, ``apply_overrides`` fija su configuración y su ``run_dir`` propio bajo
   ``results/escenarios/<exp_id>/<nombre>/``. La comparación (comparison.csv + index.html) queda en
   ``results/escenarios/<exp_id>/``.
3. Cada escenario corre el pipeline completo (como ``main.py``), sin las etapas comunes ya
   preparadas por ``_ensure_experiment_inputs`` (``download``/``dataset``/``features``): en orden
   ``ml`` (``train_and_score``) → ``watchlist`` (``build_watchlist``) → ``backtest``
   (``run_backtest``) → ``viewer`` (``build_viewer``) → ``report`` (``build_final_report``). Todas
   honran ``settings.run_dir``, así que el escenario queda autocontenido con su propio
   ``viewer/index.html`` navegable. De la salida del backtest se extraen las métricas con
   ``collect_metrics``.
4. La etapa cara (``train_and_score``) se ejecuta UNA vez por ``scoring_cache_key`` — el hash de lo
   que afecta al scoring (overrides de ml, campos ML de Settings, tickers). Escenarios que solo
   cambian estrategia comparten scoring y se marcan ``re_scored=False``.

La caché guarda los artefactos que la etapa ml produce en carpetas compartidas —``scored_universe``
y ``meta_weights`` en ``PROCESSED_DIR``, más el ``model_walk_forward_diagnostics.csv`` que va al
``run_dir``— y los restaura antes de cada backtest que reutiliza esa clave.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from environment import MASTER_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, RESULTS_DIR, Settings
from module.backtest import run_backtest
from module.ml import train_and_score

from .metrics import collect_metrics
from .overrides import apply_overrides, split_overrides
from .scenario import Scenario

log = logging.getLogger(__name__)

# Artefactos que la etapa ml deja en carpetas compartidas (PROCESSED_DIR) — hay que preservarlos y
# restaurarlos por clave de caché, porque el siguiente escenario los sobrescribiría.
_SCORING_SHARED_ARTIFACTS = [
    "scored_universe.parquet",
    "meta_weights_by_snapshot.parquet",
    "model_explainability.json",
]
# Artefacto de diagnóstico que ml escribe en el run_dir del escenario y que collect_metrics necesita
# para leer el rank-IC; se cachea aparte y se copia al run_dir de cada escenario que reusa la clave.
_SCORING_RUNDIR_ARTIFACT = "model_walk_forward_diagnostics.csv"

# Campos de Settings que, si cambian, obligan a re-scorear (afectan al entrenamiento ML).
_ML_SETTINGS_FIELDS = [
    "walk_forward_scoring",
    "walk_forward_label_horizon_months",
    "min_walk_forward_training_rows",
    "min_walk_forward_training_years",
    "max_walk_forward_training_years",
    "train_cutoff_date",
    "walk_forward_train_frequency",
    "start_date",
    "end_date",
    "data_start_date",
    "dev_mode",
]


_ALL_SCENARIOS_FILE = "experiments/escenarios_todos.py"

# Fila terminada de un escenario, escrita en su run_dir. Su presencia marca el escenario como
# completado y permite reanudar un barrido sin re-ejecutarlo.
_SCENARIO_ROW_FILE = "_scenario_row.json"


def load_scenarios(spec: str) -> list[Scenario]:
    """Carga la lista SCENARIOS de un fichero de escenarios.

    `spec` es la ruta a un fichero ``experiments/escenarios_*.py`` o el literal ``"todos"``, que
    resuelve al fichero que junta los cuatro bloques. Las rutas relativas se resuelven contra la raíz
    del proyecto, para que funcionen igual desde main.py que desde el CLI.
    """
    rel = _ALL_SCENARIOS_FILE if spec.strip().lower() == "todos" else spec
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de escenarios: {path}")
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"No se pudo cargar el fichero de escenarios: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    scenarios = getattr(module, "SCENARIOS", None)
    if not isinstance(scenarios, list) or not all(isinstance(s, Scenario) for s in scenarios):
        raise ValueError(f"{path} debe definir SCENARIOS: list[Scenario].")
    return scenarios


def scoring_cache_key(scenario: Scenario, base_settings: Settings) -> str:
    """Hash de todo lo que cambia el scoring: overrides de ml, campos ML de Settings (incluidos los
    que el escenario override), y el universo de tickers. Dos escenarios que solo difieren en
    estrategia comparten clave; uno que toca ml/ventana/horizonte/semilla no."""
    settings_ovr, ml_ovr, _ = split_overrides(scenario.overrides)
    effective = dataclasses.replace(base_settings, **settings_ovr) if settings_ovr else base_settings
    payload = {
        "ml": {k: ml_ovr[k] for k in sorted(ml_ovr)},
        "settings": {field: getattr(effective, field) for field in _ML_SETTINGS_FIELDS},
        "tickers": sorted(effective.tickers),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def _save_scoring_to_cache(cache_dir: Path, run_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in _SCORING_SHARED_ARTIFACTS:
        source = PROCESSED_DIR / name
        if source.exists():
            shutil.copy2(source, cache_dir / name)
    diag = run_dir / _SCORING_RUNDIR_ARTIFACT
    if diag.exists():
        shutil.copy2(diag, cache_dir / _SCORING_RUNDIR_ARTIFACT)


@contextmanager
def _preserve_processed_scoring():
    """Guarda y restaura los artefactos de scoring de PROCESSED_DIR alrededor de todo el experimento.

    El runner sobrescribe scored_universe/meta_weights/model_explainability en la carpeta compartida
    data/processed/ escenario tras escenario. Sin esto, al terminar quedaría el scoring del último
    escenario (posiblemente dev o con overrides) en su sitio, y un `python main.py` o etapa backtest
    posterior lo leería como si fuera el scoring por defecto. Restauramos el estado previo al salir.
    """
    backup = Path(tempfile.mkdtemp(prefix="exp_processed_backup_"))
    existed: list[str] = []
    for name in _SCORING_SHARED_ARTIFACTS:
        source = PROCESSED_DIR / name
        if source.exists():
            shutil.copy2(source, backup / name)
            existed.append(name)
    try:
        yield
    finally:
        for name in _SCORING_SHARED_ARTIFACTS:
            target = PROCESSED_DIR / name
            if name in existed:
                shutil.copy2(backup / name, target)
            elif target.exists():
                # No existía antes del experimento: lo creó el runner, lo quitamos para no dejar rastro.
                target.unlink()
        shutil.rmtree(backup, ignore_errors=True)


def _restore_scoring_from_cache(cache_dir: Path, run_dir: Path) -> None:
    for name in _SCORING_SHARED_ARTIFACTS:
        cached = cache_dir / name
        if cached.exists():
            shutil.copy2(cached, PROCESSED_DIR / name)
    diag = cache_dir / _SCORING_RUNDIR_ARTIFACT
    if diag.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(diag, run_dir / _SCORING_RUNDIR_ARTIFACT)


def run_scenario(
    scenario: Scenario,
    base_settings: Settings,
    exp_dir: Path,
    cache: dict[str, Path],
) -> dict:
    """Ejecuta un escenario end-to-end y devuelve su fila de resultados (métricas + trazabilidad)."""
    run_dir = exp_dir / scenario.name
    key = scoring_cache_key(scenario, base_settings)
    cache_dir = exp_dir / "_scoring_cache" / key
    re_scored = key not in cache

    # Imports diferidos: el escenario corre el pipeline completo (como main.py, pero sin
    # download/dataset/features —ya preparados una vez— ni research_ai —eliminado del proyecto—).
    # Se importan aquí para no arrastrar el viewer si solo se usa la maquinaria del runner.
    from module.report import build_final_report
    from module.strategy.selection import build_watchlist
    from module.viewer import build_viewer

    with apply_overrides(scenario, base_settings) as settings:
        settings = dataclasses.replace(settings, run_dir_override=str(run_dir))
        started = time.perf_counter()
        # ml: puntuar el universo (etapa cara, cacheada por clave de scoring).
        if re_scored:
            log.info("[%s] scoring (clave nueva %s)", scenario.name, key)
            train_and_score(settings)
            _save_scoring_to_cache(cache_dir, run_dir)
            cache[key] = cache_dir
        else:
            log.info("[%s] reutiliza scoring cacheado (clave %s)", scenario.name, key)
            _restore_scoring_from_cache(cache_dir, run_dir)
        # Resto del pipeline normal, en el mismo orden que main.py: watchlist → backtest →
        # viewer → report. Cada etapa honra settings.run_dir, así que el escenario queda
        # autocontenido en su carpeta con su propio viewer/index.html navegable.
        build_watchlist(settings)
        outputs = run_backtest(settings)
        metrics = collect_metrics(outputs, run_dir)
        build_viewer(settings)
        build_final_report(settings)
        elapsed = time.perf_counter() - started

    row = {
        "name": scenario.name,
        "why": scenario.why,
        "block": scenario.block,
        "re_scored": re_scored,
        "cache_key": key,
        "run_dir": str(run_dir),
        "overrides": json.dumps(scenario.overrides, default=str, ensure_ascii=False),
        "seconds": round(elapsed, 1),
        **metrics,
    }
    # Persistimos la fila terminada para poder REANUDAR: si un barrido posterior encuentra este
    # fichero, reutiliza el escenario sin re-ejecutarlo (ver run_experiment con resume).
    with open(run_dir / _SCENARIO_ROW_FILE, "w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
    log.info("[%s] hecho en %.1fs (rank_ic=%.4f, alpha=%.4f)", scenario.name, elapsed,
             metrics.get("rank_ic_final_mean", float("nan")), metrics.get("cumulative_alpha", float("nan")))
    return row


def _ensure_baseline(scenarios: list[Scenario]) -> list[Scenario]:
    if any(s.is_baseline for s in scenarios):
        return scenarios
    baseline = Scenario(name="baseline", why="Configuración por defecto del proyecto (referencia).", block="baseline")
    return [baseline, *scenarios]


def _dedup_by_name(scenarios: list[Scenario]) -> list[Scenario]:
    """Quita escenarios con nombre repetido (se queda el primero). Necesario al juntar varios
    ficheros de escenarios: cada uno define su propio 'baseline', y dos escenarios con el mismo
    nombre colisionarían en la misma carpeta run_dir y en la tabla comparativa."""
    seen: set[str] = set()
    unique: list[Scenario] = []
    for scenario in scenarios:
        if scenario.name in seen:
            log.info("Escenario duplicado '%s' ignorado (ya presente).", scenario.name)
            continue
        seen.add(scenario.name)
        unique.append(scenario)
    return unique


def _ensure_experiment_inputs(settings: Settings) -> None:
    """Garantiza las entradas de la etapa `ml` ejecutando las etapas previas que falten.

    La etapa `ml` (`train_and_score`) lee `data/processed/features.parquet` y
    `data/raw/prices.parquet`. Esos datos no dependen de los overrides de ml/estrategia (son iguales
    para todos los escenarios), así que se preparan UNA vez antes del barrido y, una vez en disco, se
    reutilizan en corridas posteriores.

    Reproduce el mismo comportamiento que el pipeline normal, corriendo solo lo que falte:
    `download` → `dataset` → `features`. `download_raw_data` reconstruye `prices.parquet` (y los demás
    crudos) a partir del JSON cacheado en `data/raw/json/` SIN red mientras `FORCE_RAW_DOWNLOAD=False`;
    solo iría a la red si faltara algún JSON, exactamente igual que en una ejecución normal. Así el
    barrido "corre el pipeline" para cada escenario pero sin recalcular lo que es común.
    """
    prices = RAW_DIR / "prices.parquet"
    master = MASTER_DIR / "master_point_in_time.parquet"
    features = PROCESSED_DIR / "features.parquet"

    if not prices.exists():
        log.info("Falta prices.parquet: ejecutando 'download' (reutiliza el JSON cacheado; sin red si está)...")
        from module.ingest.pipeline import download_raw_data

        download_raw_data(settings)
    if not master.exists():
        log.info("Falta master_point_in_time.parquet: ejecutando 'dataset'...")
        from module.dataset import build_master_dataset

        build_master_dataset(settings)
    if not features.exists():
        log.info("Falta features.parquet: ejecutando 'features'...")
        from module.features.pipeline import build_features

        build_features()


def _completed_row(run_dir: Path) -> dict | None:
    """Devuelve la fila persistida de un escenario ya terminado, o None si no está completado."""
    path = run_dir / _SCENARIO_ROW_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _latest_experiment_dir() -> Path | None:
    """La carpeta de experimento fechada más reciente, o None si no hay ninguna."""
    parent = RESULTS_DIR / "escenarios"
    if not parent.exists():
        return None
    dated = [d for d in parent.iterdir() if d.is_dir() and d.name[0].isdigit()]
    return max(dated, key=lambda d: d.name) if dated else None


def run_experiment(
    scenarios: list[Scenario],
    base_settings: Settings | None = None,
    resume: bool = False,
    resume_dir: Path | None = None,
) -> Path:
    """Ejecuta los escenarios en una carpeta fechada y escribe la comparación. Devuelve la ruta del
    experimento (donde quedan comparison.csv e index.html).

    Con ``resume=True`` reutiliza una carpeta existente (``resume_dir`` o la más reciente) y SALTA los
    escenarios ya completados (los que tienen su fila persistida), corriendo solo los que faltan.
    Escribe la comparación final con todos, completados y nuevos.
    """
    base_settings = base_settings or Settings()
    _ensure_experiment_inputs(base_settings)
    scenarios = _dedup_by_name(_ensure_baseline(scenarios))
    # Ordena para que los escenarios que comparten clave de scoring vayan seguidos: así el primero
    # entrena y los siguientes reutilizan la caché sin re-entrenar.
    scenarios = sorted(scenarios, key=lambda s: scoring_cache_key(s, base_settings))

    if resume:
        exp_dir = resume_dir or _latest_experiment_dir()
        if exp_dir is None or not exp_dir.exists():
            raise FileNotFoundError("resume=True pero no hay ninguna carpeta de experimento previa que reanudar.")
        log.info("Reanudando experimento en %s", exp_dir)
    else:
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = RESULTS_DIR / "escenarios" / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
    log.info("Experimento %s: %s escenarios -> %s", exp_dir.name, len(scenarios), exp_dir)

    cache: dict[str, Path] = {}
    if resume:
        # Reaprovecha el scoring cacheado en disco del barrido anterior: cada subcarpeta de
        # _scoring_cache es una clave ya scoreada, así un escenario nuevo que comparte clave con uno
        # ya hecho restaura sus artefactos en vez de re-entrenar.
        cache_root = exp_dir / "_scoring_cache"
        if cache_root.exists():
            cache = {d.name: d for d in cache_root.iterdir() if d.is_dir()}
    rows: list[dict] = []
    with _preserve_processed_scoring():
        for scenario in scenarios:
            done = _completed_row(exp_dir / scenario.name) if resume else None
            if done is not None:
                log.info("[%s] ya completado: se reutiliza sin re-ejecutar.", scenario.name)
                rows.append(done)
                continue
            rows.append(run_scenario(scenario, base_settings, exp_dir, cache))

    # Agregación global: elige el sistema final (config más estable/útil) sobre la era de desarrollo
    # y lo confirma en la reservada. Escribe system_selection.csv/json en la carpeta del experimento.
    from .aggregate import aggregate_scenarios

    verdict = aggregate_scenarios(rows, exp_dir)

    # Import diferido para no arrastrar el viewer si solo se quiere el runner.
    from .report import write_comparison

    write_comparison(rows, exp_dir, selection=verdict)
    log.info("Comparación lista: %s", exp_dir / "index.html")
    return exp_dir
