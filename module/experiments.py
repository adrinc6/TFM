"""Barrido de escenarios: ScenarioSpec, huellas por etapa, ranking, seleccion automatica.

Cada escenario es un pipeline completo (dataset -> features -> agents -> backtest -> report)
con overrides sobre `environment.Settings`. La reutilizacion se decide por huella SHA-256 de
los inputs relevantes de cada etapa: si dos escenarios tienen la misma huella para una etapa,
se enlaza al artefacto compartido en vez de regenerarlo.

Los escenarios se definen en Python (`escenarios/*.py`), no YAML/JSON, para poder incluir
listas, condicionales y calculos derivados.

La seleccion final se hace por rango medio de cuatro dimensiones anuales (beat_rate,
median_alpha, worst_year_alpha, max_drawdown). Menor rango medio -> mas estable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from environment import PROJECT_ROOT, Settings
from module.agents import build_agent_scores
from module.backtest import run_backtest
from module.dataset import build_point_in_time_dataset
from module.features import build_features
from module.report import build_comparison_report, build_run_report
from module.utils import read_parquet, write_json

log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "escenarios"


# -------- ScenarioSpec ------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Un escenario del barrido: nombre + overrides sobre `environment.Settings`.

    Los overrides se pasan por su nombre de campo del dataclass (`target_max`,
    `ridge_alpha`, `snapshot_day`, etc.), no por el nombre de la constante en
    `environment.py`. La razon: es una API tipada que valida al construir Settings.
    """

    name: str
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def apply_to(self, base: Settings) -> Settings:
        return replace(base, **dict(self.overrides))


# -------- Huellas por etapa ------------------------------------------------------


# Que campos de Settings afectan a cada etapa. Un cambio fuera de esta lista NO regenera.
FINGERPRINT_FIELDS: dict[str, tuple[str, ...]] = {
    "dataset": (
        "run_scope", "data_start_date", "end_date", "benchmark_ticker",
        "snapshot_day", "snapshot_step_months",
    ),
    "features": (
        "run_scope", "data_start_date", "end_date", "benchmark_ticker",
        "snapshot_day", "snapshot_step_months", "max_price_age_days",
        "target_horizon_months", "neutralize_by_sector", "neutralize_min_group",
        "fundamental_momentum", "market_regime_feature",
    ),
    "agents": (
        "run_scope", "data_start_date", "end_date", "benchmark_ticker",
        "snapshot_day", "snapshot_step_months", "max_price_age_days",
        "neutralize_by_sector", "neutralize_min_group",
        "fundamental_momentum", "market_regime_feature",
        "target_horizon_months", "execution_year", "execution_quarter",
        "execution_lag_days", "train_lookback_years", "fundamental_step_months",
        "meta_ic_lookback_quarters", "ridge_alpha", "min_training_rows",
        "min_rank_ic_cross_section", "label_transform", "label_winsor_pct",
        "model_type", "objective", "lgbm_n_estimators", "lgbm_max_depth",
        "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type",
    ),
    "backtest": (
        "run_scope", "data_start_date", "end_date", "benchmark_ticker",
        "snapshot_day", "snapshot_step_months", "max_price_age_days",
        "neutralize_by_sector", "neutralize_min_group",
        "fundamental_momentum", "market_regime_feature",
        "target_horizon_months", "execution_year", "execution_quarter",
        "execution_lag_days", "train_lookback_years", "fundamental_step_months",
        "meta_ic_lookback_quarters", "ridge_alpha", "min_training_rows",
        "min_rank_ic_cross_section", "label_transform", "label_winsor_pct",
        "model_type", "objective", "lgbm_n_estimators", "lgbm_max_depth",
        "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type",
        "target_min", "target_max", "entry_min_percentile", "min_hold_percentile",
        "rotation_edge_percentiles", "max_weight_per_position",
        "commission_bps", "slippage_bps", "rebalance_drift_tolerance",
    ),
}


def stage_fingerprint(stage: str, settings: Settings) -> str:
    """SHA-256 de los campos de Settings relevantes para la etapa."""
    if stage not in FINGERPRINT_FIELDS:
        raise ValueError(f"Etapa desconocida: {stage!r}. Conocidas: {list(FINGERPRINT_FIELDS)}")
    values = {name: getattr(settings, name) for name in FINGERPRINT_FIELDS[stage]}
    encoded = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


# -------- Seleccion --------------------------------------------------------------


# Dimensiones de la seleccion. NINGUNA es magnitud de alfa: el sistema se elige por
# aprendizaje (rank-IC) y consistencia, no por rentabilidad. El alfa se reporta aparte.
# Cada tupla: (columna, ascending del rank). ascending=False -> mayor es mejor.
SELECTION_DIMENSIONS = (
    ("mean_rank_ic", False),               # aprendizaje: ordena bien fuera de muestra
    ("rank_ic_positive_fraction", False),  # estabilidad del aprendizaje entre eras
    ("beat_rate", False),                  # consistencia (frecuencia, no magnitud)
    ("max_drawdown", True),                # riesgo: menor es mejor
)


def select_winner(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Ganador por rango medio de las dimensiones de aprendizaje y estabilidad.

    NO usa alfa. La pregunta del TFM es si el sistema APRENDE a ordenar activos fuera de
    muestra de forma estable; elegir por rentabilidad seria seleccionar ruido cuando el
    rank-IC es debil. El alfa se reporta como consecuencia, nunca como criterio.

    Devuelve (nombre_ganador, DataFrame ordenado con los rangos y la composite metric).
    """
    ranked = summary.copy()
    rank_columns: list[str] = []
    for column, ascending in SELECTION_DIMENSIONS:
        rank_name = f"rank_{column}"
        if column not in ranked.columns:
            ranked[column] = 0.0
        ranked[rank_name] = ranked[column].rank(ascending=ascending, method="min").astype(int)
        rank_columns.append(rank_name)
    ranked["composite_rank_mean"] = ranked[rank_columns].mean(axis=1)
    ranked = ranked.sort_values("composite_rank_mean").reset_index(drop=True)
    winner = str(ranked.iloc[0]["scenario"])
    return winner, ranked


# -------- ScenarioRunner ---------------------------------------------------------


def run_scenarios(grid_path: Path, base_settings: Settings) -> Path:
    """Carga la rejilla desde un `.py`, ejecuta cada escenario y produce comparison.html."""
    grid_path = Path(grid_path)
    specs = _load_grid(grid_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Barrido: %s escenarios en %s", len(specs), grid_path)

    shared_artifacts: dict[str, dict[str, Any]] = {"dataset": {}, "features": {}, "agents": {}}
    for spec in specs:
        _run_single_scenario(spec, base_settings, shared_artifacts)

    build_comparison_report(RESULTS_DIR)
    log.info("Barrido completado: %s", RESULTS_DIR / "comparison.html")
    return RESULTS_DIR


def _load_grid(grid_path: Path) -> list[ScenarioSpec]:
    spec = importlib.util.spec_from_file_location("_grid_module", grid_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se puede cargar la rejilla {grid_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenarios = getattr(module, "SCENARIOS", None)
    if not scenarios:
        raise RuntimeError(f"La rejilla {grid_path} no define `SCENARIOS`.")
    return list(scenarios)


def _run_single_scenario(
    spec: ScenarioSpec,
    base_settings: Settings,
    shared_artifacts: dict[str, dict[str, Any]],
) -> None:
    scenario_dir = RESULTS_DIR / spec.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    settings = spec.apply_to(base_settings)

    fingerprints = {stage: stage_fingerprint(stage, settings) for stage in FINGERPRINT_FIELDS}
    reused = {}

    processed = scenario_dir / "processed"
    processed.mkdir(exist_ok=True)

    # Dataset (comparte processed dir si la huella coincide con un escenario ya corrido)
    dataset_key = fingerprints["dataset"]
    if dataset_key in shared_artifacts["dataset"]:
        _link_processed(shared_artifacts["dataset"][dataset_key], processed)
        reused["dataset"] = shared_artifacts["dataset"][dataset_key].name
    else:
        _run_stage_dataset(settings, processed)
        shared_artifacts["dataset"][dataset_key] = processed

    # Features
    features_key = fingerprints["features"]
    if features_key in shared_artifacts["features"] and features_key != dataset_key:
        _link_features(shared_artifacts["features"][features_key], processed)
        reused["features"] = shared_artifacts["features"][features_key].name
    elif features_key not in shared_artifacts["features"]:
        _run_stage_features(settings, processed)
        shared_artifacts["features"][features_key] = processed

    # Agents (agents.py escribe en processed/agents/<run_id>)
    agents_key = fingerprints["agents"]
    agents_root = processed / "agents"
    scenario_agents_link = scenario_dir / "agents"
    if agents_key in shared_artifacts["agents"]:
        source = shared_artifacts["agents"][agents_key]
        agents_root.mkdir(exist_ok=True)
        target = agents_root / source.name
        _link_dir(source, target)
        reused["agents"] = str(source)
    else:
        _run_stage_agents(settings, processed)
        latest = sorted(path for path in agents_root.iterdir() if path.is_dir())[-1]
        shared_artifacts["agents"][agents_key] = latest
    if not scenario_agents_link.exists():
        try:
            scenario_agents_link.symlink_to(agents_root, target_is_directory=True)
        except (OSError, NotImplementedError):
            pass   # no critico: el HTML se abre desde processed/agents igualmente

    # Backtest (siempre por escenario, aunque comparta agents: los parametros de cartera
    # pueden diferir; escribe dentro del run_dir de agents)
    latest_agents_run = sorted(path for path in agents_root.iterdir() if path.is_dir())[-1]
    _run_stage_backtest(settings, processed, latest_agents_run)
    build_run_report(latest_agents_run)
    # copia el HTML al scenario_dir para que el link desde comparison.html funcione
    html_link = scenario_dir / "report.html"
    if not html_link.exists():
        try:
            html_link.symlink_to(latest_agents_run / "report.html")
        except (OSError, NotImplementedError):
            shutil.copy2(latest_agents_run / "report.html", html_link)

    _write_scenario_config(spec, settings, scenario_dir, fingerprints, reused)


def _run_stage_dataset(settings: Settings, processed: Path) -> None:
    original = settings.processed_output_dir
    _patched = replace(settings)  # noqa: F841
    # `build_point_in_time_dataset` escribe en settings.processed_output_dir; para redirigir a
    # `processed` del escenario sin tocar el modulo, redirigimos temporalmente via monkey-patch
    # de directorio: es la forma menos invasiva sin refactor.
    from module import dataset as dataset_module

    class _RedirectedSettings:
        def __init__(self, base, target):
            self._base = base
            self._target = target
        def __getattr__(self, item):
            return getattr(self._base, item)
        @property
        def processed_output_dir(self):
            return self._target
        @property
        def raw_output_dir(self):
            return self._base.raw_output_dir

    build_point_in_time_dataset(_RedirectedSettings(settings, processed))


def _run_stage_features(settings: Settings, processed: Path) -> None:
    class _RedirectedSettings:
        def __init__(self, base, target):
            self._base = base
            self._target = target
        def __getattr__(self, item):
            return getattr(self._base, item)
        @property
        def processed_output_dir(self):
            return self._target
        @property
        def raw_output_dir(self):
            return self._base.raw_output_dir
    build_features(_RedirectedSettings(settings, processed))


def _run_stage_agents(settings: Settings, processed: Path) -> None:
    class _RedirectedSettings:
        def __init__(self, base, target):
            self._base = base
            self._target = target
        def __getattr__(self, item):
            return getattr(self._base, item)
        @property
        def processed_output_dir(self):
            return self._target
        @property
        def raw_output_dir(self):
            return self._base.raw_output_dir
    # build_agent_scores usa settings.processed_output_dir para leer features y
    # settings.processed_output_dir / "agents" para escribir su run_dir. Redirigir
    # processed_output_dir al processed del escenario hace que las dos cosas caigan bien.
    build_agent_scores(_RedirectedSettings(settings, processed))


def _run_stage_backtest(settings: Settings, processed: Path, run_dir: Path) -> None:
    scores = read_parquet(run_dir / "agent_scores.parquet", "agents run_dir")
    asset_prices = read_parquet(processed / "asset_price_point_in_time.parquet", "dataset")
    benchmark = read_parquet(processed / "benchmark_point_in_time.parquet", "dataset")
    diagnostics_path = run_dir / "rank_ic_diagnostics.parquet"
    diagnostics = pd.read_parquet(diagnostics_path) if diagnostics_path.exists() else None
    result = run_backtest(scores, asset_prices, benchmark, settings, diagnostics)

    from module.utils import write_parquet
    write_parquet(result.positions, run_dir / "positions.parquet")
    write_parquet(result.orders, run_dir / "orders.parquet")
    write_parquet(result.equity, run_dir / "equity.parquet")
    write_parquet(result.annual_metrics, run_dir / "annual_metrics.parquet")
    write_json(result.summary, run_dir / "backtest_summary.json")


def _link_processed(source: Path, target: Path) -> None:
    for parquet in source.glob("*.parquet"):
        destination = target / parquet.name
        if not destination.exists():
            try:
                destination.symlink_to(parquet)
            except (OSError, NotImplementedError):
                shutil.copy2(parquet, destination)


def _link_features(source: Path, target: Path) -> None:
    for name in ("features_point_in_time.parquet", "targets_forward_3m.parquet",
                 "baseline_scores.parquet"):
        source_file = source / name
        if source_file.exists():
            destination = target / name
            if not destination.exists():
                try:
                    destination.symlink_to(source_file)
                except (OSError, NotImplementedError):
                    shutil.copy2(source_file, destination)


def _link_dir(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        target.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(source, target)


def _write_scenario_config(
    spec: ScenarioSpec,
    settings: Settings,
    scenario_dir: Path,
    fingerprints: dict[str, str],
    reused: dict[str, str],
) -> None:
    config = {
        "name": spec.name,
        "overrides": dict(spec.overrides),
        "fingerprints": fingerprints,
        "reused": reused,
        "settings": {name: getattr(settings, name) for name in FINGERPRINT_FIELDS["backtest"]},
    }
    (scenario_dir / "scenario_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


# -------- Entry point ------------------------------------------------------------


def run_experiments_from_settings(settings: Settings) -> None:
    """Handler para RUN_MODE=experiments. Usa `escenarios/rejilla_base.py` por defecto."""
    grid_path = PROJECT_ROOT / "escenarios" / "rejilla_base.py"
    if not grid_path.exists():
        raise RuntimeError(f"No hay rejilla en {grid_path}.")
    run_scenarios(grid_path, settings)
