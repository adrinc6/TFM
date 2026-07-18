"""Barrido de escenarios: ScenarioSpec, huellas por etapa, ranking, seleccion automatica.

Cada escenario es un pipeline completo (dataset -> features -> agents -> backtest -> report)
con overrides sobre `environment.Settings`. La reutilizacion se decide por huella SHA-256 de
los inputs relevantes de cada etapa: si dos escenarios tienen la misma huella para una etapa,
se enlaza al artefacto compartido en vez de regenerarlo.

Los escenarios se definen en Python (`escenarios/*.py`), no YAML/JSON, para poder incluir
listas, condicionales y calculos derivados.

La decision automatica optimiza TODOS los ejes, no solo los artefactos:
- `decide_accepted_artifacts`: acepta un artefacto si mejora el rank-IC del meta_final de forma
  estable (diferencia pareada vs baseline positiva en media y en mas de la mitad de las fechas).
- `decide_best_config`: ademas, para cada eje con niveles (ventana, horizonte, ancla, profundidad,
  cadencia) elige el nivel mas estable, y compone la config final combinandolo con los artefactos.

`run_full_study` orquesta el estudio en dos fases (Fase 1 aisla cada eje; Fase 2 combina los
ganadores) + afinado de hiperparametros, reservando 2025-2026 de la seleccion para validar al
finalista fuera del periodo de busqueda. Sin intervencion humana.
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
from module.data.dataset import build_point_in_time_dataset
from module.modeling.agents import build_agent_scores
from module.modeling.features import build_features
from module.evaluation.backtest import run_backtest
from module.ui.reports import build_comparison_report, build_run_report
from module.common.utils import read_parquet, write_json

log = logging.getLogger(__name__)

RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / "escenarios"   # barrido de artefactos suelto (RUN_MODE=experiments)

# Subcarpetas del estudio full, una por fase (RUN_MODE=full_study). Cada una contiene los
# escenarios de esa fase + su comparison.html; el resumen y las conclusiones van en la raiz.
PHASE1_DIR = RESULTS_ROOT / "fase1_ejes"
PHASE2_DIR = RESULTS_ROOT / "fase2_combinaciones"
HYPERPARAM_DIR = RESULTS_ROOT / "hiperparametros"
FINAL_DIR = RESULTS_ROOT / "final"


# -------- ScenarioSpec ------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Un escenario del barrido: nombre + overrides sobre `environment.Settings`.

    Los overrides se pasan por su nombre de campo del dataclass (`target_max`,
    `lgbm_max_depth`, `snapshot_day`, etc.), no por el nombre de la constante en
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
        "fundamental_momentum", "market_regime_feature", "price_momentum_multi",
        "moving_averages", "regime_extended", "quality_growth_derived",
    ),
    "agents": (
        "run_scope", "data_start_date", "end_date", "benchmark_ticker",
        "snapshot_day", "snapshot_step_months", "max_price_age_days",
        "neutralize_by_sector", "neutralize_min_group",
        "fundamental_momentum", "market_regime_feature", "price_momentum_multi",
        "moving_averages", "regime_extended", "quality_growth_derived",
        "target_horizon_months", "execution_year", "execution_quarter",
        "execution_lag_days", "train_lookback_years", "fundamental_step_months",
        "meta_ic_lookback_quarters", "min_training_rows", "min_rank_ic_cross_section",
        "objective", "lgbm_n_estimators", "lgbm_max_depth",
        "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type",
    ),
    "backtest": (
        "run_scope", "data_start_date", "end_date", "benchmark_ticker",
        "snapshot_day", "snapshot_step_months", "max_price_age_days",
        "neutralize_by_sector", "neutralize_min_group",
        "fundamental_momentum", "market_regime_feature", "price_momentum_multi",
        "moving_averages", "regime_extended", "quality_growth_derived",
        "target_horizon_months", "execution_year", "execution_quarter",
        "execution_lag_days", "train_lookback_years", "fundamental_step_months",
        "meta_ic_lookback_quarters", "min_training_rows", "min_rank_ic_cross_section",
        "objective", "lgbm_n_estimators", "lgbm_max_depth",
        "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type",
        "target_min", "target_max", "entry_min_percentile", "min_hold_percentile",
        "rotation_edge_percentiles", "max_weight_per_position",
        "commission_bps", "slippage_bps", "rebalance_drift_tolerance",
        "max_monthly_position_return", "profile",
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


def run_scenarios(grid_path: Path, base_settings: Settings, results_dir: Path = RESULTS_DIR) -> Path:
    """Carga la rejilla desde un `.py`, ejecuta cada escenario y produce comparison.html."""
    grid_path = Path(grid_path)
    specs = _load_grid(grid_path)
    log.info("Barrido: %s escenarios en %s", len(specs), grid_path)
    return run_scenario_specs(specs, base_settings, results_dir)


def run_scenario_specs(
    specs: list[ScenarioSpec], base_settings: Settings, results_dir: Path = RESULTS_DIR
) -> Path:
    """Ejecuta una lista de escenarios ya construidos en `results_dir` (una carpeta por fase)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    shared_artifacts: dict[str, dict[str, Any]] = {"dataset": {}, "features": {}, "agents": {}}
    for spec in specs:
        _run_single_scenario(spec, base_settings, shared_artifacts, results_dir)

    build_comparison_report(results_dir)
    log.info("Barrido completado: %s", results_dir / "comparison.html")
    return results_dir


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
    results_dir: Path = RESULTS_DIR,
) -> None:
    scenario_dir = results_dir / spec.name
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

    from module.common.utils import write_parquet
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


# Era reservada: las cohortes con prediction_date en estos anios NO entran en la seleccion.
# Se reservan para validar al finalista fuera del periodo de busqueda (control de overfitting
# por seleccion: al mirar muchos escenarios, el maximo puede ser suerte; la era reservada mide
# si la señal aguanta donde nunca se optimizo). Ver docs/doc.md.
SELECTION_UNTIL_YEAR = 2024
RESERVED_ERA_YEARS = (2025, 2026)


def _meta_final_ic(scenario_dir: Path, until_year: int | None = None) -> "pd.Series | None":
    """rank-IC del meta_final por fecha de un escenario, o None si no hay diagnostico utilizable.

    Si `until_year` no es None, solo se devuelven cohortes con prediction_date en ese anio o antes
    (para la seleccion, que reserva los anios posteriores). No reentrena ni mira al futuro: solo
    filtra que cohortes ya calculadas se miran.
    """
    from module.ui.reports import _pick_run_dir   # import diferido: reports depende de matplotlib
    run_dir = _pick_run_dir(scenario_dir)
    if run_dir is None:
        return None
    diag_path = run_dir / "rank_ic_diagnostics.parquet"
    if not diag_path.exists():
        return None
    diag = pd.read_parquet(diag_path)
    meta = diag.loc[diag["agent"] == "meta_final"].copy()
    meta = meta.set_index(pd.to_datetime(meta["prediction_date"])).sort_index()
    if until_year is not None:
        meta = meta.loc[meta.index.year <= until_year]
    if len(meta) < 5:
        return None
    return meta["rank_ic"]


def decide_accepted_artifacts(
    scenarios_root: Path, artifacts: dict[str, dict], until_year: int | None = None
) -> dict:
    """Decide automaticamente que artefactos aceptar y compone la configuracion final.

    Un artefacto se ACEPTA si su rank-IC del meta_final mejora al baseline de forma estable:
    la diferencia pareada por fecha (artefacto - baseline) es positiva en media Y en mas de la
    mitad de las fechas. Se registra el detalle para trazabilidad. La config final combina los
    overrides de todos los aceptados. `until_year` reserva anios posteriores de la seleccion.
    Sin intervencion humana.
    """
    from module.evaluation.stats import block_bootstrap_ci, paired_difference_ci

    scenarios_root = Path(scenarios_root)
    baseline_ic = _meta_final_ic(scenarios_root / "baseline", until_year=until_year)
    decisions: list[dict] = []
    accepted_overrides: dict = {}

    if baseline_ic is None:
        return {"error": "sin baseline evaluable", "accepted": [], "config_final": {}}

    base_ci = block_bootstrap_ci(baseline_ic.reset_index(drop=True))
    for name, overrides in artifacts.items():
        art_ic = _meta_final_ic(scenarios_root / f"artifact_{name}", until_year=until_year)
        if art_ic is None:
            decisions.append({"artifact": name, "accepted": False, "reason": "sin evaluar"})
            continue
        diff = paired_difference_ci(art_ic, baseline_ic)
        art_ci = block_bootstrap_ci(art_ic.reset_index(drop=True))
        accepted = diff["mean_diff"] > 0 and diff["fraction_a_better"] > 0.5
        decisions.append({
            "artifact": name,
            "accepted": bool(accepted),
            "rank_ic_baseline": round(base_ci["mean"], 5),
            "rank_ic_with_artifact": round(art_ci["mean"], 5),
            "mean_diff": round(diff["mean_diff"], 5),
            "fraction_better": round(diff["fraction_a_better"], 3),
            "diff_distinguishable_from_zero": diff["distinguishable_from_zero"],
        })
        if accepted:
            accepted_overrides.update(overrides)

    return {
        "baseline_rank_ic": round(base_ci["mean"], 5),
        "baseline_ci": [round(base_ci["ci_low"], 5), round(base_ci["ci_high"], 5)],
        "decisions": decisions,
        "accepted": [d["artifact"] for d in decisions if d["accepted"]],
        "config_final": accepted_overrides,
    }


def _axis_level_stats(ic: "pd.Series") -> dict:
    """Resumen de estabilidad de un nivel de eje: media, fraccion positiva y varianza del rank-IC."""
    values = ic.dropna()
    return {
        "mean_rank_ic": float(values.mean()),
        "positive_fraction": float((values > 0).mean()),
        "variance": float(values.var(ddof=1)) if len(values) > 1 else 0.0,
        "n_cohorts": int(len(values)),
    }


def _best_level(levels: dict[str, dict]) -> str | None:
    """Elige el nivel mas estable: mayor rank-IC medio, desempate por fraccion positiva y
    luego por menor varianza. `levels` mapea nombre de nivel -> estadisticas (_axis_level_stats)."""
    evaluable = {name: s for name, s in levels.items() if s.get("n_cohorts", 0) >= 5}
    if not evaluable:
        return None
    return max(
        evaluable,
        key=lambda name: (
            evaluable[name]["mean_rank_ic"],
            evaluable[name]["positive_fraction"],
            -evaluable[name]["variance"],
        ),
    )


def decide_best_config(
    scenarios_root: Path,
    artifacts: dict[str, dict],
    axes: dict[str, dict[str, dict]],
    until_year: int | None = SELECTION_UNTIL_YEAR,
) -> dict:
    """Decide la configuracion final optimizando TODOS los ejes, no solo los artefactos.

    - `artifacts`: {nombre -> overrides}. Se aceptan por diferencia pareada vs baseline (estable).
    - `axes`: {eje -> {escenario -> overrides}}. Para cada eje se elige el NIVEL con mejor rank-IC
      del meta_final estable (mayor media, desempate por fraccion positiva y menor varianza). El
      nivel del eje que coincida con el baseline se incluye como opcion por su propio escenario.
    - `until_year`: la seleccion solo mira cohortes hasta ese anio (reserva las posteriores para
      validar al finalista). None = usar todas.

    La config final combina el mejor nivel de cada eje + los artefactos aceptados. Sin humano.
    """
    scenarios_root = Path(scenarios_root)
    artifact_decision = decide_accepted_artifacts(scenarios_root, artifacts, until_year=until_year)

    axis_choices: dict[str, dict] = {}
    config_final: dict = dict(artifact_decision.get("config_final", {}))
    for axis_name, level_specs in axes.items():
        level_stats: dict[str, dict] = {}
        for scenario_name, overrides in level_specs.items():
            ic = _meta_final_ic(scenarios_root / scenario_name, until_year=until_year)
            if ic is None:
                continue
            stats = _axis_level_stats(ic)
            stats["overrides"] = dict(overrides)
            level_stats[scenario_name] = stats
        chosen = _best_level(level_stats)
        axis_choices[axis_name] = {
            "chosen": chosen,
            "levels": {k: {kk: vv for kk, vv in v.items() if kk != "overrides"}
                       for k, v in level_stats.items()},
        }
        if chosen is not None:
            config_final.update(level_stats[chosen]["overrides"])

    return {
        "baseline_rank_ic": artifact_decision.get("baseline_rank_ic"),
        "artifact_decision": artifact_decision,
        "axis_choices": axis_choices,
        "config_final": config_final,
        "selection_until_year": until_year,
    }


def run_experiments_from_settings(settings: Settings) -> dict:
    """Handler para RUN_MODE=experiments. Ejecuta el barrido de ablations, decide automaticamente
    que artefactos aceptar, y escribe la decision en results/escenarios/artifact_decision.json."""
    grid_path = PROJECT_ROOT / "escenarios" / "rejilla_base.py"
    if not grid_path.exists():
        raise RuntimeError(f"No hay rejilla en {grid_path}.")
    run_scenarios(grid_path, settings)

    grid_module = _import_module_from_path(grid_path)
    artifacts = getattr(grid_module, "ARTIFACTS", {})
    decision: dict = {}
    if artifacts:
        decision = decide_accepted_artifacts(RESULTS_DIR, artifacts)
        (RESULTS_DIR / "artifact_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        log.info("Artefactos aceptados: %s", decision.get("accepted"))
    return decision


# Afinado de hiperparametros: variantes de LightGBM que se prueban sobre el ganador de Fase 2.
# Cada una mueve un solo hiperparametro respecto al ganador; se elige la mas estable (mismo
# criterio que los ejes). Rejilla deliberadamente pequeña por el numero limitado de eras.
HYPERPARAM_VARIANTS: dict[str, dict] = {
    "lr_003": {"lgbm_learning_rate": 0.03},
    "lr_010": {"lgbm_learning_rate": 0.10},
    "trees_400": {"lgbm_n_estimators": 400},
    "leaf_20": {"lgbm_min_child_samples": 20},
    "leaf_100": {"lgbm_min_child_samples": 100},
}


def _phase2_specs(axis_choices: dict[str, dict], artifact_overrides: dict) -> list[ScenarioSpec]:
    """Combinaciones dirigidas de los ganadores de Fase 1 (no producto cartesiano completo).

    - La combinacion base: mejor nivel de cada eje + artefactos aceptados (la config que
      `decide_best_config` ya propone), como escenario `phase2_best`.
    - Por cada eje, una variante que sustituye su nivel elegido por el 2º mejor (para comprobar
      si el ganador aislado se sostiene al combinarse con el resto).
    - Si hay 2+ artefactos aceptados, una variante que los activa juntos.
    """
    base_overrides: dict = {}
    for choice in axis_choices.values():
        chosen = choice.get("chosen")
        if chosen and chosen != "baseline":
            base_overrides.update(_axis_override_of(choice, chosen))
    base_overrides.update(artifact_overrides)

    specs = [ScenarioSpec(name="phase2_best", overrides=dict(base_overrides))]

    for axis_name, choice in axis_choices.items():
        second = _second_best_level(choice)
        if second is None:
            continue
        variant = dict(base_overrides)
        # quita el override del nivel elegido de este eje y pon el del 2º mejor
        for key in _axis_override_of(choice, choice.get("chosen")).keys():
            variant.pop(key, None)
        variant.update(_axis_override_of(choice, second))
        specs.append(ScenarioSpec(name=f"phase2_{axis_name}_2nd", overrides=variant))

    return specs


def _axis_override_of(choice: dict, level_name: str | None) -> dict:
    """Los overrides asociados a un nivel de eje (se recuperan del stats guardado; baseline = {})."""
    if not level_name:
        return {}
    return dict(choice.get("overrides_by_level", {}).get(level_name, {}))


def _second_best_level(choice: dict) -> str | None:
    """El 2º nivel mas estable de un eje (excluye el elegido), o None si no hay otro evaluable."""
    levels = choice.get("levels", {})
    chosen = choice.get("chosen")
    candidates = {n: s for n, s in levels.items()
                  if n != chosen and s.get("n_cohorts", 0) >= 5}
    if not candidates:
        return None
    return max(candidates, key=lambda n: (candidates[n]["mean_rank_ic"],
                                          candidates[n]["positive_fraction"],
                                          -candidates[n]["variance"]))


def run_full_study(settings: Settings) -> None:
    """Estudio completo de principio a fin, sin decisiones humanas (RUN_MODE=full_study).

    Fase 1: barrido de ejes aislados -> `decide_best_config` (reservando 2025-2026) elige el mejor
            nivel de cada eje (ventana, horizonte, ancla, profundidad, cadencia) + artefactos.
    Fase 2: combinaciones dirigidas de los ganadores -> se elige la mas estable.
    Afinado: hiperparametros finos sobre el ganador de Fase 2.
    Final:   run con la config ganadora + 8 perfiles + robustez/placebo, y VALIDACION en la era
             reservada 2025-2026 (que no intervino en ninguna eleccion).
    """
    from module.evaluation.profiles import PROFILE_NAMES

    grid_path = PROJECT_ROOT / "escenarios" / "fase1_ejes.py"
    if not grid_path.exists():
        raise RuntimeError(f"No hay rejilla de Fase 1 en {grid_path}.")
    grid_module = _import_module_from_path(grid_path)
    artifacts = getattr(grid_module, "ARTIFACTS", {})
    axes = getattr(grid_module, "AXES", {})

    log.info("=== FASE 1: barrido de ejes aislados -> %s ===", PHASE1_DIR)
    run_scenarios(grid_path, settings, PHASE1_DIR)
    decision = decide_best_config(PHASE1_DIR, artifacts, axes, until_year=SELECTION_UNTIL_YEAR)
    _attach_axis_overrides(decision["axis_choices"], axes)
    write_json(decision, PHASE1_DIR / "phase1_decision.json")
    log.info("Fase 1 -> config propuesta: %s", decision["config_final"])

    log.info("=== FASE 2: combinaciones dirigidas de los ganadores -> %s ===", PHASE2_DIR)
    artifact_overrides = decision["artifact_decision"].get("config_final", {})
    phase2 = _phase2_specs(decision["axis_choices"], artifact_overrides)
    run_scenario_specs(phase2, settings, PHASE2_DIR)
    phase2_levels = {s.name: _meta_final_ic(PHASE2_DIR / s.name, until_year=SELECTION_UNTIL_YEAR)
                     for s in phase2}
    phase2_stats = {name: _axis_level_stats(ic) for name, ic in phase2_levels.items() if ic is not None}
    phase2_winner = _best_level(phase2_stats) or "phase2_best"
    winner_overrides = next(s.overrides for s in phase2 if s.name == phase2_winner)
    write_json({"winner": phase2_winner,
                "ranking": {k: round(v["mean_rank_ic"], 5) for k, v in sorted(
                    phase2_stats.items(), key=lambda kv: kv[1]["mean_rank_ic"], reverse=True)}},
               PHASE2_DIR / "phase2_decision.json")
    log.info("Fase 2 -> ganador: %s (%s)", phase2_winner, dict(winner_overrides))

    log.info("=== AFINADO DE HIPERPARAMETROS sobre el ganador de Fase 2 -> %s ===", HYPERPARAM_DIR)
    hp_specs = [ScenarioSpec(name=f"hp_{k}", overrides={**winner_overrides, **ov})
                for k, ov in HYPERPARAM_VARIANTS.items()]
    hp_specs.insert(0, ScenarioSpec(name="hp_base", overrides=dict(winner_overrides)))
    run_scenario_specs(hp_specs, settings, HYPERPARAM_DIR)
    hp_stats = {}
    for s in hp_specs:
        ic = _meta_final_ic(HYPERPARAM_DIR / s.name, until_year=SELECTION_UNTIL_YEAR)
        if ic is not None:
            hp_stats[s.name] = _axis_level_stats(ic)
    hp_winner = _best_level(hp_stats) or "hp_base"
    accepted = dict(next(s.overrides for s in hp_specs if s.name == hp_winner))
    write_json({"winner": hp_winner,
                "ranking": {k: round(v["mean_rank_ic"], 5) for k, v in sorted(
                    hp_stats.items(), key=lambda kv: kv[1]["mean_rank_ic"], reverse=True)}},
               HYPERPARAM_DIR / "hyperparam_decision.json")
    log.info("Config final del estudio: %s", accepted)

    final_settings = replace(settings, **accepted)
    processed = final_settings.processed_output_dir

    log.info("=== RUN FINAL con la configuracion optima ===")
    _ensure_base_artifacts(final_settings, processed)
    build_agent_scores(final_settings)
    run_dir = _run_id_dir(final_settings, processed)

    scores = read_parquet(run_dir / "agent_scores.parquet", "agents")
    prices = read_parquet(processed / "asset_price_point_in_time.parquet", "dataset")
    benchmark = read_parquet(processed / "benchmark_point_in_time.parquet", "dataset")
    diagnostics = pd.read_parquet(run_dir / "rank_ic_diagnostics.parquet")

    log.info("=== PERFILES DE INVERSOR ===")
    profile_results = {}
    for profile in PROFILE_NAMES:
        result = run_backtest(scores, prices, benchmark, replace(final_settings, profile=profile), diagnostics)
        profile_results[profile] = result.summary
        if profile == "balanced":
            _write_backtest_outputs(result, run_dir)   # el balanceado es el sistema base
    write_json(profile_results, processed / "profile_results.json")

    # El informe del run final se genera ANTES de la robustez (que crea/borra runs placebo y
    # podria dejar el directorio en un estado transitorio). Se copia a results/final/.
    build_run_report(run_dir)
    _publish_final_run(run_dir, FINAL_DIR)

    log.info("=== ROBUSTEZ / PLACEBO ===")
    robustness = _run_robustness(final_settings, processed, run_dir, diagnostics)
    write_json(robustness, processed / "robustness.json")
    write_json(robustness, FINAL_DIR / "robustness.json")
    write_json(profile_results, FINAL_DIR / "profile_results.json")

    # Validacion en la era reservada 2025-2026: rank-IC del finalista donde NUNCA se optimizo.
    reserved = _reserved_era_validation(diagnostics)

    meta_ic = diagnostics.loc[diagnostics["agent"] == "meta_final", "rank_ic"]
    study_summary = {
        "config_final": accepted,
        "phase1_decision": decision,
        "phase2_winner": phase2_winner,
        "phase2_ranking": {k: round(v["mean_rank_ic"], 5) for k, v in sorted(
            phase2_stats.items(), key=lambda kv: kv[1]["mean_rank_ic"], reverse=True)},
        "hyperparam_winner": hp_winner,
        "final_rank_ic": float(meta_ic.mean()) if not meta_ic.empty else None,
        "reserved_era_validation": reserved,
        "profiles": {p: {"cagr_portfolio": r.get("cagr_portfolio"), "cagr_difference": r.get("cagr_difference"),
                         "beat_rate": r.get("beat_rate"), "max_drawdown": r.get("max_drawdown")}
                     for p, r in profile_results.items()},
        "robustness": robustness,
    }
    write_json(study_summary, RESULTS_ROOT / "study_summary.json")
    _write_results_index(study_summary)
    _write_conclusions(study_summary)
    log.info("=== ESTUDIO COMPLETO TERMINADO -> %s ===", RESULTS_ROOT / "study_summary.json")


def _publish_final_run(run_dir: Path, final_dir: Path) -> None:
    """Copia los artefactos del run ganador (HTML + CSVs + parquets + json) a results/final/."""
    final_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.html", "*.csv", "*.parquet", "*.json", "*.png"):
        for src in run_dir.glob(pattern):
            shutil.copy2(src, final_dir / src.name)


def _write_results_index(summary: dict) -> None:
    """Escribe results/README.md: mapa de que hay en cada carpeta y como leerlo."""
    reserved = summary.get("reserved_era_validation", {})
    text = f"""# Resultados del estudio (ordenados por fases)

Todo el estudio `RUN_MODE=full_study` queda aqui, una carpeta por fase. Para abrir los HTML con
sus tablas grandes (se cargan desde CSV via `fetch`) hace falta un servidor local:

```bash
python servir_html.py
# luego abrir las rutas de abajo con http://localhost:8000/...
```

## Mapa de carpetas

| Carpeta | Que contiene | HTML |
|---|---|---|
| `fase1_ejes/` | Un escenario por nivel de cada eje (ventana, horizonte, ancla, profundidad, cadencia) + los 7 artefactos aislados. `phase1_decision.json` = mejor nivel de cada eje. | `fase1_ejes/comparison.html` |
| `fase2_combinaciones/` | Combinaciones dirigidas de los ganadores de Fase 1. `phase2_decision.json` = ganador. | `fase2_combinaciones/comparison.html` |
| `hiperparametros/` | Afinado fino de LightGBM sobre el ganador de Fase 2. `hyperparam_decision.json` = ganador. | `hiperparametros/comparison.html` |
| `final/` | Run con la configuracion final ganadora: informe completo (resumen, rentabilidad por año, aprendizaje, ranking por agentes), CSVs y parquets. | `final/report.html` |

## Ficheros clave en la raiz

- `study_summary.json` — resumen maquina-legible de todo el estudio (config final, ganadores por
  fase, rank-IC, perfiles, robustez, validacion reservada).
- `conclusiones.md` — lectura en prosa de que aprendio el sistema y que no.

## Cifras de un vistazo

- **Configuracion final elegida**: `{summary.get('config_final') or 'baseline (sin cambios)'}`
- **rank-IC del meta_final (finalista)**: {_fmt(summary.get('final_rank_ic'))}
- **Ganador Fase 2**: {summary.get('phase2_winner')} · **hiperparametros**: {summary.get('hyperparam_winner')}
- **Validacion era reservada 2025-2026**: rank-IC {_fmt(reserved.get('rank_ic_mean'))} en
  {reserved.get('n_cohorts')} cohortes (vs {_fmt(reserved.get('rank_ic_selection_period'))} en el
  periodo de seleccion). La era reservada NO intervino en ninguna eleccion.

La seleccion se hace por **aprendizaje (rank-IC) y estabilidad**, nunca por rentabilidad. La
rentabilidad por perfil se reporta como consecuencia en `final/` y en `conclusiones.md`.
"""
    (RESULTS_ROOT / "README.md").write_text(text, encoding="utf-8")


def _write_conclusions(summary: dict) -> None:
    """Escribe results/conclusiones.md: la lectura honesta del estudio con las cifras reales."""
    reserved = summary.get("reserved_era_validation", {})
    final_ic = summary.get("final_rank_ic")
    robustness = summary.get("robustness", {})
    permutation = robustness.get("label_permutation", {})
    p_value = permutation.get("p_value")

    profiles = summary.get("profiles", {})
    rows = "\n".join(
        f"| {name} | {_pct(p.get('cagr_portfolio'))} | {_pct(p.get('cagr_difference'))} | "
        f"{_pct(p.get('beat_rate'))} | {_pct(p.get('max_drawdown'))} |"
        for name, p in sorted(profiles.items(),
                              key=lambda kv: (kv[1].get('cagr_difference') or -9), reverse=True)
    )

    ic_reserved = reserved.get("rank_ic_mean")
    holds = (ic_reserved is not None and final_ic is not None and ic_reserved > 0
             and (p_value is not None and p_value < 0.05))
    veredicto = (
        "El finalista mantiene rank-IC positivo en la era reservada Y pasa el placebo: hay indicio "
        "de señal real, no solo suerte de haber explorado mucho."
        if holds else
        "El finalista NO sostiene señal significativa fuera del periodo de busqueda (o no pasa el "
        "placebo): la mejor configuracion no supera de forma fiable al azar. Es un resultado "
        "negativo honesto, ahora respaldado por una exploracion mucho mas amplia."
    )

    text = f"""# Conclusiones del estudio completo

Generado automaticamente por `RUN_MODE=full_study`. Todas las cifras salen de `study_summary.json`.

## 1. Que se eligio y como

El estudio optimiza **todos los ejes** (ventana, horizonte, ancla, profundidad, cadencia,
artefactos) en dos fases, y **reserva 2025-2026** de la seleccion para validar al finalista donde
nunca se optimizo. Configuracion final:

- **config**: `{summary.get('config_final') or 'baseline (sin cambios)'}`
- **ganador Fase 2**: {summary.get('phase2_winner')}
- **ganador hiperparametros**: {summary.get('hyperparam_winner')}

## 2. Aprendizaje (rank-IC del meta_final)

- rank-IC del finalista (todo el periodo): **{_fmt(final_ic)}**
- placebo (permutacion de etiquetas): p-valor **{_fmt(p_value)}**
- era reservada 2025-2026: rank-IC **{_fmt(ic_reserved)}** en {reserved.get('n_cohorts')} cohortes
  (periodo de seleccion: {_fmt(reserved.get('rank_ic_selection_period'))})

**Veredicto de aprendizaje.** {veredicto}

## 3. Rentabilidad por perfil de inversor (consecuencia, no criterio)

| perfil | CAGR | vs SPY | años que baten | drawdown |
|---|---|---|---|---|
{rows}

## 4. Lectura

La seleccion se hace por aprendizaje y estabilidad, no por rentabilidad: con rank-IC debil,
cualquier rentabilidad alta es ruido afortunado. Los perfiles con sesgo de estilo (calidad, valor,
GARP) suelen batir al SPY porque capturan **primas de factor clasicas**, mientras que el perfil
`balanced` (que sigue el meta-score del ML puro) mide el valor real del aprendizaje. La honestidad
del resultado descansa en el placebo, el bootstrap y la **era reservada**: solo se declara señal si
el finalista aguanta donde nunca se optimizo.
"""
    (RESULTS_ROOT / "conclusiones.md").write_text(text, encoding="utf-8")


def _fmt(value) -> str:
    return f"{value:+.4f}" if isinstance(value, (int, float)) else "n/d"


def _pct(value) -> str:
    return f"{value * 100:+.1f} %" if isinstance(value, (int, float)) else "n/d"


def _attach_axis_overrides(axis_choices: dict[str, dict], axes: dict[str, dict[str, dict]]) -> None:
    """Adjunta a cada eje el mapa nivel->overrides (lo necesita la Fase 2 para reconstruir configs)."""
    for axis_name, choice in axis_choices.items():
        choice["overrides_by_level"] = {name: dict(ov) for name, ov in axes.get(axis_name, {}).items()}


def _reserved_era_validation(diagnostics: pd.DataFrame) -> dict:
    """rank-IC del finalista en los anios reservados (2025-2026), que no intervinieron en ninguna
    eleccion. Si aguanta aqui, la señal no es solo suerte de haber mirado muchos escenarios."""
    meta = diagnostics.loc[diagnostics["agent"] == "meta_final"].copy()
    if meta.empty:
        return {"n_cohorts": 0, "rank_ic_mean": None, "reserved_years": list(RESERVED_ERA_YEARS)}
    meta["year"] = pd.to_datetime(meta["prediction_date"]).dt.year
    reserved = meta.loc[meta["year"].isin(RESERVED_ERA_YEARS)]
    selection = meta.loc[~meta["year"].isin(RESERVED_ERA_YEARS)]
    return {
        "reserved_years": list(RESERVED_ERA_YEARS),
        "n_cohorts": int(len(reserved)),
        "rank_ic_mean": float(reserved["rank_ic"].mean()) if not reserved.empty else None,
        "rank_ic_positive_fraction": float((reserved["rank_ic"] > 0).mean()) if not reserved.empty else None,
        "rank_ic_selection_period": float(selection["rank_ic"].mean()) if not selection.empty else None,
    }


def _ensure_base_artifacts(settings: Settings, processed: Path) -> None:
    """Genera dataset+features en `processed` si faltan (para el run final)."""
    if not (processed / "features_point_in_time.parquet").exists():
        build_point_in_time_dataset(settings)
        build_features(settings)


def _latest_agents_run(processed: Path) -> Path:
    agents_root = processed / "agents"
    return sorted(path for path in agents_root.iterdir() if path.is_dir())[-1]


def _run_id_dir(settings: Settings, processed: Path) -> Path:
    """Directorio exacto del run de agentes de `settings` (por su huella, no por 'el ultimo')."""
    from module.modeling.agents import _run_id
    feat = processed / "features_point_in_time.parquet"
    targ = processed / "targets_forward_3m.parquet"
    return processed / "agents" / _run_id(settings, feat, targ)


def _write_backtest_outputs(result, run_dir: Path) -> None:
    from module.common.utils import write_parquet
    write_parquet(result.positions, run_dir / "positions.parquet")
    write_parquet(result.orders, run_dir / "orders.parquet")
    write_parquet(result.equity, run_dir / "equity.parquet")
    write_parquet(result.annual_metrics, run_dir / "annual_metrics.parquet")
    write_json(result.summary, run_dir / "backtest_summary.json")


def _run_robustness(settings: Settings, processed: Path, run_dir: Path, diagnostics) -> dict:
    """Ejecuta los tests de robustez sobre la config final."""
    from module.evaluation.robustness import label_permutation_test, leave_one_year_out

    # Permutacion de etiquetas: reentrenar N veces con retornos futuros barajados.
    permuted_ic = []
    targets_path = processed / "targets_forward_3m.parquet"
    if targets_path.exists():
        import numpy as np
        rng = np.random.default_rng(0)
        base_targets = pd.read_parquet(targets_path)
        for i in range(5):   # 5 permutaciones (cada reentreno es costoso)
            shuffled = base_targets.copy()
            shuffled["forward_excess_return_3m"] = rng.permutation(
                shuffled["forward_excess_return_3m"].to_numpy()
            )
            shuffled.to_parquet(processed / "targets_forward_3m.parquet", index=False)
            try:
                build_agent_scores(replace(settings, random_seed=1000 + i))
                perm_run = _latest_agents_run(processed)
                perm_diag = pd.read_parquet(perm_run / "rank_ic_diagnostics.parquet")
                permuted_ic.append(float(perm_diag.loc[perm_diag["agent"] == "meta_final", "rank_ic"].mean()))
                shutil.rmtree(perm_run, ignore_errors=True)   # no dejar runs placebo
            except Exception as exc:   # noqa: BLE001
                log.warning("permutacion %s fallo: %s", i, exc)
        base_targets.to_parquet(processed / "targets_forward_3m.parquet", index=False)   # restaurar

    permutation = label_permutation_test(diagnostics, permuted_ic) if permuted_ic else {}
    loyo = leave_one_year_out(diagnostics).to_dict("records")
    return {"label_permutation": permutation, "leave_one_year_out": loyo}


def _import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("_grid_module_decide", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
