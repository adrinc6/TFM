"""Portfolio Study: elige las variables de cartera que maximizan el Information Ratio.

**Qué es y qué no es.** El Model Study elige el *modelo* comparando Rank-IC: mide si la ordenación
transversal de acciones se sostiene fuera de muestra. Este estudio no toca el modelo. Parte de un
ganador ya congelado, reutiliza sus scores tal cual —no reentrena nada— y solo decide **cómo se
traduce esa ordenación en una cartera**: cuántas posiciones, cuánto efectivo, cómo se reparten los
pesos, cuánto se retiene una posición y cuándo se la deja caer.

**Por qué existe.** El Rank-IC y el Information Ratio miden cosas distintas y pueden divergir: se
observó un ganador con el mejor Rank-IC de su cadena y a la vez el peor coeficiente de
transferencia, es decir, ordenaba mejor el universo y convertía peor esa ordenación en
rentabilidad. Optimizar la ordenación no optimiza la cartera, así que la cartera necesita su propio
criterio. Aquí ese criterio es el IR, que penaliza la volatilidad del exceso en vez de premiar el
alfa bruto.

**Por qué cartesiano y no greedy.** Las variables de cartera interactúan fuerte: el efecto de
`target_size` depende de `max_cash_weight` (el suelo de diversificación sale de ambas) y el de
`coverage_percentile_floor` depende de si la plaza liberada se recompra o queda en efectivo. Un
recorrido greedy —que fija una variable antes de mirar la siguiente— no ve esas interacciones. El
cartesiano completo de las seis sí, y es asequible porque cada combinación solo rehace el backtest
sobre scores ya calculados (segundos, no minutos).

**Qué se guarda (regla 5 del repositorio).** Un cartesiano genera cientos de combinaciones y
conservar la evidencia entera de todas multiplicaría el disco sin aportar nada: son carteras
descartadas. Se guarda una fila de resumen de cada una y la **evidencia completa solo del mejor
vigente**; cuando otra combinación lo supera, la evidencia del anterior se borra y se persiste la
nueva. Al terminar queda exactamente una carpeta de evidencia: la del ganador.

**Qué NO se optimiza, y por qué.** `commission_bps` y `slippage_bps` quedan fuera a propósito: son
*supuestos de coste*, no decisiones de gestión. Optimizarlos equivaldría a elegir el mundo en el que
la estrategia luce mejor. Los umbrales en puntos básicos y las variables `price_only_*` también
quedan fuera: gobiernan cuándo se opera bajo información incompleta y se estresan aparte.

**Cómo se aísla la era reservada.** Durante la rejilla el backtest **se corta en 2024**: los scores
se recortan antes de simular, así que ninguna de las combinaciones llega a tocar 2025-2026. No basta
con filtrar el resumen al elegir —que es lo que haría un filtro a posteriori—, porque la cartera es
secuencial: si la simulación entra en la era reservada, sus posiciones y su resultado existen, y
basta con mirarlos para caer en la tentación de elegir por ellos. Cortando la serie, ese resultado
**no se ha calculado**. Solo al final, ya elegida la combinación ganadora, se reevalúa una vez sobre
la serie completa para medir cómo se comporta fuera de la ventana con la que se eligió. Así la era
reservada es una comprobación, nunca un criterio.
"""

from __future__ import annotations

import itertools
import json
import logging
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from environment import PORTFOLIO_GRID_WORKERS
from module.studies.catalog import BY_ID, SELECTION_UNTIL_YEAR

log = logging.getLogger(__name__)


# Las seis variables que gobiernan riesgo y, por tanto, el Information Ratio. El orden fija el de
# las columnas del artefacto y el de los bucles; no influye en el resultado, porque el cartesiano
# evalúa todas las combinaciones.
PORTFOLIO_STUDY_VARIABLES = (
    "target_size",
    "max_cash_weight",
    "sizing_mode",
    "minimum_holding_period",
    "coverage_percentile_floor",
    "rebalance_drift_tolerance",
)

# Criterio de selección. Se deja explícito y en un solo sitio para que cambiarlo sea una decisión
# consciente y no un literal escondido en el bucle.
SELECTION_METRIC = "information_ratio"

# Cada cuántas combinaciones se vuelca la rejilla a disco. Con ~6 s por combinación, 25 son unos
# dos minutos y medio: el trabajo máximo que se puede perder si el proceso muere de golpe.
GRID_CHECKPOINT_EVERY = 25

# Cada cuántas combinaciones se persiste el estado del estudio para el panel. Escribir en cada una
# no aporta resolución útil y multiplica la exposición a bloqueos transitorios del sistema de
# ficheros, que en Windows abortan el proceso entero.
STATUS_EVERY = 10


def portfolio_grid(
    definition: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Producto cartesiano de las seis variables de cartera.

    Sin `definition` usa el catálogo entero de cada variable. Con `definition` respeta los valores
    seleccionados, de modo que se pueda acotar la rejilla sin tocar el catálogo.
    """
    axes = []
    for variable_id in PORTFOLIO_STUDY_VARIABLES:
        if definition and variable_id in definition:
            values = list(definition[variable_id]["values"])
        else:
            values = list(BY_ID[variable_id].values)
        axes.append([(variable_id, value) for value in values])
    return [dict(combination) for combination in itertools.product(*axes)]


def _evaluate_combination(
    task: tuple[int, dict[str, Any], dict[str, Any], str, str],
) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], float]:
    """Evalúa una combinación en su propio proceso. Debe ser picklable y no tocar estado del estudio.

    Recibe y devuelve solo datos planos porque cruza una frontera de proceso. En particular **no
    escribe** en `study.json` ni en `events.jsonl`: esos ficheros se actualizan por lectura-
    modificación-escritura y varios procesos compitiendo por ellos se pisarían las actualizaciones.
    El padre conserva en exclusiva esa escritura, igual que la comparación del mejor.

    Cada tarea recibe además un `staging` propio, derivado del índice: la ruta única compartida que
    usaba el recorrido secuencial haría que dos workers se borrasen la evidencia mutuamente.
    """
    from module.studies.runner import run_profile_evaluation

    index, combination, values, selection_dir, staging = task
    staging_path = Path(staging)
    if staging_path.exists():
        shutil.rmtree(staging_path, ignore_errors=True)
    started = time.perf_counter()
    payload = run_profile_evaluation(values, "balanced", Path(selection_dir), retain_dir=staging_path)
    elapsed = time.perf_counter() - started
    return index, combination, _selection_summary(payload), dict(payload.get("rank_ic", {})), elapsed


def _metric(summary: Mapping[str, Any]) -> float:
    """Métrica de selección, con los no-finitos tratados como el peor valor posible.

    Una combinación degenerada puede producir `NaN` (por ejemplo si el exceso no tiene varianza).
    Devolver `-inf` la deja fuera del ganador sin necesidad de un caso especial en el bucle.
    """
    value = summary.get(SELECTION_METRIC)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if number == number and abs(number) != float("inf") else float("-inf")


def combination_key(combination: Mapping[str, Any]) -> str:
    """Identidad estable de una combinación, para reanudar sin repetir trabajo.

    Se serializa en el orden fijo de `PORTFOLIO_STUDY_VARIABLES` —no en el de inserción del dict—
    para que la clave no dependa de cómo se construyó el diccionario.
    """
    return json.dumps(
        [combination[variable_id] for variable_id in PORTFOLIO_STUDY_VARIABLES],
        ensure_ascii=False,
    )


def _row(combination: Mapping[str, Any], summary: Mapping[str, Any], elapsed: float) -> dict[str, Any]:
    """Fila escalar de una combinación: lo que se conserva de las descartadas."""
    return {
        **{key: json.dumps(value, ensure_ascii=False) for key, value in combination.items()},
        "information_ratio": summary.get("information_ratio"),
        "geometric_excess_return": summary.get("geometric_excess_return"),
        "cagr_portfolio": summary.get("cagr_portfolio"),
        "cagr_benchmark": summary.get("cagr_benchmark"),
        "max_drawdown": summary.get("max_drawdown"),
        "beat_rate": summary.get("beat_rate"),
        "mean_annual_alpha": summary.get("mean_annual_alpha"),
        "worst_year_alpha": summary.get("worst_year_alpha"),
        "annualized_turnover": summary.get("annualized_turnover"),
        "mean_cash_weight": summary.get("mean_cash_weight"),
        "total_cost_drag": summary.get("total_cost_drag"),
        "transfer_coefficient": summary.get("transfer_coefficient"),
        "elapsed_seconds": elapsed,
    }


def _selection_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resumen restringido a la ventana de selección.

    El backtest devuelve el resumen de la ventana de selección en la raíz y el de la era reservada
    bajo `confirmation`. La elección **solo** puede mirar la primera: 2025-2026 es confirmación y no
    participa en ninguna decisión (regla 3 del repositorio). Durante la rejilla `confirmation` viene
    además vacío, porque la simulación ni siquiera alcanza esos años.
    """
    summary = payload.get("summary", {})
    return {key: value for key, value in summary.items() if key not in {"confirmation", "full_curve"}}


def _resume(grid_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Filas ya evaluadas y sus claves, para no repetirlas al reanudar.

    Si el artefacto no existe o está corrupto —un volcado interrumpido a medias— se empieza de
    cero: es preferible repetir trabajo a arrastrar un estado inconsistente.
    """
    if not grid_path.exists():
        return [], set()
    try:
        frame = pd.read_parquet(grid_path)
    except Exception:
        return [], set()
    rows = frame.to_dict("records")
    done = set()
    for row in rows:
        try:
            done.add(json.dumps(
                [json.loads(row[variable_id]) for variable_id in PORTFOLIO_STUDY_VARIABLES],
                ensure_ascii=False,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return rows, done


def _best_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Reconstruye el mejor vigente desde las filas ya evaluadas.

    Al reanudar hace falta saber contra qué comparar. El resumen que se recupera es parcial —solo
    las columnas escalares de la fila—, pero es suficiente: la evidencia completa del mejor sigue
    en `evidence_best/` y solo se sustituye si otra combinación lo supera.
    """
    best: dict[str, Any] | None = None
    for row in rows:
        score = _metric(row)
        if best is not None and score <= best["score"]:
            continue
        try:
            combination = {
                variable_id: json.loads(row[variable_id])
                for variable_id in PORTFOLIO_STUDY_VARIABLES
            }
        except (KeyError, TypeError, ValueError):
            continue
        best = {
            "score": score,
            "combination": combination,
            "summary": {key: value for key, value in row.items() if key not in PORTFOLIO_STUDY_VARIABLES},
            "rank_ic": {},
        }
    return best


def selection_evidence(evidence_dir: Path, workspace: Path) -> Path:
    """Copia de la evidencia con los scores recortados a la ventana de selección.

    El backtest simula todos los snapshots que encuentra en `agent_scores.parquet`. Recortando ese
    fichero, la simulación **termina en 2024** y la era reservada no llega a existir para ninguna
    combinación de la rejilla: no hay resultado que mirar, ni por tanto forma de elegir por él.

    Se copian también los ficheros que el backtest necesita leer del directorio de evidencia
    (`dataset_reference.json` y los diagnósticos de Rank-IC), sin tocar los originales.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(evidence_dir / "agent_scores.parquet")
    years = pd.to_datetime(scores["snapshot_date"]).dt.year
    truncated = scores.loc[years <= SELECTION_UNTIL_YEAR]
    if truncated.empty:
        raise RuntimeError(
            f"No quedan snapshots hasta {SELECTION_UNTIL_YEAR}: la rejilla no tendría nada que simular."
        )
    truncated.to_parquet(workspace / "agent_scores.parquet", index=False)
    for name in ("dataset_reference.json", "rank_ic_diagnostics.parquet"):
        source = evidence_dir / name
        if source.exists():
            shutil.copy2(source, workspace / name)
    return workspace


def run_portfolio_study(
    winner_values: Mapping[str, Any],
    evidence_dir: Path,
    output_dir: Path,
    *,
    definition: Mapping[str, Mapping[str, Any]] | None = None,
    progress: Any | None = None,
    profiles: list[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Recorre la rejilla, elige por IR y conserva la evidencia solo del mejor.

    `winner_values` es la configuración completa del ganador del Model Study; las seis variables de
    cartera se sobrescriben en cada iteración y el resto se mantiene intacto, de modo que el modelo
    evaluado es siempre el mismo.

    `workers` reparte la rejilla entre procesos. No altera el resultado —cada combinación es
    independiente y el ganador se elige con el mismo criterio sobre el mismo conjunto—, solo el
    tiempo. Con `1` el recorrido es el secuencial de siempre.
    """
    workers = PORTFOLIO_GRID_WORKERS if workers is None else max(1, int(workers))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_evidence = output_dir / "evidence_best"
    grid_path = output_dir / "portfolio_grid.parquet"
    grid = portfolio_grid(definition)
    best: dict[str, Any] | None = None

    # Reanudación: las combinaciones ya evaluadas se leen del artefacto y no se repiten. Sin esto,
    # pausar el estudio tiraría horas de trabajo, porque la rejilla no tiene runs individuales que
    # reutilizar como sí los tiene el Model Study.
    rows, done = _resume(grid_path)
    if done:
        best = _best_from_rows(rows)

    # Toda la rejilla se evalúa contra una evidencia recortada en 2024. Ninguna combinación puede
    # ver la era reservada porque su simulación no llega hasta ahí.
    selection_dir = selection_evidence(evidence_dir, output_dir / "_selection_evidence")

    # Las combinaciones pendientes son independientes entre sí —el cartesiano no es greedy: ninguna
    # parte del resultado de otra—, así que se reparten entre procesos. La reducción (comparar,
    # promover la evidencia y volcar la rejilla) se queda **entera en el padre**: es la parte con
    # estado compartido y paralelizarla sería una carrera sobre el sistema de ficheros.
    pending = [
        (index, combination)
        for index, combination in enumerate(grid, start=1)
        if combination_key(combination) not in done
    ]
    for index, combination in enumerate(grid, start=1):
        if combination_key(combination) in done and progress is not None and best is not None:
            progress(index, len(grid), combination, best["score"], best)

    staging_root = output_dir / "_staging"
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            index,
            dict(combination),
            {**dict(winner_values), **combination},
            str(selection_dir),
            str(staging_root / f"c{index:06d}"),
        )
        for index, combination in pending
    ]

    def reduce_one(
        index: int, combination: dict[str, Any], summary: dict[str, Any],
        rank_ic: dict[str, Any], elapsed: float,
    ) -> None:
        """Integra el resultado de una combinación. Solo la llama el padre, en serie."""
        nonlocal best, completed
        rows.append(_row(combination, summary, elapsed))
        done.add(combination_key(combination))
        score = _metric(summary)
        staging = staging_root / f"c{index:06d}"
        if best is None or score > best["score"]:
            if best_evidence.exists():
                shutil.rmtree(best_evidence, ignore_errors=True)
            staging.replace(best_evidence)
            best = {
                "score": score, "combination": dict(combination),
                "summary": dict(summary), "rank_ic": dict(rank_ic),
            }
        else:
            shutil.rmtree(staging, ignore_errors=True)
        completed += 1
        # Volcado periódico: si el proceso muere entre dos volcados solo se pierde ese tramo, y la
        # evidencia del mejor ya está en disco, así que reanudar reconstruye el estado.
        if completed % GRID_CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_parquet(grid_path, index=False)
        if progress is not None:
            progress(len(done), len(grid), combination, score, best)

    completed = 0
    if tasks and workers > 1:
        # `spawn` es el único modo en Windows y evita heredar estado del padre; el coste de arranque
        # se amortiza porque cada worker atiende cientos de combinaciones.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for result in pool.map(_evaluate_combination, tasks, chunksize=1):
                reduce_one(*result)
    else:
        for task in tasks:
            reduce_one(*_evaluate_combination(task))

    shutil.rmtree(staging_root, ignore_errors=True)
    pd.DataFrame(rows).to_parquet(grid_path, index=False)

    if best is None:
        raise RuntimeError("La rejilla de cartera quedó vacía: no hay combinación que evaluar.")

    # Ya elegida la combinación, y solo ella, se reevalúa sobre la serie completa. Es la primera y
    # única vez que 2025-2026 se calcula en todo el estudio: como comprobación del ganador, nunca
    # como criterio para elegirlo.
    from module.studies.runner import run_profile_evaluation

    confirmation_evidence = output_dir / "evidence_best_full"
    if confirmation_evidence.exists():
        shutil.rmtree(confirmation_evidence, ignore_errors=True)
    full = run_profile_evaluation(
        {**dict(winner_values), **best["combination"]}, "balanced", evidence_dir,
        retain_dir=confirmation_evidence, include_model_artifacts=True,
    )
    full_summary = full.get("summary", {})

    # Los ocho perfiles de inversor, ya con la cartera ganadora. Responden a una pregunta distinta
    # —«cómo le habría ido a un inversor value, defensivo, momentum… con la mejor gestión»— y por
    # eso NO compiten en la rejilla: un perfil reordena la señal (sustituye el `meta_rank` y
    # recalibra el alfa), mientras las seis variables solo gestionan la cartera ya elegida.
    # Elegir perfil por IR sería escoger el estilo de inversor por su rentabilidad conocida.
    profile_rows = _profiles_with_winner(
        {**dict(winner_values), **best["combination"]}, evidence_dir, selection_dir, output_dir,
        profiles,
    )
    shutil.rmtree(selection_dir, ignore_errors=True)

    winner = {
        "selection_metric": SELECTION_METRIC,
        "selection_until_year": SELECTION_UNTIL_YEAR,
        "known_stress_role": "known_stress_not_selection",
        "grid_backtest_truncated_at": SELECTION_UNTIL_YEAR,
        "variables": list(PORTFOLIO_STUDY_VARIABLES),
        "combinations": len(grid),
        "profiles_evaluated": [row["profile"] for row in profile_rows],
        "winner_combination": best["combination"],
        "winner_summary": best["summary"],
        "winner_confirmation": dict(full_summary.get("confirmation", {})),
        "winner_full_curve": dict(full_summary.get("full_curve", {})),
        "configuration": {**dict(winner_values), **best["combination"]},
    }
    (output_dir / "portfolio_winner.json").write_text(
        json.dumps(winner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    winner["diagnostics"] = run_diagnostics(
        confirmation_evidence, evidence_dir, winner["configuration"], output_dir,
    )
    return winner


def run_diagnostics(
    winner_evidence: Path, evidence_dir: Path, configuration: Mapping[str, Any], output_dir: Path,
) -> dict[str, Any]:
    """Costes, capacidad y narrativa de cartera sobre el ganador ya congelado.

    Se calculan **aquí y automáticamente** porque son parte del informe, no un análisis aparte que
    haya que acordarse de lanzar: en cuanto la cadena termina, las cifras del capítulo económico
    están completas. Los tres son posteriores a la elección y ninguno escribe en `winner.json`,
    `decisions.json` ni `portfolio_winner.json`.

    Un fallo en un diagnóstico **no tumba el estudio**: la rejilla ya ha corrido —horas de cómputo— y
    su ganador ya está en disco. El error se registra en el propio artefacto, que es donde se va a
    buscar, en vez de perderse en un traceback que aborta un trabajo ya terminado.
    """
    from module.research import capacity, cost_sensitivity, portfolio_narrative
    from module.studies.config import settings_from_values

    status: dict[str, Any] = {}

    def panel() -> Path:
        """Ruta al dataset preparado, exigida solo por quien la necesita.

        La sensibilidad a costes se calcula entera sobre la evidencia de cartera, así que no debe
        caerse porque falte la referencia al panel: solo capacidad y narrativa la necesitan.
        """
        path = evidence_dir / "dataset_reference.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No hay {path}: sin referencia al panel no se puede localizar el dataset preparado."
            )
        return Path(json.loads(path.read_text(encoding="utf-8"))["prepared_path"])

    def attempt(name: str, action: Any) -> None:
        log.info("Diagnóstico %s del ganador de cartera", name)
        try:
            action()
        except Exception as error:
            log.exception("Diagnóstico %s fallido", name)
            status[name] = {"available": False, "error": str(error)}
            (output_dir / f"{name}.json").write_text(
                json.dumps({"available": False, "error": str(error)}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        else:
            status[name] = {"available": True}

    # La familia congelada sale de la curva del ganador; la resimulada vuelve a simular sobre los
    # scores del Model Study de origen, que es donde viven con seguridad.
    attempt("cost_sensitivity", lambda: cost_sensitivity.write_cost_sensitivity(
        output_dir / "cost_sensitivity.json",
        cost_sensitivity.build_cost_sensitivity(
            winner_evidence, configuration, simulation_dir=evidence_dir,
        ),
    ))
    attempt("capacity", lambda: capacity.write_capacity(
        output_dir / "capacity.json", capacity.build_capacity(winner_evidence, panel()),
    ))

    def narrative() -> None:
        from module.modeling.features import load_sector_map

        settings = settings_from_values(dict(configuration), profile="balanced")
        payload, holdings = portfolio_narrative.build_portfolio_narrative(
            winner_evidence, panel(), configuration, load_sector_map(settings),
        )
        portfolio_narrative.write_portfolio_narrative(output_dir, payload, holdings)

    attempt("portfolio_narrative", narrative)
    return status


def _profiles_with_winner(
    values: Mapping[str, Any], evidence_dir: Path, selection_dir: Path, output_dir: Path,
    profiles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evalúa los ocho perfiles de inversor con la cartera ganadora.

    Cada perfil se mide dos veces y por el mismo motivo que el ganador: sobre la serie **recortada**
    para las cifras de la ventana de selección, y sobre la **completa** para las de la era
    reservada. Así la comparación entre perfiles nunca se contamina con 2025-2026, y a la vez puede
    reportarse cómo se comportó cada estilo fuera de la ventana.

    Es un diagnóstico informativo: ningún perfil se elige por su Information Ratio.
    """
    from module.evaluation.profiles import PROFILE_NAMES
    from module.studies.runner import run_profile_evaluation

    selected = list(PROFILE_NAMES) if profiles is None else [
        name for name in PROFILE_NAMES if name in profiles
    ]
    rows: list[dict[str, Any]] = []
    for profile in selected:
        retain = output_dir / "profiles" / profile
        if retain.exists():
            shutil.rmtree(retain, ignore_errors=True)
        try:
            selection = _selection_summary(
                run_profile_evaluation(dict(values), profile, selection_dir),
            )
            full = run_profile_evaluation(
                dict(values), profile, evidence_dir, retain_dir=retain, include_model_artifacts=True,
            )
        except ValueError as exc:
            # Un perfil exige los rangos de agentes que pondera; si el ganador se entrenó sin
            # alguno, el perfil no es aplicable y se registra como tal en vez de romper el estudio.
            rows.append({"profile": profile, "applicable": False, "reason": str(exc)})
            continue
        confirmation = full.get("summary", {}).get("confirmation", {})
        rows.append({
            "profile": profile,
            "applicable": True,
            "information_ratio": selection.get("information_ratio"),
            "geometric_excess_return": selection.get("geometric_excess_return"),
            "annualized_turnover": selection.get("annualized_turnover"),
            "mean_cash_weight": selection.get("mean_cash_weight"),
            "max_drawdown": selection.get("max_drawdown"),
            "beat_rate": selection.get("beat_rate"),
            "confirmation_information_ratio": confirmation.get("information_ratio"),
            "confirmation_excess": confirmation.get("geometric_excess_return"),
            "confirmation_beat_rate": confirmation.get("beat_rate"),
        })
    pd.DataFrame(rows).to_parquet(output_dir / "portfolio_profiles.parquet", index=False)
    return rows


def execute_portfolio_study(study_id: str) -> dict[str, Any]:
    """Ejecuta un Portfolio Study completo, con eventos y progreso para el panel.

    Toma el ganador del Model Study de referencia, recorre la rejilla y deja en el directorio la
    comparación de todas las combinaciones más la evidencia de la mejor.
    """
    from module.storage.evidence import write_portfolio_report
    from module.storage.studies import (
        append_event, read_study, safe_study_path, update_study, write_storage_manifest,
    )

    study = read_study(study_id)
    directory = safe_study_path(study_id)
    source_id = study["source_study_id"]
    source = safe_study_path(source_id)
    winner_values = json.loads((source / "winner.json").read_text(encoding="utf-8"))["configuration"]

    definition = study.get("configuration") or {}
    grid = portfolio_grid(definition)
    update_study(study_id, status="running", phase="grid", progress=0.01)
    append_event(
        study_id, "info", "portfolio_study_started",
        f"Portfolio Study sobre {source_id}: {len(grid)} combinaciones por {SELECTION_METRIC}.",
    )

    # La cartera de partida es la del ganador tal cual: la referencia contra la que se mide si
    # optimizar por IR aporta algo. Se evalúa con la MISMA serie recortada que la rejilla; medirla
    # sobre la serie completa la compararía contra ventanas distintas y la mejora sería ficticia.
    baseline_dir = selection_evidence(source / "evidence", directory / "_baseline_evidence")
    baseline_summary = _selection_summary(_evaluate(winner_values, baseline_dir, None))
    shutil.rmtree(baseline_dir, ignore_errors=True)

    def progress(index: int, total: int, combination: Mapping[str, Any], score: float, best: Mapping[str, Any]) -> None:
        # El estado se persiste cada `STATUS_EVERY` combinaciones, no en cada una. Escribir 1.728
        # veces multiplica por 1.728 las probabilidades de tropezar con un bloqueo transitorio del
        # sistema de ficheros, y el progreso no necesita esa resolución: con ~6 s por combinación,
        # actualizar cada 10 es refrescar el panel cada minuto.
        if index % STATUS_EVERY == 0 or index == total or index == 1:
            update_study(
                study_id, completed_runs=index, progress=min(0.99, index / max(total, 1)),
                phase="grid", heartbeat=_now(),
            )
        if index % 25 == 0 or index == total:
            append_event(
                study_id, "info", "portfolio_progress",
                f"{index}/{total} · mejor {SELECTION_METRIC}={best['score']:.4f} con {best['combination']}",
            )

    winner = run_portfolio_study(
        winner_values, source / "evidence", directory, definition=definition, progress=progress,
        profiles=study.get("profiles"),
    )
    winner["source_study_id"] = source_id
    winner["study_id"] = study_id
    winner["baseline_summary"] = baseline_summary
    winner["improvement"] = improvement(baseline_summary, winner["winner_summary"])
    (directory / "portfolio_winner.json").write_text(
        json.dumps(winner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    # Los diagnósticos ya están en disco; se anuncia cuál salió y cuál no, porque un artefacto que
    # falló en silencio es peor que uno que falta.
    for name, state in (winner.get("diagnostics") or {}).items():
        level = "info" if state.get("available") else "warning"
        message = (
            f"Diagnóstico {name} escrito." if state.get("available")
            else f"Diagnóstico {name} no disponible: {state.get('error')}"
        )
        append_event(study_id, level, f"diagnostic_{name}", message)

    write_portfolio_report(directory / "report.md", {**winner, **_diagnostic_payloads(directory)})
    write_storage_manifest(study_id)
    update_study(
        study_id, status="succeeded", phase="completed", progress=1.0,
        completed_runs=winner["combinations"], heartbeat=_now(),
    )
    append_event(
        study_id, "info", "portfolio_study_succeeded",
        f"Ganador: {winner['winner_combination']} con {SELECTION_METRIC}="
        f"{winner['winner_summary'].get(SELECTION_METRIC)}.",
    )
    return winner


def _diagnostic_payloads(directory: Path) -> dict[str, Any]:
    """Lee de disco los tres diagnósticos para componer el informe.

    Se releen en vez de arrastrarlos en memoria para que el informe describa exactamente lo que hay
    en los artefactos, que es lo que el manuscrito va a citar. Un diagnóstico ausente deja su bloque
    vacío en vez de romper el informe.
    """
    payloads: dict[str, Any] = {}
    for name in ("cost_sensitivity", "capacity", "portfolio_narrative"):
        path = directory / f"{name}.json"
        payloads[name] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
    return payloads


def _evaluate(values: Mapping[str, Any], evidence_dir: Path, retain: Path | None) -> dict[str, Any]:
    from module.studies.runner import run_profile_evaluation

    return run_profile_evaluation(dict(values), "balanced", evidence_dir, retain_dir=retain)


def _now() -> str:
    from module.storage.studies import utc_now

    return utc_now()


def improvement(baseline: Mapping[str, Any], winner: Mapping[str, Any]) -> dict[str, Any]:
    """Mejora del ganador de cartera frente a la cartera de partida, métrica a métrica."""
    keys = (
        "information_ratio", "geometric_excess_return", "annualized_turnover",
        "max_drawdown", "beat_rate", "transfer_coefficient",
    )
    result = {}
    for key in keys:
        before, after = baseline.get(key), winner.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            result[key] = {"baseline": float(before), "winner": float(after), "delta": float(after) - float(before)}
    return result
