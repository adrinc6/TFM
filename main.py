"""Entrada única del proyecto y selector de etapas.

``RUN_MODE`` permite ejecutar una etapa concreta o ``full``; ``RUN_SCOPE``
separa los artefactos de desarrollo de los resultados completos.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from environment import Settings, ensure_directories
from module.data.dataset import build_point_in_time_dataset
from module.data.ingest.pipeline import download_raw_data
from module.modeling.agents import build_agent_scores
from module.modeling.features import build_features
from module.evaluation.backtest import run_backtest_from_run_dir
from module.runs.experiments import run_experiments_from_settings
from module.runs.execution import execute_official_optimization
from module.ui.reports import build_report_from_settings
from module.common.utils import setup_logging


log = logging.getLogger(__name__)

# `full_study` es el comando de principio a fin (barrido de ablations -> decision automatica de
# artefactos -> run final optimizado -> perfiles -> robustez -> HTML), sin decisiones humanas.
STAGE_ORDER = ("download", "dataset", "features", "agents", "backtest", "report")
STAGE_HANDLERS: dict[str, Callable[[Settings], None]] = {
    "download": download_raw_data,
    "dataset": build_point_in_time_dataset,
    "features": build_features,
    "agents": build_agent_scores,
    "backtest": run_backtest_from_run_dir,
    "report": build_report_from_settings,
    "experiments": run_experiments_from_settings,
    "full_study": execute_official_optimization,
}


def stages_for_run(run_mode: str) -> tuple[str, ...]:
    """Resuelve el modo solicitado sin ocultar etapas todavía no implementadas."""
    if run_mode == "full":
        return tuple(stage for stage in STAGE_ORDER if stage in STAGE_HANDLERS)
    if run_mode not in STAGE_HANDLERS:
        raise NotImplementedError(
            f"RUN_MODE={run_mode!r} todavía no está implementado. "
            "Ejecuta una etapa disponible o RUN_MODE='full'."
        )
    return (run_mode,)


def main() -> None:
    # Sin RUN_MODE explicito, la entrada principal es la consola local. El comportamiento
    # historico de CLI se conserva intacto cuando el usuario fija RUN_MODE.
    if "RUN_MODE" not in os.environ:
        from module.ui.dashboard import serve
        serve()
        return
    settings = Settings()
    ensure_directories(settings)
    setup_logging()
    stages = stages_for_run(settings.run_mode)
    log.info(
        "Ejecución: mode=%s scope=%s stages=%s",
        settings.run_mode,
        settings.run_scope,
        ", ".join(stages),
    )
    for stage in stages:
        log.info("Iniciando etapa: %s", stage)
        STAGE_HANDLERS[stage](settings)
        log.info("Etapa completada: %s", stage)


if __name__ == "__main__":
    main()
