"""Proceso hijo de un Model Study."""

from __future__ import annotations

import os
import sys
import threading
import traceback

from module.studies.runner import execute_model_study
from module.common.utils import pid_alive
from module.storage.studies import append_event, read_study, update_run, update_study


def _heartbeat(study_id: str, parent_pid: int, stop: threading.Event) -> None:
    while not stop.wait(2.0):
        try:
            if not pid_alive(parent_pid):
                update_study(
                    study_id, status="interrupted", worker_pid=None,
                    error="El proceso del dashboard terminó durante la ejecución.",
                    heartbeat=_now(),
                )
                append_event(
                    study_id, "warning", "parent_disappeared",
                    "El worker se detiene porque desapareció el dashboard.",
                )
                os._exit(2)
            study = read_study(study_id)
            if study["status"] not in {"queued", "running"}:
                return
            update_study(study_id, heartbeat=_now(), worker_pid=os.getpid())
            run_id = study.get("current_run_id")
            if isinstance(run_id, str):
                update_run(study_id, run_id)
        except Exception:
            return


def run(study_id: str, parent_pid: int) -> int:
    stop = threading.Event()
    thread = threading.Thread(
        target=_heartbeat, args=(study_id, parent_pid, stop), daemon=True,
    )
    update_study(study_id, status="running", worker_pid=os.getpid(), heartbeat=_now())
    append_event(study_id, "info", "worker_started", f"Worker iniciado con PID {os.getpid()}.")
    thread.start()
    try:
        # El tipo decide el motor: el Model Study optimiza el modelo por Rank-IC; el Portfolio
        # Study no toca el modelo y recorre la rejilla de cartera por Information Ratio.
        if read_study(study_id).get("study_type") == "portfolio_study":
            from module.studies.portfolio_study import execute_portfolio_study
            execute_portfolio_study(study_id)
        else:
            execute_model_study(study_id)
        return 0
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        update_study(
            study_id, status="failed", worker_pid=None,
            error=error, traceback=traceback.format_exc(), heartbeat=_now(),
        )
        append_event(study_id, "error", "worker_failed", f"Worker falló: {error}.")
        return 1
    finally:
        stop.set()
        thread.join(timeout=3)
        current = read_study(study_id)
        if current.get("worker_pid") == os.getpid():
            update_study(study_id, worker_pid=None, heartbeat=_now())


def _now() -> str:
    from module.storage.studies import utc_now
    return utc_now()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Uso: python -m module.studies.worker <study_id> <parent_pid>")
    raise SystemExit(run(sys.argv[1], int(sys.argv[2])))
