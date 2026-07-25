"""Gestión de procesos hijos asociados a Studies."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from module.common.utils import pid_alive
from module.storage.studies import append_event, list_studies, read_study, update_study


class StudyWorkers:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, subprocess.Popen[Any]] = {}
        atexit.register(self.stop_all)

    def reconcile(self) -> None:
        for study in list_studies():
            if study.get("status") not in {"queued", "running"}:
                continue
            pid = study.get("worker_pid")
            if not isinstance(pid, int) or not pid_alive(pid):
                update_study(
                    study["study_id"], status="interrupted", worker_pid=None,
                    error="El worker desapareció antes de cerrar el Study.",
                )

    def start(self, study_id: str) -> int:
        with self._lock:
            current = self._workers.get(study_id)
            if current and current.poll() is None:
                return int(current.pid)
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "module.studies.worker",
                    study_id, str(os.getpid()),
                ],
                creationflags=flags,
            )
            self._workers[study_id] = process
            update_study(study_id, worker_pid=process.pid, status="queued")
            return int(process.pid)

    def cancel(self, study_id: str) -> None:
        self._stop(study_id, "cancelled", "study_cancelled", "Study cancelado por el usuario.")

    def pause(self, study_id: str) -> None:
        self._stop(study_id, "interrupted", "study_paused", "Study pausado por el usuario.")

    def _stop(self, study_id: str, status: str, event: str, message: str) -> None:
        with self._lock:
            process = self._workers.get(study_id)
        if process and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        pid = process.pid if process else read_study(study_id).get("worker_pid")
        if isinstance(pid, int) and pid_alive(pid):
            _terminate_tree(pid)
        if isinstance(pid, int) and pid_alive(pid):
            raise RuntimeError(f"No se pudo terminar el worker PID {pid}.")
        update_study(
            study_id, status=status, worker_pid=None,
            heartbeat=datetime.now(timezone.utc).isoformat(),
        )
        append_event(study_id, "warning", event, message)

    def stop_all(self) -> None:
        with self._lock:
            study_ids = [
                study_id for study_id, process in self._workers.items()
                if process.poll() is None
            ]
        for study_id in study_ids:
            try:
                self.cancel(study_id)
            except Exception:
                pass


def _terminate_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False, capture_output=True,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


WORKERS = StudyWorkers()
