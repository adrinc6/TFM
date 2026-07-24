"""Servidor HTTP local del flujo Exploratory → Confirmatory."""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from environment import PROJECT_ROOT
from module.studies.catalog import public_catalog
from module.studies.confirmatory import confirmatory_preflight, run_confirmatory
from module.studies.exploratory import (
    advance_exploratory,
    create_exploratory,
    exploratory_preflight,
    freeze_exploratory,
)
from module.storage.cache import cache_usage
from module.storage.datasets import prepared_usage
from module.storage.evidence import storage_usage
from module.web import queries


APP_ROOT = PROJECT_ROOT / "app"


class Jobs:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self, kind: str, function: Callable[[], Any]) -> str:
        job_id = uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"job_id": job_id, "kind": kind, "status": "running"}

        def target() -> None:
            try:
                result = function()
                self._set(job_id, status="succeeded", result=result)
            except Exception as exc:
                self._set(
                    job_id, status="failed", error=str(exc),
                    traceback=traceback.format_exc().splitlines()[-20:],
                )

        threading.Thread(target=target, name=f"job-{kind}-{job_id[:8]}", daemon=True).start()
        return job_id

    def _set(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise FileNotFoundError("Job desconocido.")
            return dict(self._jobs[job_id])


JOBS = Jobs()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path
            if path == "/api/catalog":
                return self._send(public_catalog())
            if path == "/api/studies":
                return self._send(queries.studies())
            if path == "/api/hypotheses":
                return self._send(queries.hypotheses())
            if path == "/api/models":
                return self._send(queries.models())
            if path == "/api/storage":
                return self._send({
                    "results": storage_usage(),
                    "cache": cache_usage(),
                    "prepared": prepared_usage(),
                })
            if path.startswith("/api/jobs/"):
                return self._send(JOBS.get(path.removeprefix("/api/jobs/")))
            if path.startswith("/api/studies/"):
                return self._send(queries.study_detail(path.removeprefix("/api/studies/")))
            if path.startswith("/api/entities/"):
                return self._entity(path, query)
            return self._static(path)
        except FileNotFoundError as exc:
            self._send({"error": str(exc)}, 404)
        except (ValueError, KeyError) as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            self._send({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            payload = self._body()
            if path == "/api/exploratory/preflight":
                return self._send(exploratory_preflight(payload))
            if path == "/api/exploratory":
                job_id = JOBS.start("exploratory", lambda: create_exploratory(payload))
                return self._send({"job_id": job_id}, 202)
            if path.startswith("/api/exploratory/") and path.endswith("/advance"):
                study_id = path.removeprefix("/api/exploratory/").removesuffix("/advance")
                job_id = JOBS.start(
                    "exploratory_advance",
                    lambda: advance_exploratory(
                        study_id,
                        candidate_id=payload.get("candidate_id"),
                        reason=str(payload.get("reason") or "automatic"),
                    ),
                )
                return self._send({"job_id": job_id}, 202)
            if path.startswith("/api/exploratory/") and path.endswith("/freeze"):
                study_id = path.removeprefix("/api/exploratory/").removesuffix("/freeze")
                job_id = JOBS.start("exploratory_freeze", lambda: freeze_exploratory(study_id))
                return self._send({"job_id": job_id}, 202)
            if path == "/api/confirmatory/preflight":
                return self._send(confirmatory_preflight(payload))
            if path == "/api/confirmatory":
                job_id = JOBS.start("confirmatory", lambda: run_confirmatory(payload))
                return self._send({"job_id": job_id}, 202)
            self._send({"error": "Ruta no encontrada."}, 404)
        except FileNotFoundError as exc:
            self._send({"error": str(exc)}, 404)
        except (ValueError, KeyError) as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            self._send({"error": str(exc)}, 500)

    def _entity(self, path: str, query: dict[str, list[str]]) -> None:
        parts = path.removeprefix("/api/entities/").strip("/").split("/")
        if len(parts) not in {2, 3}:
            raise ValueError("Ruta analítica inválida.")
        entity_id, view = parts[:2]
        profile = query.get("profile", [None])[0]
        if view == "performance":
            return self._send(queries.performance(entity_id, profile))
        if view == "learning":
            return self._send(queries.learning(entity_id))
        if view == "rankings":
            return self._send(queries.rankings(entity_id, query.get("snapshot", [None])[0]))
        if view == "portfolio":
            return self._send(queries.portfolio(entity_id, profile))
        if view == "trades":
            return self._send(queries.trades(entity_id, profile))
        if view == "stocks" and len(parts) == 3:
            return self._send(queries.stock(entity_id, parts[2]))
        raise ValueError("Vista analítica desconocida.")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("Payload demasiado grande.")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("El payload debe ser un objeto.")
        return data

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        if relative.startswith("app/"):
            relative = relative.removeprefix("app/")
        target = (APP_ROOT / relative).resolve()
        target.relative_to(APP_ROOT.resolve())
        if not target.is_file():
            target = APP_ROOT / "index.html"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"Dashboard: http://{host}:{port}")
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard detenido.")
    finally:
        server.shutdown()
        server.server_close()
