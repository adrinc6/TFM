"""Servidor HTTP del único flujo Model Study."""

from __future__ import annotations

import json
import math
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from environment import PROJECT_ROOT
from module.studies.catalog import public_catalog
from module.studies.config import initial_values, retention_budget, validate_definition
from module.storage.studies import create_run, create_study, read_study, update_study
from module.web import dev, queries
from module.web.jobs import WORKERS


APP_ROOT = PROJECT_ROOT / "app"


def _json_safe(value: Any) -> Any:
    # NaN/Infinity son floats válidos en Python pero no en JSON: json.dumps los emite como
    # literales NaN/Infinity que JSON.parse del navegador rechaza. Se convierten a null aquí.
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def study_preflight(payload: dict) -> dict:
    allowed = {"name", "note", "definition", "run_scope", "retain_all_runs"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Campos desconocidos: {sorted(unknown)}.")
    definition, budget = validate_definition(payload.get("definition"))
    scope = payload.get("run_scope", "full")
    if scope not in {"full", "dev"}:
        raise ValueError("run_scope debe ser full o dev.")
    retain_all = payload.get("retain_all_runs", False)
    if not isinstance(retain_all, bool):
        raise ValueError("retain_all_runs debe ser booleano.")
    return {
        "valid": True,
        "definition": definition,
        "budget": retention_budget(budget, retain_all),
        "run_scope": scope,
        "retain_all_runs": retain_all,
    }


def launch_study(payload: dict) -> dict:
    preflight = study_preflight(payload)
    catalog = public_catalog()
    study_id, _ = create_study({
        "name": str(payload.get("name") or "Model Study"),
        "note": str(payload.get("note") or ""),
        "run_scope": preflight["run_scope"],
        "retain_all_runs": preflight["retain_all_runs"],
        "configuration": preflight["definition"],
        "budget": preflight["budget"],
        "catalog_version": catalog["version"],
        "catalog_hash": catalog["hash"],
        "catalog": catalog,
    })
    create_run(
        study_id,
        logical_key="predictive:baseline",
        phase="temporal",
        variable_id="baseline",
        value="baseline",
        configuration=initial_values(preflight["definition"]),
    )
    pid = WORKERS.start(study_id)
    return {"study_id": study_id, "status": "queued", "worker_pid": pid}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            if path.startswith("/dev/api/"):
                return self._send(dev.get(path.removeprefix("/dev"), query))
            if path == "/api/catalog":
                return self._send(public_catalog())
            if path == "/api/studies":
                return self._send(queries.studies())
            if path.startswith("/api/studies/"):
                return self._study_get(path, query)
            return self._static(path)
        except FileNotFoundError as exc:
            self._send({"error": str(exc)}, 404)
        except (ValueError, KeyError) as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            self._send({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, payload = urlparse(self.path).path, self._body()
            if path.startswith("/dev/api/"):
                result, status = dev.post(path.removeprefix("/dev"), payload)
                return self._send(result, status)
            if path == "/api/studies/preflight":
                return self._send(study_preflight(payload))
            if path == "/api/studies":
                return self._send(launch_study(payload), 202)
            if path.startswith("/api/studies/") and path.endswith("/cancel"):
                study_id = path.removeprefix("/api/studies/").removesuffix("/cancel")
                WORKERS.cancel(study_id)
                return self._send(read_study(study_id))
            if path.startswith("/api/studies/") and path.endswith("/pause"):
                study_id = path.removeprefix("/api/studies/").removesuffix("/pause")
                WORKERS.pause(study_id)
                return self._send(read_study(study_id))
            if path.startswith("/api/studies/") and path.endswith("/resume"):
                study_id = path.removeprefix("/api/studies/").removesuffix("/resume")
                study = read_study(study_id)
                if study["status"] == "succeeded":
                    return self._send(study)
                update_study(study_id, status="queued", error=None, traceback=None)
                pid = WORKERS.start(study_id)
                return self._send({"study_id": study_id, "status": "queued", "worker_pid": pid}, 202)
            return self._send({"error": "Ruta no encontrada."}, 404)
        except FileNotFoundError as exc:
            self._send({"error": str(exc)}, 404)
        except (ValueError, KeyError) as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            self._send({"error": str(exc)}, 500)

    def _study_get(self, path: str, query: dict[str, list[str]]) -> None:
        parts = path.removeprefix("/api/studies/").strip("/").split("/")
        study_id = parts[0]
        if len(parts) == 1:
            return self._send(queries.study_detail(study_id))
        if parts[1] == "runs" and len(parts) == 3:
            return self._send(queries.run_detail(study_id, parts[2]))
        if parts[1] == "events" and len(parts) == 2:
            return self._send(queries.events(study_id, int(query.get("after", ["0"])[0])))
        if parts[1] == "analysis" and len(parts) == 3:
            return self._send(queries.analysis(study_id, parts[2], query))
        raise FileNotFoundError("Ruta no encontrada.")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("Payload demasiado grande.")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("El payload debe ser un objeto.")
        return payload

    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/", "/dev"} else path.lstrip("/")
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
    WORKERS.reconcile()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard detenido.", flush=True)
    finally:
        WORKERS.stop_all()
        server.server_close()
