"""Consola web local, sin dependencias externas, para runs y resultados."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import traceback
from dataclasses import asdict, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from environment import (MAX_PRICE_AGE_DAYS, MIN_TRAINING_ROWS, NEUTRALIZE_MIN_GROUP,
                         PROJECT_ROOT, RECENCY_HALFLIFE_YEARS, Settings)
from module.runs.execution import execute_official_optimization, execute_run, execute_study
from module.runs.results_store import RESULTS_ROOT, ResultsStore, list_registry


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, callback) -> str:
        job_id = f"job-{len(self._jobs) + 1}"
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "name": name, "status": "queued", "error": None, "result": None}

        def target() -> None:
            self._update(job_id, status="running")
            try:
                self._update(job_id, result=callback(), status="succeeded")
            except Exception as exc:  # noqa: BLE001
                self._update(job_id, status="failed", error=f"{exc}\n{traceback.format_exc()}")

        threading.Thread(target=target, name=job_id, daemon=True).start()
        return job_id

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._jobs.values())


JOBS = JobManager()
STORE = ResultsStore()

# Carpeta del frontend (archivos reales servidos como estáticos en / y /app/*).
# Vive en la raíz del proyecto, junto a module/, results/ y docs/.
APP_ROOT = (PROJECT_ROOT / "app").resolve()

# Valores guiados para studies. Un study no acepta JSON arbitrario: solo combina opciones
# metodológicamente admitidas; Experimental sigue permitiendo ajustar valores individuales.
SETTINGS_GROUPS = {
    "Periodo y entrenamiento": ["execution_year", "execution_quarter", "execution_lag_days", "train_lookback_years", "snapshot_step_months", "fundamental_step_months", "target_horizon_months", "meta_ic_lookback_quarters"],
    "Modelo LightGBM": ["objective", "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type", "recency_weighting"],
    "Artefactos": ["neutralize_by_sector", "fundamental_momentum", "market_regime_feature", "price_momentum_multi", "moving_averages", "regime_extended", "quality_growth_derived"],
    "Laboratorio ML": ["enabled_feature_blocks", "enabled_agents", "enabled_model_families", "intra_agent_ensemble_mode", "feature_weighting_mode", "feature_selection_min_coverage", "feature_selection_lookback_quarters", "feature_selection_min_permutation_importance", "feature_selection_min_positive_fraction", "feature_selection_max_features_per_agent", "metric_winsorization_percentile", "risk_feature_windows", "technical_feature_windows"],
    "Cartera y perfil": ["target_min", "target_max", "entry_min_percentile", "min_hold_percentile", "rotation_edge_percentiles", "max_weight_per_position", "commission_bps", "slippage_bps", "rebalance_drift_tolerance", "max_monthly_position_return", "profile"],
}

# Catálogo de valores admitidos por variable: fuente única en module/scenarios/variables.py, compartida
# con el orquestador de estudios (que deriva de aquí el barrido completo del full_study).
from module.scenarios.variables import COST_STRESS_CASES, STUDY_OPTIONS

# Orden y semántica de las variables que la vista Study puede barrer. Mantiene el diccionario
# ejecutable separado de la presentación y evita una lista plana de decenas de controles.
STUDY_OPTION_GROUPS = {
    "Datos y calendario": ["execution_lag_days", "train_lookback_years", "snapshot_step_months",
                             "fundamental_step_months", "target_horizon_months", "random_seed"],
    "Modelo base": ["objective", "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate",
                      "lgbm_min_child_samples", "recency_weighting"],
    "Meta-agente": ["meta_type", "meta_ic_lookback_quarters", "min_rank_ic_cross_section"],
    "Factores y bloques": ["neutralize_by_sector", "fundamental_momentum", "market_regime_feature",
                             "price_momentum_multi", "moving_averages", "regime_extended",
                             "quality_growth_derived", "enabled_feature_blocks",
                             "metric_winsorization_percentile", "risk_feature_windows",
                             "technical_feature_windows"],
    "Agentes y modelos": ["enabled_agents", "enabled_model_families", "intra_agent_ensemble_mode",
                            "feature_weighting_mode"],
    "Selección de factores": ["feature_selection_min_coverage", "feature_selection_lookback_quarters",
                               "feature_selection_min_permutation_importance",
                               "feature_selection_min_positive_fraction",
                               "feature_selection_max_features_per_agent"],
    "Cartera y perfil": ["target_min", "target_max", "entry_min_percentile", "min_hold_percentile",
                          "rotation_edge_percentiles", "max_weight_per_position", "commission_bps",
                          "slippage_bps", "profile"],
}

EXPERIMENT_PRESETS = {
    "training": [
        {"id": "baseline", "label": "Baseline 2015 · 8 años · reentreno 3m · horizonte 6m", "overrides": {"train_lookback_years": 8, "target_horizon_months": 6, "fundamental_step_months": 3}},
        {"id": "short", "label": "Ventana corta 2015 · 4 años · horizonte 6m", "overrides": {"train_lookback_years": 4, "target_horizon_months": 6, "fundamental_step_months": 3}},
        {"id": "long", "label": "Ventana larga 2015 · 12 años · horizonte 6m", "overrides": {"train_lookback_years": 12, "target_horizon_months": 6, "fundamental_step_months": 6}},
        {"id": "recency", "label": "Baseline + pesos de recencia (exponencial)", "overrides": {"train_lookback_years": 8, "target_horizon_months": 6, "recency_weighting": "exponential"}},
    ],
    "portfolio": [
        {"id": "compact", "label": "Concentrada · 5–8 posiciones · peso máx. 20 %", "overrides": {"target_min": 5, "target_max": 8, "max_weight_per_position": 0.20, "entry_min_percentile": 85, "min_hold_percentile": 55, "rotation_edge_percentiles": 8}},
        {"id": "balanced", "label": "Equilibrada · 8–12 posiciones · peso máx. 15 %", "overrides": {"target_min": 8, "target_max": 12, "max_weight_per_position": 0.15, "entry_min_percentile": 80, "min_hold_percentile": 50, "rotation_edge_percentiles": 5}},
        {"id": "diversified", "label": "Diversificada · 12–15 posiciones · peso máx. 10 %", "overrides": {"target_min": 12, "target_max": 15, "max_weight_per_position": 0.10, "entry_min_percentile": 75, "min_hold_percentile": 45, "rotation_edge_percentiles": 3}},
    ],
    "features": [
        {"id": "base", "label": "Base · sin artefactos", "overrides": {"neutralize_by_sector": False, "fundamental_momentum": False, "market_regime_feature": False, "price_momentum_multi": False, "moving_averages": False, "regime_extended": False, "quality_growth_derived": False}},
        {"id": "price", "label": "Contexto de precio", "overrides": {"price_momentum_multi": True, "moving_averages": True}},
        {"id": "fundamental", "label": "Contexto fundamental", "overrides": {"fundamental_momentum": True, "quality_growth_derived": True}},
        {"id": "regime", "label": "Contexto de régimen", "overrides": {"market_regime_feature": True, "regime_extended": True}},
    ],
}

PROFILE_LABELS = {
    "balanced": "Balanceado · meta-agente puro", "conservative": "Conservador · calidad y valor",
    "aggressive": "Agresivo · momentum", "value": "Value · valoración y calidad",
    "quality": "Calidad · negocio estable", "momentum": "Momentum · fuerza relativa",
    "garp": "GARP · valor, calidad y momentum", "contrarian": "Contrarian · reversión controlada",
}

# Audit trail shown before an official full study.  These are deliberately fixed
# integrity constants rather than optimisation axes.
FULL_STUDY_FIXED_SETTINGS = {
    "data_start_date": Settings().data_start_date,
    "end_date": Settings().end_date,
    "benchmark_ticker": Settings().benchmark_ticker,
    "execution_year": Settings().execution_year,
    "execution_quarter": Settings().execution_quarter,
    "run_scope": Settings().run_scope,
    "max_price_age_days": MAX_PRICE_AGE_DAYS,
    "min_training_rows": MIN_TRAINING_ROWS,
    "recency_halflife_years": RECENCY_HALFLIFE_YEARS,
    "neutralize_min_group": NEUTRALIZE_MIN_GROUP,
    "rebalance_drift_tolerance": Settings().rebalance_drift_tolerance,
    "max_monthly_position_return": Settings().max_monthly_position_return,
    "random_seed": Settings().random_seed,
}
FULL_STUDY_STRESS_SETTINGS = {
    "commission_bps": sorted({case["overrides"]["commission_bps"] for case in COST_STRESS_CASES}),
    "slippage_bps": sorted({case["overrides"]["slippage_bps"] for case in COST_STRESS_CASES}),
}

# Contrato del explorador: son variables existentes en el panel PIT y, salvo los retornos de
# momentum, ratios fundamentales que ya informan la construcción de features/agentes.
STOCK_METRICS = {
    "quality": [
        ("roe", "ROE"), ("roic", "ROIC"), ("net_margin", "Margen neto"),
        ("operating_margin", "Margen operativo"), ("gross_margin", "Margen bruto"),
        ("fcf_margin", "Margen FCF"), ("debt_equity", "Deuda / patrimonio"),
        ("current_ratio", "Ratio corriente"),
    ],
    "growth": [("eps_growth_yoy", "Crecimiento EPS interanual"),
               ("sales_per_share_growth_yoy", "Crecimiento ventas/acción interanual")],
    "valuation": [("pe", "P/E"), ("pb", "P/B"), ("ps", "P/S"), ("ev_ebitda", "EV / EBITDA")],
    "momentum": [("price_return_3m", "Retorno precio 3 meses"),
                 ("price_return_6m", "Retorno precio 6 meses"),
                 ("price_return_12m", "Retorno precio 12 meses")],
}
STOCK_METRIC_LABELS = {key: label for values in STOCK_METRICS.values() for key, label in values}
STOCK_COMPANIONS = {
    "pe": ("eps", "EPS"), "pb": ("book_value", "Valor contable"),
    "ps": ("sales_per_share", "Ventas por acción"), "ev_ebitda": ("ebitda", "EBITDA"),
}
STOCK_RATIO_COLUMNS = tuple(
    key for group, values in STOCK_METRICS.items() if group != "momentum" for key, _ in values
)
STOCK_FACTOR_COLUMNS = {
    "roe": "factor_roe", "roic": "factor_roic", "net_margin": "factor_net_margin",
    "operating_margin": "factor_operating_margin", "gross_margin": "factor_gross_margin",
    "fcf_margin": "factor_fcf_margin", "debt_equity": "factor_debt_equity",
    "current_ratio": "factor_current_ratio", "eps_growth_yoy": "factor_eps_growth_yoy",
    "sales_per_share_growth_yoy": "factor_sales_per_share_growth_yoy", "pe": "factor_pe",
    "pb": "factor_pb", "ps": "factor_ps", "ev_ebitda": "factor_ev_ebitda",
    "price_return_3m": "factor_relative_return_3m", "price_return_6m": "factor_relative_return_6m",
    "price_return_12m": "factor_relative_return_12m",
}


def normalize_index(values: list[object]) -> tuple[list[float | None], bool]:
    """Normaliza una serie a base 100 solo cuando la transformación es interpretable."""
    numeric = [float(value) if pd.notna(value) else None for value in values]
    usable = [value for value in numeric if value is not None]
    if not usable or usable[0] <= 0 or any(value is not None and value <= 0 for value in numeric):
        return numeric, False
    base = usable[0]
    return [None if value is None else value / base * 100 for value in numeric], True


def stock_series(panel: pd.DataFrame, ticker: str, metric: str, start: str = "", end: str = "") -> dict:
    """Prepara una sola serie temporal y su media transversal PIT, sin acceso a processed."""
    if metric not in STOCK_METRIC_LABELS:
        raise ValueError("Ratio no disponible en el explorador.")
    frame = panel.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    if start:
        frame = frame.loc[frame["snapshot_date"] >= pd.Timestamp(start)]
    if end:
        frame = frame.loc[frame["snapshot_date"] <= pd.Timestamp(end)]
    if "in_sp500" in frame:
        universe = frame.loc[frame["in_sp500"].fillna(False)]
    else:
        universe = frame
    averages = universe.groupby("snapshot_date")[metric].agg(["mean", "count"]).reset_index()
    company = frame.loc[frame["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    companion, companion_label = STOCK_COMPANIONS.get(metric, (None, None))
    columns = ["snapshot_date", metric, "price"] + ([companion] if companion else [])
    company = company[columns].merge(averages, on="snapshot_date", how="left")
    price_index, _ = normalize_index(company["price"].tolist())
    companion_values = company[companion].tolist() if companion else []
    companion_series, companion_indexed = normalize_index(companion_values) if companion else ([], False)
    points = []
    for index, row in enumerate(company.itertuples(index=False)):
        point = {
            "snapshot_date": pd.Timestamp(row.snapshot_date).date().isoformat(),
            "value": _finite_or_none(getattr(row, metric)),
            "sp500_mean": _finite_or_none(row.mean), "sp500_observations": int(row.count),
            "price_index": price_index[index],
        }
        if companion:
            point["companion"] = _finite_or_none(getattr(row, companion))
            point["companion_display"] = companion_series[index]
        points.append(point)
    return {
        "ticker": ticker.upper(), "metric": metric, "metric_label": STOCK_METRIC_LABELS[metric],
        "companion": companion, "companion_label": companion_label,
        "companion_indexed": companion_indexed, "points": points,
    }


def _finite_or_none(value: object) -> float | None:
    return float(value) if pd.notna(value) else None


def _settings_from_payload(payload: dict) -> Settings:
    defaults = asdict(Settings())
    allowed = {field.name: field.type for field in fields(Settings)}
    for name, value in payload.get("settings", {}).items():
        if name not in allowed:
            continue
        original = defaults[name]
        normalized = tuple(value) if isinstance(original, tuple) and isinstance(value, list) else value
        if name in STUDY_OPTIONS and normalized not in STUDY_OPTIONS[name]:
            raise ValueError(f"Valor no permitido para {name}: {value!r}.")
        if isinstance(original, bool):
            defaults[name] = bool(value) if not isinstance(value, str) else value.lower() in {"1", "true", "si", "sí"}
        elif isinstance(original, int) and not isinstance(original, bool):
            defaults[name] = int(value)
        elif isinstance(original, float):
            defaults[name] = float(value)
        elif isinstance(original, tuple):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} debe recibirse como una lista de valores.")
            defaults[name] = normalized
        else:
            defaults[name] = value
    return Settings(**defaults)


def _sanitize_json(value):
    """Sustituye NaN/Infinity por None de forma recursiva.

    `json.dumps` los escribe como tokens literales (`NaN`, `Infinity`), que son JSON inválido y
    `JSON.parse` del navegador rechaza (rompía p. ej. la vista Cartera, con NaN en el lifecycle).
    Se sanean aquí, en el punto central, para cubrir todos los endpoints.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _json(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    data = json.dumps(_sanitize_json(payload), ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    size = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(size).decode("utf-8")) if size else {}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TFMDashboard/1.0"

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._static("index.html")
        if parsed.path.startswith("/app/"):
            return self._static(parsed.path.removeprefix("/app/"))
        if parsed.path == "/api/defaults":
            return _json(self, {"settings": asdict(Settings()), "groups": SETTINGS_GROUPS,
                                "study_options": STUDY_OPTIONS, "experiment_presets": EXPERIMENT_PRESETS,
                                "study_option_groups": STUDY_OPTION_GROUPS,
                                "full_study_fixed_settings": FULL_STUDY_FIXED_SETTINGS,
                                "full_study_stress_settings": FULL_STUDY_STRESS_SETTINGS,
                                "profile_labels": PROFILE_LABELS})
        if parsed.path == "/api/runs":
            return _json(self, {"runs": list(reversed(list_registry())), "jobs": JOBS.all()})
        if parsed.path == "/api/studies":
            studies = []
            if STORE.studies_root.exists():
                for directory in sorted(STORE.studies_root.iterdir(), reverse=True):
                    manifest = directory / "study_manifest.json"
                    if manifest.exists():
                        studies.append(json.loads(manifest.read_text(encoding="utf-8")))
            return _json(self, {"studies": studies})
        if parsed.path.startswith("/api/study/"):
            return self._study_detail(parsed.path.removeprefix("/api/study/"))
        if parsed.path.startswith("/api/run/"):
            return self._run_detail(parsed.path.removeprefix("/api/run/"))
        if parsed.path == "/api/meta_weights":
            query = parse_qs(parsed.query)
            return self._table(query.get("run_id", [""])[0], "meta_weights.parquet", query)
        if parsed.path == "/api/ticker":
            query = parse_qs(parsed.query)
            return self._ticker(query.get("run_id", [""])[0], query.get("ticker", [""])[0])
        if parsed.path == "/api/stocks":
            query = parse_qs(parsed.query)
            return self._stocks(query.get("run_id", [""])[0], query.get("query", [""])[0])
        if parsed.path == "/api/stock/summary":
            query = parse_qs(parsed.query)
            return self._stock_summary(query.get("run_id", [""])[0], query.get("ticker", [""])[0])
        if parsed.path == "/api/stock/history":
            query = parse_qs(parsed.query)
            return self._stock_history(query)
        if parsed.path == "/api/stock/agents":
            query = parse_qs(parsed.query)
            return self._stock_agents(query)
        if parsed.path == "/api/ranking":
            query = parse_qs(parsed.query)
            return self._table(query.get("run_id", [""])[0], "agent_scores.parquet", query)
        if parsed.path == "/api/portfolio":
            query = parse_qs(parsed.query)
            return self._portfolio(query.get("run_id", [""])[0], query)
        if parsed.path == "/api/learning":
            query = parse_qs(parsed.query)
            return self._learning(query.get("run_id", [""])[0])
        if parsed.path == "/api/performance":
            query = parse_qs(parsed.query)
            return self._performance(query.get("run_id", [""])[0])
        if parsed.path == "/api/trades":
            query = parse_qs(parsed.query)
            return self._trades(query.get("run_id", [""])[0], query)
        if parsed.path.startswith("/artifacts/"):
            return self._artifact(parsed.path.removeprefix("/artifacts/"))
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = _read_json(self)
            if self.path == "/api/experimental":
                settings = _settings_from_payload(payload)
                mode = str(payload.get("mode", "full"))
                job = JOBS.submit("experimental", lambda: execute_run(
                    settings, mode=mode, label=str(payload.get("label", "Run experimental")),
                    description=str(payload.get("description", "")), tags=payload.get("tags", ()),
                ))
                return _json(self, {"job_id": job}, 202)
            if self.path == "/api/study":
                settings = _settings_from_payload(payload)
                variables = payload.get("variables", {})
                if not isinstance(variables, dict) or not all(isinstance(v, list) and v for v in variables.values()):
                    return _json(self, {"error": "Cada variable del study debe tener una lista de valores."}, 400)
                job = JOBS.submit("study", lambda: execute_study(settings, study_payload=payload.get("study", {}),
                                                                    variables=variables, mode=str(payload.get("mode", "full"))))
                return _json(self, {"job_id": job}, 202)
            if self.path == "/api/optimization":
                settings = _settings_from_payload(payload)
                study = payload.get("study", {})
                if not isinstance(study, dict):
                    raise ValueError("Los metadatos del full study deben ser un objeto.")
                job = JOBS.submit("optimization", lambda: execute_official_optimization(
                    settings, name=str(study.get("name", "optimization-official")),
                    hypothesis=str(study.get("hypothesis", "")),
                ))
                return _json(self, {"job_id": job}, 202)
            return _json(self, {"error": "Ruta no disponible."}, 404)
        except (ValueError, TypeError) as exc:
            return _json(self, {"error": str(exc)}, 400)

    def _run_detail(self, run_id: str) -> None:
        run_dir = _safe_run(run_id)
        manifest = run_dir / "run_manifest.json"
        if not manifest.exists():
            return _json(self, {"error": "Run no encontrado."}, 404)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        summary = run_dir / "artifacts" / "backtest_summary.json"
        payload["summary"] = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
        # Robustez multi-era (bootstrap por bloques + leave-one-year-out): evidencia de primer nivel
        # sobre el aprendizaje. Solo existe en el finalista de un study.
        robustness = run_dir / "artifacts" / "robustness.json"
        payload["robustness"] = json.loads(robustness.read_text(encoding="utf-8")) if robustness.exists() else {}
        payload["artifacts"] = [str(p.relative_to(run_dir / "artifacts")).replace("\\", "/")
                                for p in (run_dir / "artifacts").rglob("*") if p.is_file()]
        _json(self, payload)

    def _study_detail(self, study_id: str) -> None:
        directory = (STORE.studies_root / study_id).resolve()
        try:
            directory.relative_to(STORE.studies_root.resolve())
        except ValueError:
            return _json(self, {"error": "Estudio invalido."}, 404)
        manifest_path = directory / "study_manifest.json"
        if not manifest_path.exists():
            return _json(self, {"error": "Estudio no encontrado."}, 404)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_ids_path = directory / "run_ids.json"
        run_ids = json.loads(run_ids_path.read_text(encoding="utf-8")) if run_ids_path.exists() else {}
        decision_path = directory / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
        # Runs miembros: se toman del registro por study_id (no hace falta abrir cada manifest).
        members = [entry for entry in list_registry() if entry.get("study_id") == study_id]
        # Comparativa por fases: comparison_data.parquet ya lleva la columna `phase` y las métricas
        # por escenario; se expone para que la app agrupe la comparativa por fase.
        comparison_path = directory / "comparison_data.parquet"
        comparison = (pd.read_parquet(comparison_path).to_dict("records")
                      if comparison_path.exists() else [])
        # Los estudios que se publicaron antes de que la comparativa incluyera las fases
        # económicas pueden tener 1--3 en el parquet aunque su decisión sí conserva los
        # runs de cartera y perfiles. Los reconstruimos para que la consola muestre el ciclo
        # completo sin alterar los resultados inmutables del estudio.
        comparison = _complete_phase_comparison(comparison, decision)
        return _json(self, {"manifest": manifest, "run_ids": run_ids.get("run_ids", []),
                            "reused_run_ids": run_ids.get("reused_run_ids", []),
                            "decision": decision, "runs": members, "comparison": comparison})

    def _table(self, run_id: str, name: str, query: dict[str, list[str]]) -> None:
        path = _safe_run(run_id) / "artifacts" / name
        if not path.exists():
            return _json(self, {"columns": [], "rows": [], "total": 0})
        frame = pd.read_parquet(path)
        if "snapshot_date" in query and query["snapshot_date"][0]:
            frame = frame.loc[frame["snapshot_date"].astype(str) == query["snapshot_date"][0]]
        ticker = query.get("ticker", [""])[0].upper()
        if ticker and "ticker" in frame:
            frame = frame.loc[frame["ticker"].astype(str).str.upper().str.contains(ticker, na=False)]
        limit = min(int(query.get("limit", ["500"])[0]), 2000)
        return _json(self, {"columns": list(frame.columns), "rows": frame.head(limit).to_dict("records"), "total": len(frame)})

    def _ticker(self, run_id: str, ticker: str) -> None:
        artifacts = _safe_run(run_id) / "artifacts"
        scores_path = artifacts / "agent_scores.parquet"
        if not scores_path.exists() or not ticker:
            return _json(self, {"scores": [], "positions": [], "orders": []})
        scores = pd.read_parquet(scores_path)
        scores = scores.loc[scores["ticker"].astype(str).str.upper() == ticker.upper()]
        positions_path, orders_path = artifacts / "position_lifecycle.parquet", artifacts / "orders.parquet"
        prices_path = artifacts / "asset_price_point_in_time.parquet"
        positions = pd.read_parquet(positions_path) if positions_path.exists() else pd.DataFrame()
        orders = pd.read_parquet(orders_path) if orders_path.exists() else pd.DataFrame()
        if not positions.empty:
            positions = positions.loc[positions["ticker"].astype(str).str.upper() == ticker.upper()]
        if not orders.empty:
            orders = orders.loc[orders["ticker"].astype(str).str.upper() == ticker.upper()]
            # P&L realizado en las ventas (mismo cálculo que la vista de trades): sin esto el
            # historial de la acción mostraba las ventas sin su ganancia.
            orders = self._attach_realized_pnl(artifacts, orders)
        prices = pd.read_parquet(prices_path) if prices_path.exists() else pd.DataFrame()
        if not prices.empty:
            prices = prices.loc[prices["ticker"].astype(str).str.upper() == ticker.upper()]
        _json(self, {"scores": scores.to_dict("records"), "positions": positions.to_dict("records"),
                     "orders": orders.to_dict("records"), "prices": prices.to_dict("records")})

    def _stock_panel(self, run_id: str) -> pd.DataFrame | None:
        path = _safe_run(run_id) / "artifacts" / "stock_panel.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def _stocks(self, run_id: str, query: str) -> None:
        panel = self._stock_panel(run_id)
        if panel is None:
            return _json(self, {"compatible": False, "tickers": [],
                                "message": "Este run no conserva el panel de stocks. Relánzalo para explorarlo."})
        values = sorted(panel["ticker"].dropna().astype(str).str.upper().unique())
        needle = query.strip().upper()
        if needle:
            values = [ticker for ticker in values if needle in ticker]
        scores_path = _safe_run(run_id) / "artifacts" / "agent_scores.parquet"
        scores = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
        dates = pd.to_datetime(scores.get("snapshot_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()
        # Con búsqueda se devuelven todas las coincidencias (se puede buscar cualquier acción del
        # universo); sin filtro se limita el listado para el datalist inicial.
        shown = values if needle else values[:400]
        _json(self, {"compatible": True, "tickers": shown, "total": len(values),
                     "metrics": STOCK_METRICS,
                     "oos_start": dates.min().date().isoformat() if not dates.empty else None,
                     "oos_end": dates.max().date().isoformat() if not dates.empty else None})

    def _stock_summary(self, run_id: str, ticker: str) -> None:
        panel = self._stock_panel(run_id)
        if panel is None:
            return _json(self, {"compatible": False, "message": "Este run no conserva datos inmutables de stocks."}, 409)
        ticker = ticker.strip().upper()
        if not ticker:
            return _json(self, {"error": "Selecciona un ticker."}, 400)
        scores_path = _safe_run(run_id) / "artifacts" / "agent_scores.parquet"
        scores = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
        scores = scores.loc[scores.get("ticker", pd.Series(dtype=str)).astype(str).str.upper() == ticker].copy()
        panel = panel.loc[panel["ticker"].astype(str).str.upper() == ticker].copy()
        if panel.empty:
            return _json(self, {"compatible": True, "found": False, "ticker": ticker}, 404)
        panel["snapshot_date"] = pd.to_datetime(panel["snapshot_date"])
        latest_score = scores.sort_values("snapshot_date").tail(1)
        snapshot = pd.Timestamp(latest_score.iloc[0]["snapshot_date"]) if not latest_score.empty else panel["snapshot_date"].max()
        row = panel.loc[panel["snapshot_date"] == snapshot].tail(1)
        if row.empty:
            row = panel.loc[panel["snapshot_date"] <= snapshot].tail(1)
        latest = row.iloc[0]
        universe = pd.read_parquet(_safe_run(run_id) / "artifacts" / "stock_panel.parquet")
        universe["snapshot_date"] = pd.to_datetime(universe["snapshot_date"])
        universe = universe.loc[universe["snapshot_date"] == latest.snapshot_date]
        ratios = []
        for metric in STOCK_RATIO_COLUMNS:
            value = _finite_or_none(latest.get(metric))
            cohort = pd.to_numeric(universe.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            percentile = (float((cohort <= value).mean()) if value is not None and not cohort.empty else None)
            ratios.append({"metric": metric, "label": STOCK_METRIC_LABELS[metric], "value": value,
                           "percentile": percentile, "observations": int(len(cohort))})
        score_payload = latest_score.iloc[0].to_dict() if not latest_score.empty else {}
        positions_path = _safe_run(run_id) / "artifacts" / "position_lifecycle.parquet"
        position = {}
        if positions_path.exists():
            positions = pd.read_parquet(positions_path)
            positions = positions.loc[(positions["ticker"].astype(str).str.upper() == ticker)
                                      & (pd.to_datetime(positions["snapshot_date"]) <= latest.snapshot_date)]
            if not positions.empty:
                position = positions.sort_values("snapshot_date").tail(1).iloc[0].to_dict()
        _json(self, {"compatible": True, "found": True, "ticker": ticker,
                     "snapshot_date": latest.snapshot_date.date().isoformat(),
                     "price": _finite_or_none(latest.get("price")), "price_as_of_date": latest.get("price_as_of_date"),
                     "fundamental_period": latest.get("fundamental_period"),
                     "fundamental_filed_date": latest.get("fundamental_filed_date"),
                     "fundamental_age_days": _finite_or_none(latest.get("fundamental_age_days")),
                     "ratios": ratios, "scores": score_payload, "position": position})

    def _stock_history(self, query: dict[str, list[str]]) -> None:
        run_id = query.get("run_id", [""])[0]
        ticker = query.get("ticker", [""])[0]
        metric = query.get("metric", ["pe"])[0]
        panel = self._stock_panel(run_id)
        if panel is None:
            return _json(self, {"compatible": False, "message": "Este run no conserva datos inmutables de stocks."}, 409)
        try:
            payload = stock_series(panel, ticker, metric, query.get("start", [""])[0], query.get("end", [""])[0])
        except (ValueError, TypeError) as exc:
            return _json(self, {"error": str(exc)}, 400)
        _json(self, {"compatible": True, **payload})

    def _stock_agents(self, query: dict[str, list[str]]) -> None:
        run_id = query.get("run_id", [""])[0]
        ticker = query.get("ticker", [""])[0].strip().upper()
        artifacts = _safe_run(run_id) / "artifacts"
        scores_path = artifacts / "agent_scores.parquet"
        if not scores_path.exists() or not ticker:
            return _json(self, {"scores": [], "contributions": [], "weights": [], "global_importance": []})
        scores = pd.read_parquet(scores_path)
        scores = scores.loc[scores["ticker"].astype(str).str.upper() == ticker].sort_values("snapshot_date")
        date = query.get("snapshot_date", [""])[0] or (str(scores.iloc[-1]["snapshot_date"]) if not scores.empty else "")
        attribution_path = artifacts / "agent_local_attribution.parquet"
        attribution = pd.read_parquet(attribution_path) if attribution_path.exists() else pd.DataFrame()
        if not attribution.empty:
            attribution = attribution.loc[(attribution["ticker"].astype(str).str.upper() == ticker)
                                          & (attribution["snapshot_date"].astype(str) == date)]
        weights_path = artifacts / "meta_weights.parquet"
        weights = pd.read_parquet(weights_path) if weights_path.exists() else pd.DataFrame()
        if not weights.empty:
            weights = weights.loc[weights["snapshot_date"].astype(str) == date]
        importance_path = artifacts / "model_feature_attribution.parquet"
        importance = pd.read_parquet(importance_path) if importance_path.exists() else pd.DataFrame()
        if not scores.empty and not importance.empty:
            retrain = str(scores.loc[scores["snapshot_date"].astype(str) == date, "model_retrain_date"].iloc[0])
            importance = importance.loc[importance["model_retrain_date"].astype(str) == retrain]
        contribution_rows = (attribution.sort_values(["agent", "importance_rank"]).to_dict("records")
                             if {"agent", "importance_rank"}.issubset(attribution.columns) else [])
        importance_rows = (importance.sort_values(["agent", "coefficient"], ascending=[True, False]).to_dict("records")
                           if {"agent", "coefficient"}.issubset(importance.columns) else [])
        diagnostics_path = artifacts / "feature_diagnostics.parquet"
        diagnostics = pd.read_parquet(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
        diagnostics_rows = (diagnostics.sort_values("model_importance_mean", ascending=False).to_dict("records")
                            if "model_importance_mean" in diagnostics else [])
        _json(self, {"ticker": ticker, "snapshot_date": date, "scores": scores.to_dict("records"),
                     "selected_scores": scores.loc[scores["snapshot_date"].astype(str) == date].to_dict("records"),
                     "contributions": contribution_rows, "weights": weights.to_dict("records"),
                     "global_importance": importance_rows, "feature_diagnostics": diagnostics_rows})

    def _learning(self, run_id: str) -> None:
        artifacts = _safe_run(run_id) / "artifacts"
        path = artifacts / "rank_ic_diagnostics.parquet"
        if not path.exists():
            return _json(self, {"summary": [], "series": []})
        frame = pd.read_parquet(path)
        # Usa el resumen precomputado en publicación si existe; si no, lo recompone al vuelo.
        precomputed = artifacts / "learning_summary.json"
        if precomputed.exists():
            summary = json.loads(precomputed.read_text(encoding="utf-8")).get("summary", [])
        else:
            summary = []
            for agent, group in frame.groupby("agent"):
                values = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
                summary.append({"agent": agent, "mean_rank_ic": float(values.mean()) if not values.empty else None,
                                "positive_fraction": float((values > 0).mean()) if not values.empty else None,
                                "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                                "n_cohorts": int(len(values))})
        _json(self, {"summary": summary, "series": frame.to_dict("records")})

    def _performance(self, run_id: str) -> None:
        artifacts = _safe_run(run_id) / "artifacts"
        equity = pd.read_parquet(artifacts / "equity.parquet") if (artifacts / "equity.parquet").exists() else pd.DataFrame()
        annual = pd.read_parquet(artifacts / "annual_metrics.parquet") if (artifacts / "annual_metrics.parquet").exists() else pd.DataFrame()
        _json(self, {"equity": equity.to_dict("records"), "annual": annual.to_dict("records")})

    def _portfolio(self, run_id: str, query: dict[str, list[str]]) -> None:
        """Composición de una fecha y desenlace posterior de sus posiciones.

        No usa el límite genérico de tablas: el selector necesita todas las fechas, pero cada
        respuesta solo devuelve las posiciones de una composición.
        """
        artifacts = _safe_run(run_id) / "artifacts"
        lifecycle_path = artifacts / "position_lifecycle.parquet"
        orders_path = artifacts / "orders.parquet"
        if not lifecycle_path.exists():
            return _json(self, {"dates": [], "selected_date": None, "is_latest": True, "rows": []})
        lifecycle = pd.read_parquet(lifecycle_path).copy()
        if lifecycle.empty or "snapshot_date" not in lifecycle:
            return _json(self, {"dates": [], "selected_date": None, "is_latest": True, "rows": []})
        lifecycle["snapshot_date"] = pd.to_datetime(lifecycle["snapshot_date"], errors="coerce")
        lifecycle["entry_date"] = pd.to_datetime(lifecycle["entry_date"], errors="coerce")
        lifecycle = lifecycle.dropna(subset=["snapshot_date"])
        lifecycle["ticker_key"] = lifecycle["ticker"].astype(str).str.upper()
        dates = sorted(lifecycle["snapshot_date"].dt.strftime("%Y-%m-%d").unique().tolist())
        requested = query.get("date", [""])[0]
        selected_date = requested if requested in dates else (dates[-1] if dates else None)
        if not selected_date:
            return _json(self, {"dates": [], "selected_date": None, "is_latest": True, "rows": []})
        selected_at = pd.Timestamp(selected_date)
        selected = lifecycle.loc[lifecycle["snapshot_date"] == selected_at].copy()
        orders = pd.read_parquet(orders_path).copy() if orders_path.exists() else pd.DataFrame()
        if not orders.empty:
            orders["snapshot_date"] = pd.to_datetime(orders["snapshot_date"], errors="coerce")
            orders["ticker_key"] = orders["ticker"].astype(str).str.upper()
        buys = orders.loc[orders.get("side", pd.Series(dtype=str)) == "buy"] if not orders.empty else pd.DataFrame()
        rows: list[dict] = []
        for position in selected.sort_values("weight", ascending=False).to_dict("records"):
            ticker = position["ticker_key"]
            entry_at = pd.Timestamp(position["entry_date"]) if pd.notna(position.get("entry_date")) else selected_at
            reason = None
            if not buys.empty:
                matches = buys.loc[(buys["ticker_key"] == ticker) & (buys["snapshot_date"] == entry_at)]
                if not matches.empty:
                    reason = matches.iloc[-1].get("reason")
            exit_order = pd.DataFrame()
            if not orders.empty:
                weight_after = (pd.to_numeric(orders["weight_after"], errors="coerce").fillna(0)
                                if "weight_after" in orders else pd.Series(0.0, index=orders.index))
                exits = orders.loc[(orders["ticker_key"] == ticker) & (orders["side"] == "sell")
                                   & (orders["snapshot_date"] > selected_at)
                                   & (weight_after <= 1e-12)]
                exit_order = exits.sort_values("snapshot_date").head(1)
            if not exit_order.empty:
                exit_at = exit_order.iloc[0]["snapshot_date"]
                final_price = exit_order.iloc[0].get("price")
                months_held = max(0, round((exit_at - entry_at).days / 30.4375))
            else:
                exit_at, final_price = None, None
                same_position = lifecycle.loc[(lifecycle["ticker_key"] == ticker)
                                              & (lifecycle["entry_date"] == entry_at)
                                              & (lifecycle["snapshot_date"] >= selected_at)].sort_values("snapshot_date")
                if not same_position.empty:
                    last_state = same_position.iloc[-1]
                    final_price = last_state.get("valuation_price")
                    months_held = last_state.get("months_held")
                else:
                    months_held = position.get("months_held")
            entry_price = position.get("entry_price")
            pnl = None
            if pd.notna(entry_price) and pd.notna(final_price) and float(entry_price) != 0:
                pnl = (float(final_price) / float(entry_price) - 1) * 100
            rows.append({
                "snapshot_date": selected_date, "ticker": position.get("ticker"),
                "entry_date": entry_at.strftime("%Y-%m-%d"),
                "entry_price": entry_price, "reason": reason, "weight": position.get("weight"),
                "months_held": months_held,
                "exit_date": exit_at.strftime("%Y-%m-%d") if exit_at is not None else None,
                "final_price": final_price, "pnl_pct": pnl,
            })
        _json(self, {"dates": dates, "selected_date": selected_date,
                     "is_latest": selected_date == dates[-1], "rows": rows})

    def _trades(self, run_id: str, query: dict[str, list[str]]) -> None:
        artifacts = _safe_run(run_id) / "artifacts"
        path = artifacts / "orders.parquet"
        if not path.exists():
            return _json(self, {"rows": [], "summary": {}, "years": []})
        frame = pd.read_parquet(path)
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        # P&L realizado por venta: cada venta se empareja con el precio de entrada de esa posición
        # (de position_lifecycle). Retorno = precio_venta / precio_entrada - 1.
        frame = self._attach_realized_pnl(artifacts, frame)
        years = sorted({int(d.year) for d in frame["snapshot_date"].dropna()})

        # --- Filtros: año, rango de fechas, ticker ---
        year = query.get("year", [""])[0]
        if year:
            frame = frame.loc[frame["snapshot_date"].dt.year == int(year)]
        start = query.get("start", [""])[0]
        end = query.get("end", [""])[0]
        if start:
            frame = frame.loc[frame["snapshot_date"] >= pd.Timestamp(start)]
        if end:
            frame = frame.loc[frame["snapshot_date"] <= pd.Timestamp(end)]
        ticker = query.get("ticker", [""])[0].upper()
        if ticker:
            frame = frame.loc[frame["ticker"].astype(str).str.upper().str.contains(ticker, na=False)]

        sells = frame.loc[frame["side"] == "sell"]
        realized = pd.to_numeric(sells.get("realized_return_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        summary = {
            "orders": int(len(frame)), "buys": int((frame["side"] == "buy").sum()),
            "sells": int(len(sells)),
            "commission": float(frame.get("commission", pd.Series(dtype=float)).sum()),
            "slippage": float(frame.get("slippage", pd.Series(dtype=float)).sum()),
            "sells_with_gain": int((realized > 0).sum()), "sells_with_loss": int((realized < 0).sum()),
            "avg_realized_return_pct": float(realized.mean()) if not realized.empty else None,
            "total_gain_pct": float(realized[realized > 0].sum()) if not realized.empty else 0.0,
            "total_loss_pct": float(realized[realized < 0].sum()) if not realized.empty else 0.0,
        }
        _json(self, {"rows": frame.head(2000).to_dict("records"), "summary": summary, "years": years})

    @staticmethod
    def _attach_realized_pnl(artifacts: Path, orders: pd.DataFrame) -> pd.DataFrame:
        """Añade realized_return_pct a las ventas usando el precio de entrada del lifecycle."""
        lifecycle_path = artifacts / "position_lifecycle.parquet"
        orders = orders.copy()
        orders["realized_return_pct"] = None
        if not lifecycle_path.exists() or "price" not in orders:
            return orders
        lifecycle = pd.read_parquet(lifecycle_path)
        if "entry_price" not in lifecycle or lifecycle.empty:
            return orders
        # Precio de entrada más reciente por ticker (una posición se mantiene hasta que se vende).
        entries = lifecycle.dropna(subset=["entry_price"]).sort_values("snapshot_date").drop_duplicates("ticker", keep="last")
        entry_by_ticker = {str(t).upper(): p for t, p in zip(entries["ticker"], entries["entry_price"])}
        sell_mask = orders["side"] == "sell"
        for idx in orders.loc[sell_mask].index:
            ticker = str(orders.at[idx, "ticker"]).upper()
            entry = entry_by_ticker.get(ticker)
            price = orders.at[idx, "price"]
            if entry and pd.notna(price) and float(entry) > 0:
                orders.at[idx, "realized_return_pct"] = (float(price) / float(entry) - 1.0) * 100.0
        return orders

    def _artifact(self, relative: str) -> None:
        candidate = (RESULTS_ROOT / relative).resolve()
        try:
            candidate.relative_to(RESULTS_ROOT.resolve())
        except ValueError:
            return self.send_error(HTTPStatus.FORBIDDEN)
        if not candidate.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        content = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _static(self, relative: str) -> None:
        """Sirve un archivo del frontend bajo app/ (raíz) con guardia de path-traversal."""
        candidate = (APP_ROOT / relative).resolve()
        try:
            candidate.relative_to(APP_ROOT)
        except ValueError:
            return self.send_error(HTTPStatus.FORBIDDEN)
        if not candidate.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") or "javascript" in content_type else ""))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def _safe_run(run_id: str) -> Path:
    candidate = (STORE.runs_root / run_id).resolve()
    try:
        candidate.relative_to(STORE.runs_root.resolve())
    except ValueError as exc:
        raise FileNotFoundError("Run invalido") from exc
    return candidate


def _complete_phase_comparison(comparison: list[dict], decision: dict) -> list[dict]:
    """Añade Fase 4 y 5 a comparativas históricas cuando están en ``decision.json``."""
    present = {(str(row.get("phase")), str(row.get("run_id")))
               for row in comparison if row.get("run_id")}
    additions: list[dict] = []

    portfolio = decision.get("phase_portfolio", {}).get("axes", {})
    for axis, detail in portfolio.items():
        for candidate in detail.get("candidates", []):
            run_id = candidate.get("run_id")
            value = candidate.get("value")
            key = ("4_cartera", str(run_id))
            if run_id and key not in present:
                additions.append(_comparison_row(str(run_id), "4_cartera", f"{axis}={value}", axis, {axis: value}))
                present.add(key)

    for profile, run_id in decision.get("profile_run_ids", {}).items():
        key = ("5_perfiles", str(run_id))
        if run_id and key not in present:
            additions.append(_comparison_row(str(run_id), "5_perfiles", str(profile), "profile", {"profile": profile}))
            present.add(key)
    return [*comparison, *additions]


def _comparison_row(run_id: str, phase: str, scenario: str, axis: str, overrides: dict) -> dict:
    """Construye una fila de comparativa desde el resumen publicado de un run."""
    artifacts = _safe_run(run_id) / "artifacts"
    summary_path = artifacts / "backtest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {"phase": phase, "scenario": scenario, "axis": axis, "overrides": overrides,
            "run_id": run_id, **summary}


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Consola TFM disponible en http://{host}:{port}")
    server.serve_forever()
