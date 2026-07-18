"""Consola web local, sin dependencias externas, para runs y resultados."""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from dataclasses import asdict, fields, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from environment import PROJECT_ROOT, Settings
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
    "Periodo y entrenamiento": ["execution_year", "execution_quarter", "execution_lag_days", "train_lookback_years", "snapshot_step_months", "fundamental_step_months", "snapshot_day", "target_horizon_months", "max_price_age_days", "meta_ic_lookback_quarters"],
    "Modelo LightGBM": ["objective", "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type"],
    "Artefactos": ["neutralize_by_sector", "neutralize_min_group", "fundamental_momentum", "market_regime_feature", "price_momentum_multi", "moving_averages", "regime_extended", "quality_growth_derived"],
    "Cartera y perfil": ["target_min", "target_max", "entry_min_percentile", "min_hold_percentile", "rotation_edge_percentiles", "max_weight_per_position", "commission_bps", "slippage_bps", "rebalance_drift_tolerance", "max_monthly_position_return", "profile"],
}

STUDY_OPTIONS = {
    "execution_year": [2012, 2014, 2016, 2018, 2020], "execution_quarter": [1, 2, 3, 4],
    "execution_lag_days": [15, 30, 45, 60], "train_lookback_years": [5, 6, 7, 8, 10, 12],
    "snapshot_step_months": [1, 3], "fundamental_step_months": [3, 6, 12],
    "target_horizon_months": [1, 3, 6, 12], "objective": ["rank_regression", "ranking", "quartile"],
    "lgbm_n_estimators": [100, 200, 400], "lgbm_max_depth": [3, 4, 5, 6],
    "lgbm_learning_rate": [0.03, 0.05, 0.10], "lgbm_min_child_samples": [20, 50, 100],
    "meta_type": ["equal", "rank_ic", "regime"],
    "neutralize_by_sector": [False, True], "fundamental_momentum": [False, True],
    "market_regime_feature": [False, True], "price_momentum_multi": [False, True],
    "moving_averages": [False, True], "regime_extended": [False, True],
    "quality_growth_derived": [False, True], "target_min": [6, 8, 10, 12],
    "target_max": [8, 10, 12, 15], "entry_min_percentile": [70, 80, 90],
    "min_hold_percentile": [40, 50, 60], "rotation_edge_percentiles": [3, 5, 10],
    "max_weight_per_position": [0.10, 0.15, 0.20], "commission_bps": [0, 5, 10],
    "slippage_bps": [5, 10, 20], "profile": ["balanced", "conservative", "aggressive", "value", "quality", "momentum", "garp", "contrarian"],
}

EXPERIMENT_PRESETS = {
    "training": [
        {"id": "baseline", "label": "Baseline 2016 · 10 años · 3 meses", "overrides": {"execution_year": 2016, "train_lookback_years": 10, "target_horizon_months": 3, "fundamental_step_months": 3}},
        {"id": "early", "label": "Era temprana 2012 · 10 años · 3 meses", "overrides": {"execution_year": 2012, "train_lookback_years": 10, "target_horizon_months": 3, "fundamental_step_months": 3}},
        {"id": "medium", "label": "Ventana media 2016 · 8 años · 3 meses", "overrides": {"execution_year": 2016, "train_lookback_years": 8, "target_horizon_months": 3, "fundamental_step_months": 3}},
        {"id": "long", "label": "Ventana larga 2016 · 12 años · 6 meses", "overrides": {"execution_year": 2016, "train_lookback_years": 12, "target_horizon_months": 6, "fundamental_step_months": 6}},
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
        if name in STUDY_OPTIONS and value not in STUDY_OPTIONS[name] and str(value) not in {str(v) for v in STUDY_OPTIONS[name]}:
            raise ValueError(f"Valor no permitido para {name}: {value!r}.")
        original = defaults[name]
        if isinstance(original, bool):
            defaults[name] = bool(value) if not isinstance(value, str) else value.lower() in {"1", "true", "si", "sí"}
        elif isinstance(original, int) and not isinstance(original, bool):
            defaults[name] = int(value)
        elif isinstance(original, float):
            defaults[name] = float(value)
        else:
            defaults[name] = value
    return Settings(**defaults)


def _json(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
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
            return self._table(query.get("run_id", [""])[0], "position_lifecycle.parquet", query)
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
                job = JOBS.submit("optimization", lambda: execute_official_optimization(settings))
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
        return _json(self, {"manifest": manifest, "run_ids": run_ids.get("run_ids", []),
                            "reused_run_ids": run_ids.get("reused_run_ids", []),
                            "decision": decision, "runs": members})

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
        _json(self, {"compatible": True, "tickers": values[:100], "total": len(values),
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
        _json(self, {"ticker": ticker, "snapshot_date": date, "scores": scores.to_dict("records"),
                     "selected_scores": scores.loc[scores["snapshot_date"].astype(str) == date].to_dict("records"),
                     "contributions": contribution_rows, "weights": weights.to_dict("records"),
                     "global_importance": importance_rows})

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

    def _trades(self, run_id: str, query: dict[str, list[str]]) -> None:
        artifacts = _safe_run(run_id) / "artifacts"
        path = artifacts / "orders.parquet"
        if not path.exists():
            return _json(self, {"rows": [], "summary": {}})
        frame = pd.read_parquet(path)
        ticker = query.get("ticker", [""])[0].upper()
        if ticker:
            frame = frame.loc[frame["ticker"].astype(str).str.upper().str.contains(ticker, na=False)]
        summary = {"orders": int(len(frame)), "buys": int((frame["side"] == "buy").sum()),
                   "sells": int((frame["side"] == "sell").sum()),
                   "commission": float(frame.get("commission", pd.Series(dtype=float)).sum()),
                   "slippage": float(frame.get("slippage", pd.Series(dtype=float)).sum())}
        _json(self, {"rows": frame.head(2000).to_dict("records"), "summary": summary})

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


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Consola TFM disponible en http://{host}:{port}")
    server.serve_forever()

