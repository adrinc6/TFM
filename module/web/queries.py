"""Consultas del dashboard para el único Model Study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from module.evaluation.profiles import PROFILE_NAMES
from module.evaluation.signal_diagnostics import alpha_curve_points
from module.storage.datasets import validate_dataset_reference
from module.storage.studies import (
    list_runs, list_studies, read_events, read_run, read_study, safe_study_path,
)


ANALYSIS_VIEWS = {
    "winner", "learning", "robustness", "profiles", "portfolio-comparisons",
    "portfolio", "stocks", "report", "attribution",
}

# Columnas de `positions.parquet` que sí se muestran: `market_value`, `entry_cost` y `units` son
# cifras absolutas sin contexto (sin `%` de cartera, sin comparabilidad entre posiciones); `weight`
# y `current_percentile` ya dicen lo mismo de forma comparable. `entry_price` y `valuation_price`
# sí se muestran porque dan contexto de compra/valoración; `unrealized_pnl_pct` se deriva de ambas.
POSITION_COLUMNS = [
    "ticker", "snapshot_date", "weight", "entry_date", "entry_price", "valuation_price",
    "unrealized_pnl_pct", "months_held", "current_percentile", "meta_rank",
]

ORDER_COLUMNS = [
    "ticker", "snapshot_date", "side", "reason", "weight_before", "weight_after",
    "buy_price", "sell_price", "realized_pnl_pct", "notional", "commission_amount",
    "slippage_amount",
]

AGENT_SCORE_COLUMNS = ["quality", "value", "growth", "momentum", "risk", "meta_score", "meta_rank"]

# Ratios cuyos componentes en bruto también están en `panel_point_in_time.parquet`, así que se
# pueden graficar por separado junto al ratio. El resto de ratios (ev_ebitda, debt_equity, ...) no
# tienen sus componentes como columna PIT propia y se muestran sin descomposición.
RATIO_COMPONENTS = {
    "pe": ("price", "eps"),
    "pb": ("price", "book_value"),
    "ps": ("price", "sales_per_share"),
    "pfcf": ("price", "fcf_per_share"),
}


def studies() -> list[dict[str, Any]]:
    rows = []
    for study in list_studies():
        run_rows = list_runs(study["study_id"])
        completed = int(study.get("completed_runs", 0))
        budgeted = int(study.get("budget", {}).get("total_runs", 0))
        rank_values = [
            float(run["result"]["summary"]["mean_rank_ic"])
            for run in run_rows
            if (run.get("result") or {}).get("summary", {}).get("mean_rank_ic") is not None
        ]
        rows.append({
            **study,
            "max_rank_ic": max(rank_values) if rank_values else None,
            "runs_total": budgeted,
            # El presupuesto cuenta también recombinaciones metodológicas sin un run físico. Una
            # vez finalizado el Study no queda trabajo pendiente aunque ambas cifras no coincidan.
            "runs_remaining": 0 if study.get("status") == "succeeded" else max(0, budgeted - completed),
        })
    return rows


def study_detail(study_id: str) -> dict[str, Any]:
    payload = read_study(study_id)
    directory = safe_study_path(study_id)
    payload["runs"] = list_runs(study_id)
    payload["winner"] = _json(directory / "winner.json")
    payload["decisions"] = _json(directory / "decisions.json").get("decisions", [])
    payload["storage"] = _json(directory / "storage_manifest.json")
    return payload


def run_detail(study_id: str, run_id: str) -> dict[str, Any]:
    return read_run(study_id, run_id)


def events(study_id: str, after: int) -> list[dict[str, Any]]:
    return read_events(study_id, after)


def _evidence_dir(directory: Path, study_id: str, source: str | None) -> Path:
    """Resuelve qué evidencia lee una vista analítica.

    ``baseline`` y el ganador son las dos rutas históricas. Con la ruta científica completa cada
    run guarda la suya, y se direcciona como ``run:<run_id>``: el directorio se toma del propio
    artefacto del run, nunca de la cadena recibida, y se confina bajo el Study.
    """
    if source == "baseline":
        return directory / "evidence_baseline"
    if source and source.startswith("run:"):
        run = read_run(study_id, source.removeprefix("run:"))
        relative = run.get("evidence_path")
        if not relative:
            raise FileNotFoundError("Ese run no conserva evidencia propia.")
        resolved = (directory / relative).resolve()
        resolved.relative_to(directory.resolve())
        return resolved
    return directory / "evidence"


def analysis(study_id: str, view: str, query: dict[str, list[str]]) -> dict[str, Any]:
    if view not in ANALYSIS_VIEWS:
        raise ValueError("Vista analítica desconocida.")
    directory = safe_study_path(study_id)
    source = query.get("source", [None])[0]
    evidence = _evidence_dir(directory, study_id, source)
    profile = query.get("profile", [None])[0]
    profile_dir = evidence / "profiles" / profile if profile else evidence
    if view == "winner":
        return _json(directory / "winner.json")
    if view == "learning":
        return {
            "rank_ic": _parquet(evidence / "rank_ic_diagnostics.parquet"),
            "tail": _parquet(evidence / "rank_tail_diagnostics.parquet"),
            "weights": _parquet(evidence / "meta_weights.parquet"),
            "features": _parquet(evidence / "feature_diagnostics.parquet", limit=2_000),
            "attribution": _parquet(evidence / "model_feature_attribution.parquet", limit=2_000),
        }
    if view == "robustness":
        return _json(directory / "robustness.json")
    if view == "attribution":
        return _json(directory / "attribution.json")
    if view == "profiles":
        rows = _parquet(directory / "profile_comparison.parquet")
        annual = []
        for name in PROFILE_NAMES:
            for row in _parquet(evidence / "profiles" / name / "annual_metrics.parquet"):
                annual.append({"profile": name, **row})
        return {"comparison": rows, "annual": annual}
    if view == "portfolio-comparisons":
        return {"rows": _parquet(directory / "portfolio_comparison.parquet")}
    if view == "portfolio":
        positions = _with_unrealized_pnl(_read_frame(profile_dir / "positions.parquet"))
        orders = _read_frame(profile_dir / "orders.parquet")
        equity = _read_frame(profile_dir / "equity.parquet")
        snapshots = _snapshots(positions, orders)
        selected_snapshot = _selected_snapshot(snapshots, query.get("snapshot", [None])[0])
        position_rows = _project_columns(_records_at_snapshot(positions, selected_snapshot), POSITION_COLUMNS)
        return {
            "summary": _json(profile_dir / "summary.json"),
            "equity": _parquet(profile_dir / "equity.parquet"),
            "annual": _parquet(profile_dir / "annual_metrics.parquet"),
            "available_snapshots": snapshots,
            "selected_snapshot": selected_snapshot,
            "positions": _with_cash_row(position_rows, equity, selected_snapshot),
            "orders": _project_columns(_records_at_snapshot(orders, selected_snapshot), ORDER_COLUMNS),
            "alpha_curve": _alpha_curve(evidence, selected_snapshot),
        }
    if view == "stocks":
        ticker = query.get("ticker", [""])[0].strip().upper()
        return _stock(evidence, ticker, query)
    return {"markdown": (directory / "report.md").read_text(encoding="utf-8") if (directory / "report.md").exists() else ""}


def _project_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in columns if key in row} for row in rows]


def _with_cash_row(
    positions: list[dict[str, Any]], equity: pd.DataFrame, snapshot: str | None,
) -> list[dict[str, Any]]:
    """Antepone una fila sintética `$$CASH$$` con el peso en efectivo del snapshot.

    `positions.parquet` no incluye el efectivo (sus pesos solo suman `invested_weight`, nunca
    1.0 — ver `docs/gestion_cartera.md` sección G), así que se toma de `equity.parquet`, la
    única fuente que ya lo calcula por snapshot.
    """
    cash_rows = _records_at_snapshot(equity, snapshot)
    cash_weight = cash_rows[0].get("cash_weight") if cash_rows else None
    if cash_weight is None:
        return positions
    return [{"ticker": "$$CASH$$", "snapshot_date": snapshot, "weight": cash_weight}, *positions]


def _with_unrealized_pnl(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or "entry_price" not in positions.columns or "valuation_price" not in positions.columns:
        return positions
    positions = positions.copy()
    positions["unrealized_pnl_pct"] = positions["valuation_price"] / positions["entry_price"] - 1.0
    return positions


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # `.where(cond, None)` por sí solo no basta: en una columna float64 pandas reconvierte el
    # `None` de vuelta a NaN, que `json.dumps` serializa como el token `NaN` (JSON inválido, y
    # `JSON.parse` del navegador lo rechaza). `.astype(object)` primero evita esa reconversión.
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records") if not frame.empty else []


def _stock(evidence: Path, ticker: str, query: dict[str, list[str]]) -> dict[str, Any]:
    reference = _json(evidence / "dataset_reference.json")
    prepared = validate_dataset_reference(str(reference["dataset_hash"]))
    panel = pd.read_parquet(prepared / "panel_point_in_time.parquet")
    features = pd.read_parquet(prepared / "features_point_in_time.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    positions = _read_frame(evidence / "positions.parquet")
    orders = _read_frame(evidence / "orders.parquet")
    tickers = sorted(scores["ticker"].dropna().astype(str).unique().tolist())
    value_columns = [
        column for column in panel.columns
        if column not in {"ticker", "snapshot_date", "review_type", "in_sp500", "fundamental_period", "fundamental_filed_date", "price"}
        and pd.api.types.is_numeric_dtype(panel[column])
    ]
    ratio_options = [
        {"id": column, "label": column, "components": list(RATIO_COMPONENTS.get(column, ()))}
        for column in value_columns
    ]
    base = {"available_tickers": tickers, "ratio_options": ratio_options}
    if not ticker:
        return base
    if ticker not in tickers:
        raise ValueError("Ticker no disponible en el Study.")
    view = query.get("view", ["scores"])[0]
    if view == "scores":
        return {**base, "ticker": ticker, **_stock_scores(scores, panel, ticker)}
    if view == "portfolio":
        return {**base, "ticker": ticker, **_stock_portfolio(scores, positions, orders, ticker)}
    if view == "ratios-summary":
        return {**base, "ticker": ticker, **_stock_ratios_summary(scores, features, panel, ticker, query.get("snapshot", [None])[0])}
    if view == "ratio-detail":
        return {**base, "ticker": ticker, **_stock_ratio_detail(panel, ticker, query.get("ratio", [None])[0])}
    raise ValueError("Vista de acción desconocida.")


def _stock_scores(scores: pd.DataFrame, panel: pd.DataFrame, ticker: str) -> dict[str, Any]:
    ticker_scores = scores.loc[scores["ticker"].eq(ticker), ["snapshot_date", *AGENT_SCORE_COLUMNS]].dropna(subset=["snapshot_date"])
    ticker_price = panel.loc[panel["ticker"].eq(ticker), ["snapshot_date", "price"]]
    history = ticker_scores.merge(ticker_price, on="snapshot_date", how="left").sort_values("snapshot_date")
    history_records = _records(history)
    return {"current": history_records[-1] if history_records else {}, "history": history_records}


def _stock_portfolio(scores: pd.DataFrame, positions: pd.DataFrame, orders: pd.DataFrame, ticker: str) -> dict[str, Any]:
    ticker_scores = scores.loc[scores["ticker"].eq(ticker), ["snapshot_date", "meta_rank"]].copy()
    ticker_scores["percentile"] = pd.to_numeric(ticker_scores["meta_rank"], errors="coerce") * 100
    ticker_positions = positions.loc[positions.get("ticker", pd.Series(dtype=str)).eq(ticker)].sort_values("snapshot_date") if not positions.empty else positions
    ticker_orders = orders.loc[orders.get("ticker", pd.Series(dtype=str)).eq(ticker)].sort_values("snapshot_date") if not orders.empty else orders
    events = ticker_orders.merge(ticker_scores[["snapshot_date", "percentile"]], on="snapshot_date", how="left") if not ticker_orders.empty else ticker_orders
    position_records = _records(ticker_positions)
    last_position = position_records[-1] if position_records else None
    current_status = {
        "in_portfolio": last_position is not None,
        "weight": last_position.get("weight") if last_position else None,
        "percentile": last_position.get("current_percentile") if last_position else None,
        "entry_date": last_position.get("entry_date") if last_position else None,
        "months_held": last_position.get("months_held") if last_position else None,
        "as_of": last_position.get("snapshot_date") if last_position else None,
    }
    return {
        "status": current_status,
        "history": _project_columns(position_records, ["snapshot_date", "weight", "current_percentile"]),
        "events": _records(events),
    }


def _stock_ratios_summary(
    scores: pd.DataFrame, features: pd.DataFrame, panel: pd.DataFrame, ticker: str, snapshot: str | None,
) -> dict[str, Any]:
    snapshots = sorted(scores["snapshot_date"].dropna().astype(str).unique().tolist())
    selected_snapshot = _selected_snapshot(snapshots, snapshot)
    feature_row = features.loc[features["ticker"].eq(ticker) & features["snapshot_date"].astype(str).eq(selected_snapshot)]
    factor_columns = [column for column in feature_row.columns if column.startswith("factor_")]
    panel_row = panel.loc[panel["ticker"].eq(ticker) & panel["snapshot_date"].astype(str).eq(selected_snapshot)]
    value_columns = [
        column for column in panel_row.columns
        if column not in {"ticker", "snapshot_date", "review_type", "in_sp500", "fundamental_period", "fundamental_filed_date"}
        and pd.api.types.is_numeric_dtype(panel_row[column])
    ]
    return {
        "available_snapshots": snapshots,
        "selected_snapshot": selected_snapshot,
        "scores": _records(feature_row[factor_columns]),
        "values": _records(panel_row[value_columns]),
    }


def _stock_ratio_detail(panel: pd.DataFrame, ticker: str, ratio: str | None) -> dict[str, Any]:
    ticker_panel = panel.loc[panel["ticker"].eq(ticker)].sort_values("snapshot_date")
    if not ratio or ratio not in ticker_panel.columns:
        return {"ratio": None, "history": [], "components": [], "price_history": []}
    history = ticker_panel[["snapshot_date", ratio]].dropna()
    components = [
        {"id": component, "history": ticker_panel[["snapshot_date", component]].dropna().to_dict("records")}
        for component in RATIO_COMPONENTS.get(ratio, ())
        if component in ticker_panel.columns
    ]
    return {
        "ratio": ratio,
        "current_value": history[ratio].iloc[-1] if not history.empty else None,
        "history": history.to_dict("records"),
        "components": components,
        "price_history": ticker_panel[["snapshot_date", "price"]].dropna().to_dict("records"),
    }


def _alpha_curve(evidence: Path, snapshot: str | None) -> dict[str, Any]:
    """Curva decil -> alfa real anualizado en `snapshot`, por cada ventana de la cascada.

    Devuelve `{}` cuando falta cualquier pieza (estudio sin scores, dataset preparado ausente): la
    vista es un diagnóstico añadido, y su ausencia no debe tumbar la pestaña de cartera entera.
    """
    scores_path = evidence / "agent_scores.parquet"
    reference_path = evidence / "dataset_reference.json"
    if not snapshot or not scores_path.exists() or not reference_path.exists():
        return {}
    try:
        prepared = validate_dataset_reference(str(_json(reference_path)["dataset_hash"]))
        targets_path = prepared / "targets_forward.parquet"
        if not targets_path.exists():
            return {}
        scores = pd.read_parquet(scores_path)
        scores = scores.loc[scores["snapshot_date"].astype(str).le(snapshot)]
        if scores.empty:
            return {}
        horizon = int(_json(evidence / "manifest.json").get("config", {}).get("target_horizon_months", 12))
        windows = alpha_curve_points(
            scores, pd.read_parquet(targets_path), horizon_months=horizon,
        )
    except (KeyError, ValueError, FileNotFoundError):
        return {}
    return {"snapshot_date": snapshot, "horizon_months": horizon, "windows": windows}


def _snapshots(*frames: pd.DataFrame) -> list[str]:
    values = {
        str(value) for frame in frames if not frame.empty and "snapshot_date" in frame.columns
        for value in frame["snapshot_date"].dropna()
    }
    return sorted(values)


def _selected_snapshot(snapshots: list[str], requested: str | None) -> str | None:
    if requested in snapshots:
        return requested
    return snapshots[-1] if snapshots else None


def _records_at_snapshot(frame: pd.DataFrame, snapshot: str | None) -> list[dict[str, Any]]:
    if frame.empty or not snapshot or "snapshot_date" not in frame.columns:
        return []
    selected = frame.loc[frame["snapshot_date"].astype(str).eq(snapshot)]
    return selected.where(pd.notna(selected), None).to_dict("records")


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _parquet(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    frame = _read_frame(path)
    if limit is not None:
        frame = frame.tail(limit)
    return frame.where(pd.notna(frame), None).to_dict("records") if not frame.empty else []


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
