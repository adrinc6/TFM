"""Fixtures visuales del dashboard; nunca ejecutan ciencia."""

from __future__ import annotations

from typing import Any

from module.studies.catalog import PROFILE_NAMES, public_catalog


STUDY_ID = "study-demo-001"


def get(path: str, query: dict[str, list[str]]) -> dict[str, Any] | list[dict[str, Any]]:
    if path == "/api/catalog":
        return public_catalog()
    if path == "/api/studies":
        return [_study()]
    if path == f"/api/studies/{STUDY_ID}":
        return {**_study(), "runs": _runs(), "decisions": _decisions(), "winner": _winner()}
    if path == f"/api/studies/{STUDY_ID}/runs":
        return _runs()
    if path.startswith(f"/api/studies/{STUDY_ID}/runs/"):
        run_id = path.rsplit("/", 1)[-1]
        return next(row for row in _runs() if row["run_id"] == run_id)
    if path == f"/api/studies/{STUDY_ID}/events":
        return _events()
    prefix = f"/api/studies/{STUDY_ID}/analysis/"
    if path.startswith(prefix):
        return _analysis(path.removeprefix(prefix), query)
    raise FileNotFoundError("Fixture no encontrada.")


def post(path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if path == "/api/studies/preflight":
        return {
            "valid": True,
            "definition": payload.get("definition", {}),
            "budget": {
                "predictive_evaluations": 16, "expensive_fits": 9,
                "meta_recombinations": 3, "portfolio_diagnostics": 8,
                "profiles": 8, "robustness_groups": 10,
                "total_runs": 42, "estimated_minutes": 210,
                "estimated_incremental_bytes": 1_400_000_000,
            },
            "run_scope": "dev",
        }, 200
    if path == "/api/studies":
        return {"study_id": STUDY_ID, "status": "queued", "worker_pid": 12345}, 202
    if path.endswith("/cancel"):
        return {**_study(), "status": "cancelled"}, 200
    if path.endswith("/resume"):
        return {**_study(), "status": "queued"}, 202
    raise FileNotFoundError("Fixture no encontrada.")


def _study() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID, "name": "Model Study demo", "study_type": "model_study",
        "status": "succeeded", "phase": "completed", "progress": 1.0,
        "completed_runs": 42, "created_at": "2026-07-25T10:00:00+00:00",
        "updated_at": "2026-07-25T12:30:00+00:00", "heartbeat": "2026-07-25T12:30:00+00:00",
        "winner_run_id": "run-meta-bounded", "budget": {"total_runs": 42},
    }


def _runs() -> list[dict[str, Any]]:
    return [
        {
            "run_id": "run-baseline", "logical_key": "predictive:baseline",
            "phase": "temporal", "variable_id": "baseline", "value": "baseline",
            "status": "succeeded", "progress": 1.0, "elapsed_seconds": 123.4,
            "result": {"summary": {"mean_rank_ic": 0.041, "rank_ic_positive_fraction": 0.59, "mean_annual_alpha": 0.023, "stress_mean_alpha": -0.014}},
        },
        {
            "run_id": "run-lag-45", "logical_key": "predictive:execution_lag_days:45",
            "phase": "temporal", "variable_id": "execution_lag_days", "value": 45,
            "status": "succeeded", "progress": 1.0, "elapsed_seconds": 118.0,
            "result": {"summary": {"mean_rank_ic": 0.047, "rank_ic_positive_fraction": 0.63, "mean_annual_alpha": 0.031, "stress_mean_alpha": -0.008}},
        },
        {
            "run_id": "run-meta-bounded", "logical_key": "predictive:meta:bounded",
            "phase": "meta", "variable_id": "meta_method", "value": "stacked_rolling_bounded",
            "status": "succeeded", "progress": 1.0, "elapsed_seconds": 22.1,
            "result": {"summary": {"mean_rank_ic": 0.052, "rank_ic_positive_fraction": 0.66, "mean_annual_alpha": 0.038, "stress_mean_alpha": 0.006}},
        },
    ]


def _decisions() -> list[dict[str, Any]]:
    return [{
        "variable_id": "execution_lag_days", "winner_value": 45,
        "selection_rule": "robust_rank_ic", "selection_metric": "rank_ic_only",
        "known_stress_excluded": True,
        "candidates": [
            {"value": 45, "eligible": True, "median_era_rank_ic": 0.046, "reason": "Elegible y mayor mediana por eras."},
            {"value": 60, "eligible": True, "median_era_rank_ic": 0.041, "reason": "Elegible, no seleccionado."},
        ],
    }]


def _winner() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID, "winner_run_id": "run-meta-bounded",
        "selection_metric": "rank_ic_only", "known_stress_role": "known_stress_not_selection",
        "configuration": {"execution_lag_days": 45, "meta_method": "stacked_rolling_bounded"},
        "summary": {"summary": {"mean_rank_ic": 0.052, "rank_ic_positive_fraction": 0.66}},
    }


def _events() -> list[dict[str, Any]]:
    messages = (
        "Study creado.", "Baseline iniciado.", "Fit quality completado.",
        "Lag 45 seleccionado por Rank-IC.", "Ganador congelado.",
        "Robustez terminada.", "Ocho perfiles terminados.", "Study terminado correctamente.",
    )
    return [
        {"sequence": index, "timestamp": f"2026-07-25T10:{index:02d}:00+00:00", "level": "info", "event": "demo", "message": message}
        for index, message in enumerate(messages, 1)
    ]


def _analysis(view: str, query: dict[str, list[str]]) -> dict[str, Any]:
    if view == "winner":
        return _winner()
    if view == "learning":
        return {
            "rank_ic": [{"agent": agent, "prediction_date": f"{year}-12-31", "rank_ic": 0.02 + ((year + index) % 4) * 0.01} for index, agent in enumerate(("quality", "value", "growth", "momentum", "risk", "meta_final")) for year in range(2015, 2025)],
            "tail": [], "weights": [{"snapshot_date": f"{year}-12-31", "agent": agent, "weight": 0.12 + ((year + index) % 5) * 0.04} for index, agent in enumerate(("quality", "value", "growth", "momentum", "risk")) for year in range(2015, 2025)],
            "features": [], "attribution": [],
        }
    if view == "robustness":
        return {"permutation": {"p_value": 0.018, "n_permutations": 9999}, "seeds": [{"seed": 7, "mean_rank_ic": 0.049}, {"seed": 2026, "mean_rank_ic": 0.046}], "dev_only_not_scientific": False}
    if view == "profiles":
        comparison = [{"profile": profile, "mean_annual_alpha": 0.01 + index * 0.002} for index, profile in enumerate(PROFILE_NAMES)]
        annual = [{"year": year, "profile": profile, "alpha": ((year + index) % 5 - 2) / 100} for index, profile in enumerate(PROFILE_NAMES) for year in range(2019, 2025)]
        return {"comparison": comparison, "annual": annual}
    if view == "portfolio-comparisons":
        return {"rows": [{"variable_id": "target_size", "base_value": 12, "diagnostic_value": 8, "information_ratio": 0.42, "mean_annual_alpha": 0.018}]}
    if view == "portfolio":
        snapshots = [f"{year}-12-31" for year in range(2018, 2025)]
        selected = query.get("snapshot", [snapshots[-1]])[0]
        return {"summary": {"summary": {"cagr_portfolio": 0.12, "cagr_benchmark": 0.10, "cagr_difference": 0.02}}, "equity": _equity(), "annual": [], "available_snapshots": snapshots, "selected_snapshot": selected, "positions": [{"snapshot_date": selected, "ticker": "MSFT", "weight": 0.12}], "orders": [{"snapshot_date": selected, "ticker": "NVDA", "side": "buy", "reason": "edge_over_worst"}]}
    if view == "stocks":
        snapshots = [f"{year}-12-31" for year in range(2018, 2025)]
        tickers = ["AAPL", "MSFT", "NVDA"]
        selected = query.get("snapshot", [snapshots[-1]])[0]
        ticker = query.get("ticker", [""])[0]
        if not ticker:
            return {"available_tickers": tickers, "available_snapshots": snapshots, "selected_snapshot": selected}
        parameter = query.get("parameter", ["factor_roe"])[0]
        options = [{"id": "factor_roe", "label": "roe", "kind": "puntuación"}, {"id": "roe", "label": "roe", "kind": "valor PIT"}]
        return {"ticker": ticker, "available_tickers": tickers, "available_snapshots": snapshots, "selected_snapshot": selected, "scores": [{"quality": 0.71, "value": 0.58, "growth": 0.63, "momentum": 0.79, "risk": 0.52, "meta_score": 0.67, "meta_rank": 0.87}], "parameter_scores": [{"factor_roe": 0.74}], "parameter_values": [{"roe": 0.18}], "positions": [{"snapshot_date": selected, "ticker": ticker, "weight": 0.12}], "orders": [], "parameter_options": options, "selected_parameter": parameter if parameter in {"factor_roe", "roe"} else "factor_roe", "parameter_history": [{"snapshot_date": value, parameter if parameter in {"factor_roe", "roe"} else "factor_roe": 0.05 + index * 0.02} for index, value in enumerate(snapshots)]}
    return {"markdown": "# Informe demo\n\nLa selección usa únicamente Rank-IC."}


def _equity() -> list[dict[str, Any]]:
    return [{"snapshot_date": f"{year}-12-31", "portfolio_value": 100 * 1.12 ** (year - 2018), "benchmark_value": 100 * 1.10 ** (year - 2018)} for year in range(2018, 2025)]
