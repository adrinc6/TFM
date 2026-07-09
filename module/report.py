"""Performance metrics and the final report entry point.

The single professional HTML report now lives in module/viewer (results/<run>/viewer/index.html).
This module keeps the vetted metric functions (`_metrics`, `drawdown_episodes`, reused by the
viewer) and `build_final_report`, which builds the viewer report so the `report` pipeline stage
stays functional and points at the one report instead of emitting a second, redundant HTML page.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from environment import PROCESSED_DIR, Settings
from module.backtest.artifacts import SMALL_SAMPLE_CAVEAT, excess_return_statistics

log = logging.getLogger(__name__)


def build_final_report(settings: Settings) -> Path:
    """The final report is the single viewer report; build it and return its path."""
    from module.viewer import build_viewer

    viewer_dir = build_viewer(settings)
    path = viewer_dir / "index.html"
    log.info("Final report is the viewer report at %s", path)
    return path


def drawdown_episodes(vs: pd.DataFrame) -> pd.DataFrame:
    if vs.empty or "portfolio_value" not in vs.columns or "date" not in vs.columns:
        return pd.DataFrame()
    values = vs["portfolio_value"].reset_index(drop=True)
    dates = pd.to_datetime(vs["date"]).reset_index(drop=True)
    running_max = values.cummax()
    drawdown = values / running_max - 1
    episodes = []
    in_drawdown = False
    peak_index = 0
    for i in range(len(values)):
        if drawdown[i] < 0 and not in_drawdown:
            in_drawdown = True
            peak_index = i - 1 if i > 0 else 0
        is_last = i == len(values) - 1
        if in_drawdown and (drawdown[i] >= 0 or is_last):
            window = drawdown[peak_index:i + 1]
            trough_offset = int(window.values.argmin())
            trough_index = peak_index + trough_offset
            recovered = drawdown[i] >= 0
            episodes.append({
                "peak_date": dates[peak_index].date().isoformat(),
                "trough_date": dates[trough_index].date().isoformat(),
                "recovery_date": dates[i].date().isoformat() if recovered else "",
                "depth": float(drawdown[trough_index]),
                "duration_days": int((dates[i] - dates[peak_index]).days),
                "recovered": bool(recovered),
            })
            in_drawdown = False
    return pd.DataFrame(episodes).sort_values("depth").head(10) if episodes else pd.DataFrame()


def _metrics(vs: pd.DataFrame) -> dict:
    if vs.empty:
        return {"CAGR": 0, "Sharpe": 0, "Sortino": 0, "Max Drawdown": 0, "Alpha": 0}
    returns = vs.get("portfolio_period_return", pd.Series(dtype=float)).fillna(0)
    benchmark_returns = vs.get("benchmark_period_return", pd.Series(dtype=float)).fillna(0)
    years = max((pd.to_datetime(vs["date"]).max() - pd.to_datetime(vs["date"]).min()).days / 365.25, 1 / 12)
    ending = float(vs["portfolio_value"].iloc[-1])
    gross_ending = float(vs["portfolio_gross_value"].iloc[-1]) if "portfolio_gross_value" in vs.columns else ending
    benchmark_ending = float(vs.get("benchmark_value", pd.Series([1])).iloc[-1])
    downside = returns[returns < 0]
    stats = excess_return_statistics(vs)
    return {
        "CAGR": ending ** (1 / years) - 1,
        "Gross CAGR": gross_ending ** (1 / years) - 1,
        "Benchmark CAGR": benchmark_ending ** (1 / years) - 1,
        "Sharpe": _annualized_ratio(returns),
        "Sortino": _annualized_ratio(returns, downside_only=True),
        "Max Drawdown": _max_drawdown(vs["portfolio_value"]),
        "Alpha": (ending - 1) - (benchmark_ending - 1),
        "Gross Alpha": (gross_ending - 1) - (benchmark_ending - 1),
        "Total Cost Drag": float(vs.get("transaction_cost_drag", pd.Series([0])).sum()),
        "Average Period Alpha": float((returns - benchmark_returns).mean()),
        "Information Ratio": stats["information_ratio"],
        "Tracking Error (annualized)": stats["tracking_error_annualized"],
        "Excess Return t-stat": stats["excess_return_t_stat"],
        "Periods (n)": stats["periods_n"],
    }


def _annualized_ratio(returns: pd.Series, downside_only: bool = False) -> float:
    series = returns[returns < 0] if downside_only else returns
    denom = float(series.std())
    if denom == 0 or pd.isna(denom):
        return 0.0
    return float(returns.mean() / denom * (12 ** 0.5))


def _max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return float(drawdown.min())
