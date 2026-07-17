"""Features point-in-time, etiquetas futuras separadas y baselines."""

from __future__ import annotations

import logging

import pandas as pd

from environment import Settings
from module.baselines import build_baseline_scores
from module.dataset import snapshot_dates
from module.utils import read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)

FACTOR_SOURCES = {
    "roe": ("factor_roe", True, False),
    "roic": ("factor_roic", True, False),
    "net_margin": ("factor_net_margin", True, False),
    "operating_margin": ("factor_operating_margin", True, False),
    "gross_margin": ("factor_gross_margin", True, False),
    "fcf_margin": ("factor_fcf_margin", True, False),
    "debt_equity": ("factor_debt_equity", False, False),
    "current_ratio": ("factor_current_ratio", True, False),
    "eps_growth_yoy": ("factor_eps_growth_yoy", True, False),
    "sales_per_share_growth_yoy": ("factor_sales_per_share_growth_yoy", True, False),
    "pe": ("factor_pe", False, True),
    "pb": ("factor_pb", False, True),
    "ps": ("factor_ps", False, True),
    "ev_ebitda": ("factor_ev_ebitda", False, True),
}


def build_features(settings: Settings) -> pd.DataFrame:
    """Construye factores observables, baselines y targets en artefactos distintos."""
    output_dir = settings.processed_output_dir
    panel = read_parquet(output_dir / "panel_point_in_time.parquet", "RUN_MODE='dataset'")
    benchmark = read_parquet(output_dir / "benchmark_point_in_time.parquet", "RUN_MODE='dataset'")
    asset_prices = read_parquet(output_dir / "asset_price_point_in_time.parquet", "RUN_MODE='dataset'")
    if benchmark.empty:
        raise RuntimeError(
            f"El benchmark {settings.benchmark_ticker} no tiene precios PIT. "
            "Ejecuta RUN_MODE='download' y después RUN_MODE='dataset' con el mismo alcance."
        )

    features = _build_feature_frame(panel, benchmark, settings)
    targets = _build_targets(features, asset_prices, benchmark, settings)
    baselines = build_baseline_scores(features)

    write_parquet(features, output_dir / "features_point_in_time.parquet")
    write_parquet(targets, output_dir / "targets_forward_3m.parquet")
    write_parquet(baselines, output_dir / "baseline_scores.parquet")
    write_json(_coverage(features, baselines, settings), output_dir / "features_coverage.json")
    log.info(
        "Features: rows=%s fresh_rows=%s targets=%s output=%s",
        len(features),
        int(features["is_price_fresh"].sum()),
        int(targets["target_available"].sum()),
        output_dir,
    )
    return features


def _build_feature_frame(panel: pd.DataFrame, benchmark: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    required_panel = {"ticker", "snapshot_date", "in_sp500", "price_age_days", "price_return_3m"}
    required_benchmark = {"snapshot_date", "price_age_days", "price_return_3m"}
    if missing := required_panel - set(panel.columns):
        raise ValueError(f"panel_point_in_time.parquet no contiene: {sorted(missing)}")
    if missing := required_benchmark - set(benchmark.columns):
        raise ValueError(f"benchmark_point_in_time.parquet no contiene: {sorted(missing)}")

    bench = benchmark.rename(
        columns={
            "price": "benchmark_price",
            "price_as_of_date": "benchmark_price_as_of_date",
            "price_age_days": "benchmark_price_age_days",
            "price_return_1m": "benchmark_return_1m",
            "price_return_3m": "benchmark_return_3m",
            "price_return_6m": "benchmark_return_6m",
            "price_return_12m": "benchmark_return_12m",
        }
    )
    frame = panel.merge(bench, on="snapshot_date", how="left", validate="many_to_one")
    frame["price_age_days"] = pd.to_numeric(frame["price_age_days"], errors="coerce")
    frame["benchmark_price_age_days"] = pd.to_numeric(
        frame["benchmark_price_age_days"], errors="coerce"
    )
    frame["is_price_fresh"] = (
        frame["in_sp500"].fillna(False).astype(bool)
        & frame["price_age_days"].le(settings.max_price_age_days)
        & frame["benchmark_price_age_days"].le(settings.max_price_age_days)
    )

    for horizon in (3, 6, 12):
        stock = pd.to_numeric(frame[f"price_return_{horizon}m"], errors="coerce")
        index = pd.to_numeric(frame[f"benchmark_return_{horizon}m"], errors="coerce")
        relative = stock - index
        frame[f"relative_return_{horizon}m"] = relative.where(frame["is_price_fresh"])
        frame[f"factor_relative_return_{horizon}m"] = _cross_section_rank(
            frame, frame[f"relative_return_{horizon}m"], ascending=True
        )

    for source, (factor, ascending, positive_only) in FACTOR_SOURCES.items():
        values = pd.to_numeric(frame[source], errors="coerce")
        if positive_only:
            values = values.where(values.gt(0))
        values = values.where(frame["is_price_fresh"])
        frame[factor] = _cross_section_rank(frame, values, ascending=ascending)

    frame.sort_values(["snapshot_date", "ticker"], inplace=True, ignore_index=True)
    return frame


def _cross_section_rank(frame: pd.DataFrame, values: pd.Series, ascending: bool) -> pd.Series:
    ranked = values.groupby(frame["snapshot_date"]).rank(method="average", pct=True, ascending=ascending)
    return ranked.where(frame["is_price_fresh"])


def _label_dates(settings: Settings) -> dict[str, str]:
    """Fecha de etiqueta de cada snapshot: el que está N posiciones más adelante en la rejilla.

    La fecha NO se deriva sumando meses. La rejilla clampa los fines de mes con
    `min(snapshot_day, days_in_month)` y `DateOffset` lo hace con otra regla distinta, así que
    con `SNAPSHOT_DAY = 31` había snapshots cuya etiqueta caía en un día inexistente: el merge
    no encontraba pareja y el target se degradaba a NaN sin ningún error, perdiendo ~40 % del
    entrenamiento en silencio. Tomando la rejilla como única fuente de verdad, la fecha de
    etiqueta existe por construcción, sea cual sea `SNAPSHOT_DAY`.

    `SNAPSHOT_DAY` es uno de los parámetros que barre la Fase 6: si la cobertura de etiquetas
    dependiese de su valor, la rejilla compararía escenarios con distinta cantidad de datos.
    Ver `docs/plan_fases.md` (Hallazgo 2).
    """
    grid = [date.date().isoformat() for date in snapshot_dates(settings)]
    step = settings.target_horizon_months // settings.snapshot_step_months
    return {date: grid[index + step] for index, date in enumerate(grid[:-step or None])}


def _build_targets(
    features: pd.DataFrame,
    asset_prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    base = features[["ticker", "snapshot_date", "price", "price_as_of_date", "is_price_fresh", "benchmark_price"]].copy()
    base["snapshot_ts"] = pd.to_datetime(base["snapshot_date"])
    base["future_snapshot_date"] = base["snapshot_date"].map(_label_dates(settings))

    future_assets = asset_prices.rename(
        columns={
            "snapshot_date": "future_snapshot_date",
            "price": "future_price",
            "price_as_of_date": "future_price_as_of_date",
            "price_age_days": "future_price_age_days",
        }
    )
    target = base.merge(
        future_assets,
        on=["ticker", "future_snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    future_benchmark = benchmark.rename(
        columns={
            "snapshot_date": "future_snapshot_date",
            "price": "future_benchmark_price",
            "price_as_of_date": "future_benchmark_as_of_date",
            "price_age_days": "future_benchmark_age_days",
        }
    )[["future_snapshot_date", "future_benchmark_price", "future_benchmark_as_of_date", "future_benchmark_age_days"]]
    target = target.merge(future_benchmark, on="future_snapshot_date", how="left", validate="many_to_one")
    target["future_price_age_days"] = pd.to_numeric(target["future_price_age_days"], errors="coerce")
    target["future_benchmark_age_days"] = pd.to_numeric(
        target["future_benchmark_age_days"], errors="coerce"
    )
    valid = (
        target["is_price_fresh"].fillna(False)
        & target["future_price_age_days"].le(settings.max_price_age_days)
        & target["future_benchmark_age_days"].le(settings.max_price_age_days)
        & pd.to_numeric(target["price"], errors="coerce").gt(0)
        & pd.to_numeric(target["benchmark_price"], errors="coerce").gt(0)
    )
    stock_return = target["future_price"] / target["price"] - 1
    benchmark_return = target["future_benchmark_price"] / target["benchmark_price"] - 1
    target["forward_return_3m"] = stock_return.where(valid)
    target["forward_benchmark_return_3m"] = benchmark_return.where(valid)
    target["forward_excess_return_3m"] = (stock_return - benchmark_return).where(valid)
    label_dates = pd.concat(
        [
            pd.to_datetime(target["future_price_as_of_date"]),
            pd.to_datetime(target["future_benchmark_as_of_date"]),
        ],
        axis=1,
    ).max(axis=1)
    target["label_end_date"] = label_dates.dt.date.astype("string")
    target["target_available"] = valid & target["forward_excess_return_3m"].notna()
    return target[
        [
            "ticker",
            "snapshot_date",
            "future_snapshot_date",
            "label_end_date",
            "target_available",
            "forward_return_3m",
            "forward_benchmark_return_3m",
            "forward_excess_return_3m",
        ]
    ]


def _coverage(features: pd.DataFrame, baselines: pd.DataFrame, settings: Settings) -> dict:
    factor_columns = [factor for factor, _, _ in FACTOR_SOURCES.values()]
    factor_columns.extend(["factor_relative_return_3m", "factor_relative_return_6m", "factor_relative_return_12m"])
    return {
        "run_scope": settings.run_scope,
        "max_price_age_days": settings.max_price_age_days,
        "rows": len(features),
        "fresh_rows": int(features["is_price_fresh"].sum()),
        "factor_rows": {column: int(features[column].notna().sum()) for column in factor_columns},
        "baseline_rows": {
            "garp": int(baselines["garp_score"].notna().sum()),
            "momentum": int(baselines["momentum_score"].notna().sum()),
        },
    }
