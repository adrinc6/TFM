"""Diagnósticos point-in-time de la cola operada y salud de la señal."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from module.modeling.targets import normalize_target_columns


def rank_tail_diagnostics(scores: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Métricas por cohorte con el mismo peso estadístico para cada fecha."""
    target_frame = normalize_target_columns(targets)
    columns = [
        "ticker", "snapshot_date", "label_end_date", "target_available", "forward_excess_return",
    ]
    available = [column for column in columns if column in target_frame]
    merged = scores.merge(
        target_frame[available].drop_duplicates(["ticker", "snapshot_date"]),
        on=["ticker", "snapshot_date"], how="inner", validate="one_to_one",
    )
    merged = merged.loc[merged["target_available"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for date, cohort in merged.groupby("snapshot_date", sort=True):
        usable = cohort[["ticker", "meta_rank", "meta_score", "forward_excess_return"]].dropna()
        if len(usable) < 10:
            continue
        ordered = usable.sort_values("meta_rank", ascending=False)
        top_n = ordered.head(min(10, len(ordered)))
        decile_size = max(1, int(math.ceil(len(ordered) * 0.10)))
        top_decile = ordered.head(decile_size)
        bottom_decile = ordered.tail(decile_size)
        universe_mean = float(usable["forward_excess_return"].mean())
        top_decile_mean = float(top_decile["forward_excess_return"].mean())
        top_10_mean = float(top_n["forward_excess_return"].mean())
        within_top = top_n["meta_score"].corr(top_n["forward_excess_return"], method="spearman")
        full_ic = usable["meta_rank"].corr(usable["forward_excess_return"], method="spearman")
        rows.append({
            "prediction_date": str(date),
            "is_quarterly": bool(cohort["is_quarterly"].fillna(False).iloc[0])
            if "is_quarterly" in cohort else False,
            "label_end_date": str(cohort["label_end_date"].dropna().max())
            if "label_end_date" in cohort and cohort["label_end_date"].notna().any() else None,
            "observations": int(len(usable)),
            "top_n": int(len(top_n)),
            "rank_ic": float(full_ic) if pd.notna(full_ic) else None,
            "top_10_excess_mean": top_10_mean,
            "top_decile_excess_mean": top_decile_mean,
            "universe_excess_mean": universe_mean,
            "top_decile_minus_universe": top_decile_mean - universe_mean,
            "top_minus_bottom": top_decile_mean - float(bottom_decile["forward_excess_return"].mean()),
            "top_10_within_spearman": float(within_top) if pd.notna(within_top) else None,
            "top_10_coverage": float(len(top_n) / len(usable)),
        })
    return pd.DataFrame(rows)


def signal_health_path(
    tail_diagnostics: pd.DataFrame,
    snapshot_dates: Iterable[str],
    *,
    lookback_quarters: int = 12,
    minimum_cohorts: int = 8,
    prior_cohorts: int = 8,
) -> pd.DataFrame:
    """Ruta causal de salud: solo usa cohortes trimestrales cerradas antes del snapshot."""
    history = tail_diagnostics.copy()
    if history.empty:
        return pd.DataFrame([
            {"snapshot_date": str(date), "closed_cohorts": 0, "shrunk_rank_ic": None,
             "shrunk_tail_spread": None}
            for date in snapshot_dates
        ])
    history["prediction_ts"] = pd.to_datetime(history["prediction_date"])
    history["label_end_ts"] = pd.to_datetime(history["label_end_date"])
    if "is_quarterly" not in history:
        raise ValueError("Los diagnósticos deben materializar is_quarterly.")
    quarterly = history.loc[history["is_quarterly"].fillna(False)].sort_values(
        "prediction_ts"
    )
    rows: list[dict[str, object]] = []
    for raw_date in snapshot_dates:
        date = pd.Timestamp(raw_date)
        closed = quarterly.loc[quarterly["label_end_ts"].le(date)].tail(lookback_quarters)
        n = len(closed)
        if n < minimum_cohorts:
            shrunk_ic = shrunk_tail = None
        else:
            factor = n / (n + prior_cohorts)
            shrunk_ic = factor * float(closed["rank_ic"].mean())
            shrunk_tail = factor * float(closed["top_decile_minus_universe"].mean())
        rows.append({
            "snapshot_date": str(pd.Timestamp(raw_date).date()),
            "closed_cohorts": int(n),
            "shrunk_rank_ic": shrunk_ic,
            "shrunk_tail_spread": shrunk_tail,
        })
    return pd.DataFrame(rows)


def calibrated_alpha_path(
    scores: pd.DataFrame, targets: pd.DataFrame, *, history_quarters: int = 16,
    half_life_quarters: float = 8.0, prior_cohorts: int = 8, ventiles: int = 20,
) -> pd.DataFrame:
    """Calibración isotónica causal de rank a retorno excedente esperado.

    Mientras no haya ``prior_cohorts`` cohortes cerradas el valor devuelto es ``NaN``, no ``0.0``:
    son cosas distintas y la cartera las trata distinto. ``0.0`` significa "se espera un alfa nulo",
    lo que activaría ventas y bloquearía compras durante todo el arranque; ``NaN`` significa "todavía
    no hay evidencia calibrada", y en ese caso la cartera decide por ordenación.
    """
    target_frame = normalize_target_columns(targets)
    labelled = scores.merge(
        target_frame[[
            "ticker", "snapshot_date", "label_end_date", "target_available",
            "forward_excess_return",
        ]].drop_duplicates(["ticker", "snapshot_date"]),
        on=["ticker", "snapshot_date"], how="left", validate="one_to_one",
    )
    labelled["prediction_ts"] = pd.to_datetime(labelled["snapshot_date"])
    labelled["label_end_ts"] = pd.to_datetime(labelled["label_end_date"])
    quarterly_mask = (
        labelled["is_quarterly"].fillna(False)
        if "is_quarterly" in labelled else pd.Series(False, index=labelled.index)
    )
    quarterly = labelled.loc[quarterly_mask].copy()
    rows: list[pd.DataFrame] = []
    for raw_date, current in labelled.groupby("snapshot_date", sort=True):
        date = pd.Timestamp(raw_date)
        closed = quarterly.loc[
            quarterly["target_available"].fillna(False)
            & quarterly["label_end_ts"].le(date)
            & quarterly["prediction_ts"].lt(date)
        ].copy()
        closed_dates = sorted(closed["prediction_ts"].dropna().unique())[-history_quarters:]
        closed = closed.loc[closed["prediction_ts"].isin(closed_dates)]
        expected = np.full(len(current), np.nan, dtype=float)
        if len(closed_dates) >= prior_cohorts:
            closed["ventile"] = np.minimum(
                (pd.to_numeric(closed["meta_rank"], errors="coerce") * ventiles).fillna(0)
                .astype(int), ventiles - 1,
            )
            cohort_means = (
                closed.groupby(["prediction_ts", "ventile"], as_index=False)
                ["forward_excess_return"].mean()
            )
            date_order = {value: index for index, value in enumerate(closed_dates)}
            cohort_means["age"] = (
                len(closed_dates) - 1 - cohort_means["prediction_ts"].map(date_order)
            )
            cohort_means["weight"] = 0.5 ** (
                cohort_means["age"] / float(half_life_quarters)
            )
            bucket_rows: list[tuple[int, float, int]] = []
            for bucket in range(ventiles):
                group = cohort_means.loc[cohort_means["ventile"] == bucket]
                if group.empty:
                    bucket_rows.append((bucket, 0.0, 0))
                    continue
                mean = float(np.average(group["forward_excess_return"], weights=group["weight"]))
                n = int(len(group))
                bucket_rows.append((bucket, mean * n / (n + prior_cohorts), n))
            x = np.arange(ventiles, dtype=float)
            y = np.asarray([item[1] for item in bucket_rows], dtype=float)
            try:
                from sklearn.isotonic import IsotonicRegression
                fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(x, y)
            except Exception:
                fitted = np.maximum.accumulate(y)
            current_buckets = np.minimum(
                (pd.to_numeric(current["meta_rank"], errors="coerce") * ventiles).fillna(0)
                .astype(int).to_numpy(), ventiles - 1,
            )
            expected = fitted[current_buckets]
        part = current[["ticker", "snapshot_date"]].copy()
        part["expected_excess_return"] = expected
        part["calibration_closed_quarters"] = len(closed_dates)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

