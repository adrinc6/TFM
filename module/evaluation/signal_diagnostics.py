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


# Ventiles, no percentiles: con ~500 tickers por snapshot, 20 buckets dejan ~25 acciones en cada uno,
# suficiente para que la media signifique algo. Con 100 percentiles serían ~5 acciones por punto y la
# recta se ajustaría sobre ruido, sobre todo en la ventana `horizon` (4 cohortes).
VENTILES = 20
# Ventanas de la cascada, en trimestres cerrados. "horizon" se resuelve por snapshot a partir de
# `target_horizon_months`; el resto son fijas. El orden es el de la cascada: de lo más reactivo a lo
# más estable, y solo se cae al siguiente escalón cuando la pendiente no es creciente.
ERA_QUARTERS = 16
# Salvaguarda: recta impuesta cuando NINGUNA ventana con datos produce pendiente creciente. Va de
# -10 % anual en el ventil peor a +10 % anual en el mejor. Es un supuesto a priori, no evidencia: se
# activa justo cuando los datos dicen lo contrario, y por eso queda registrado en la columna
# `alpha_curve_window` de cada fila para que el informe pueda contarlo.
FALLBACK_ANNUAL_ALPHA = 0.10
CURVE_WINDOWS = ("horizon", "era", "history")


def _annualize_excess(values: pd.Series, horizon_months: int) -> pd.Series:
    """Lleva el excedente del horizonte de la etiqueta a base anual, componiendo.

    Con horizonte 12 es la identidad. `1 + r` puede ser negativo si la acción cayó más de un 100 %
    frente al benchmark; elevar eso a una potencia fraccionaria da NaN, así que se recorta por
    debajo en -0.999 antes de componer.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if horizon_months == 12:
        return numeric
    growth = (1.0 + numeric).clip(lower=1e-3)
    return growth ** (12.0 / float(horizon_months)) - 1.0


def _ventile_points(closed: pd.DataFrame, half_life_quarters: float) -> pd.DataFrame:
    """Media por ventil de `meta_rank`, ponderando cada cohorte por su antigüedad.

    Devuelve una fila por ventil con datos (`ventile`, `alpha_annual`, `observations`). Un ventil sin
    ninguna observación no se inventa: se omite, y la recta se ajusta sobre los que sí existen.
    """
    if closed.empty:
        return pd.DataFrame(columns=["ventile", "alpha_annual", "observations"])
    frame = closed.copy()
    frame["ventile"] = _ventile_of(frame["meta_rank"])
    cohort_dates = sorted(frame["prediction_ts"].dropna().unique())
    order = {value: index for index, value in enumerate(cohort_dates)}
    age = len(cohort_dates) - 1 - frame["prediction_ts"].map(order).astype(float)
    frame["cohort_weight"] = 0.5 ** (age / float(half_life_quarters))
    rows: list[dict[str, float]] = []
    for ventile, group in frame.groupby("ventile", sort=True):
        usable = group.dropna(subset=["alpha_annual", "cohort_weight"])
        if usable.empty:
            continue
        rows.append({
            "ventile": int(ventile),
            "alpha_annual": float(np.average(usable["alpha_annual"], weights=usable["cohort_weight"])),
            "observations": int(len(usable)),
        })
    return pd.DataFrame(rows)


def _ventile_of(meta_rank: pd.Series) -> np.ndarray:
    """Ventil (0..VENTILES-1) de cada `meta_rank`, que ya viene normalizado en [0, 1]."""
    return np.minimum(
        (pd.to_numeric(meta_rank, errors="coerce") * VENTILES).fillna(0).astype(int).to_numpy(),
        VENTILES - 1,
    )


def _fit_line(points: pd.DataFrame) -> tuple[float, float] | None:
    """Recta alfa ~ ventil por mínimos cuadrados. None si no hay dos ventiles distintos."""
    if len(points) < 2 or points["ventile"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(
        points["ventile"].to_numpy(dtype=float), points["alpha_annual"].to_numpy(dtype=float), 1,
    )
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return None
    return float(slope), float(intercept)


def _fallback_line() -> tuple[float, float]:
    """Recta impuesta de -FALLBACK_ANNUAL_ALPHA (ventil 0) a +FALLBACK_ANNUAL_ALPHA (último)."""
    slope = 2.0 * FALLBACK_ANNUAL_ALPHA / (VENTILES - 1)
    return slope, -FALLBACK_ANNUAL_ALPHA


def alpha_curve_points(
    scores: pd.DataFrame, targets: pd.DataFrame, *, horizon_months: int,
    half_life_quarters: float = 8.0,
) -> dict[str, dict[str, object]]:
    """Puntos por decil y recta ajustada de cada ventana, para el último snapshot de `scores`.

    Pensado para el dashboard: devuelve, por ventana, los ventiles observados y la recta, de modo que
    la vista pueda dibujar los puntos reales y compararlos con la salvaguarda.
    """
    labelled = _labelled_for_curve(scores, targets, horizon_months)
    if labelled.empty:
        return {}
    date = pd.Timestamp(max(labelled["snapshot_date"]))
    quarterly = _closed_quarterly(labelled)
    result: dict[str, dict[str, object]] = {}
    for window in CURVE_WINDOWS:
        closed = _window_history(quarterly, date, window, horizon_months)
        points = _ventile_points(closed, half_life_quarters)
        line = _fit_line(points)
        result[window] = {
            "points": points.to_dict("records"),
            "slope": None if line is None else line[0],
            "intercept": None if line is None else line[1],
            "cohorts": int(closed["prediction_ts"].nunique()) if not closed.empty else 0,
        }
    slope, intercept = _fallback_line()
    result["fallback"] = {"points": [], "slope": slope, "intercept": intercept, "cohorts": 0}
    return result


def _labelled_for_curve(
    scores: pd.DataFrame, targets: pd.DataFrame, horizon_months: int,
) -> pd.DataFrame:
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
    labelled["alpha_annual"] = _annualize_excess(labelled["forward_excess_return"], horizon_months)
    return labelled


def _closed_quarterly(labelled: pd.DataFrame) -> pd.DataFrame:
    mask = (
        labelled["is_quarterly"].fillna(False)
        if "is_quarterly" in labelled else pd.Series(False, index=labelled.index)
    )
    return labelled.loc[mask & labelled["target_available"].fillna(False)].copy()


def _window_history(
    quarterly: pd.DataFrame, date: pd.Timestamp, window: str, horizon_months: int,
) -> pd.DataFrame:
    """Cohortes cerradas antes de `date` que entran en la ventana pedida.

    La causalidad es la misma en las tres: la etiqueta debe estar cerrada (`label_end_ts <= date`) y
    la predicción ser anterior (`prediction_ts < date`). Solo cambia cuántas cohortes se conservan.
    """
    closed = quarterly.loc[
        quarterly["label_end_ts"].le(date) & quarterly["prediction_ts"].lt(date)
    ]
    if closed.empty:
        return closed
    dates = sorted(closed["prediction_ts"].dropna().unique())
    if window == "horizon":
        # Las cohortes que caben en el horizonte objetivo: con 12 meses y cadencia trimestral, 4.
        keep = dates[-max(1, horizon_months // 3):]
    elif window == "era":
        keep = dates[-ERA_QUARTERS:]
    else:
        keep = dates
    return closed.loc[closed["prediction_ts"].isin(keep)]


def calibrated_alpha_path(
    scores: pd.DataFrame, targets: pd.DataFrame, *, horizon_months: int,
    half_life_quarters: float = 8.0, minimum_cohorts: int = 2,
) -> pd.DataFrame:
    """Alfa esperado anualizado por cascada de ventanas sobre la curva ventil -> retorno real.

    Para cada snapshot se ajusta una recta `alfa_anual ~ ventil de meta_rank` sobre cohortes ya
    cerradas, empezando por la ventana más reactiva y ampliando solo si la pendiente no es creciente:

    ``horizonte objetivo`` -> ``era (16 trimestres)`` -> ``todo el histórico`` -> salvaguarda fija.

    Estimación y evaluación usan granularidades distintas a propósito: la recta se **estima** sobre
    ventiles, porque una media necesita muestra suficiente detrás (~25 acciones por ventil con un
    universo de ~500), y se **evalúa** en el rank continuo de cada acción, para que el alfa esperado
    respete el orden completo del ranking y no se aplane dentro de cada ventil.

    Una pendiente creciente significa que, en esa ventana, mejor percentil se tradujo en más alfa;
    es la condición que la cartera necesita para que ordenar por `meta_rank` tenga sentido económico.
    Cuando ninguna ventana la cumple se impone la recta de ``FALLBACK_ANNUAL_ALPHA`` (ver constante).

    Mientras no haya `minimum_cohorts` cohortes cerradas el valor es ``NaN``, no ``0.0``: son cosas
    distintas y la cartera las trata distinto. ``0.0`` significa "se espera alfa nulo", lo que
    activaría ventas y bloquearía compras durante todo el arranque; ``NaN`` significa "todavía no hay
    evidencia", y en ese caso la cartera decide por ordenación.
    """
    labelled = _labelled_for_curve(scores, targets, horizon_months)
    if labelled.empty:
        return pd.DataFrame()
    quarterly = _closed_quarterly(labelled)
    rows: list[pd.DataFrame] = []
    for raw_date, current in labelled.groupby("snapshot_date", sort=True):
        date = pd.Timestamp(raw_date)
        # La recta se ESTIMA agrupando en ventiles (para que cada media tenga muestra suficiente),
        # pero se EVALÚA en el rank continuo de cada acción: así un p99 recibe estrictamente más
        # alfa esperado que un p88, en vez de compartir el valor de su ventil. Agrupar aquí también
        # crearía saltos artificiales en la frontera entre ventiles, justo donde la cartera decide.
        position = pd.to_numeric(current["meta_rank"], errors="coerce").to_numpy(dtype=float)
        position = position * (VENTILES - 1)
        chosen: tuple[float, float] | None = None
        chosen_window = "none"
        cohorts = 0
        for window in CURVE_WINDOWS:
            closed = _window_history(quarterly, date, window, horizon_months)
            window_cohorts = int(closed["prediction_ts"].nunique()) if not closed.empty else 0
            cohorts = max(cohorts, window_cohorts)
            if window_cohorts < minimum_cohorts:
                continue
            line = _fit_line(_ventile_points(closed, half_life_quarters))
            if line is not None and line[0] > 0:
                chosen, chosen_window = line, window
                break
        if chosen is None and cohorts >= minimum_cohorts:
            chosen, chosen_window = _fallback_line(), "fallback"
        if chosen is None:
            expected = np.full(len(current), np.nan, dtype=float)
            slope = np.nan
        else:
            slope, intercept = chosen
            expected = slope * position + intercept
        part = current[["ticker", "snapshot_date"]].copy()
        part["expected_excess_return"] = expected
        part["calibration_closed_quarters"] = cohorts
        part["alpha_curve_window"] = chosen_window
        part["alpha_curve_slope"] = slope
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

