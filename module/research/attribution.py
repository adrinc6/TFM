"""Atribución del alfa: ¿el sistema aprende algo propio o redescubre un factor conocido?

Es la evidencia que decide la lectura del trabajo. Sin ella, la interpretación natural de un
Rank-IC por agente de risk 0,082 frente a un meta de 0,067 es «no aprendió nada: redescubrió el
efecto baja volatilidad». Con ella, la afirmación «el sistema aprende una ordenación propia» se
sostiene o se cae sobre una regresión, no sobre una intuición.

Cuatro piezas, todas construidas con el mismo panel point-in-time y sin fuentes externas:

1. **Carteras réplica de factores** construidas del propio universo: no hay acceso a las series de
   Fama-French, así que los factores se replican como diferencial tercil superior menos tercil
   inferior de cada característica, rebalanceado en cada snapshot. Es reproducible desde los
   artefactos del repositorio y usa exactamente la misma disciplina PIT que el modelo.
2. **Regresión con errores Newey-West**, porque los retornos solapados y autocorrelacionados
   inflarían la significación con errores estándar clásicos.
3. **Deflated Sharpe Ratio** (Bailey y López de Prado): con 50 evaluaciones y 17 decisiones, un
   p-valor sin corregir por multiplicidad no es defendible ante un tribunal.
4. **Coeficiente de transferencia y curva de amplitud**: cuánta de la señal medida por el Rank-IC
   llega efectivamente al alfa de la cartera.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from module.data.universe import annual_membership_dates, members_at
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR

log = logging.getLogger(__name__)

# Réplicas de factores construidas desde el catálogo de features. La clave es el nombre del factor
# de estilo; el valor, los factores del panel que lo representan (ya en rango percentil, dirección
# ya normalizada a «mayor = mejor» por `features.py`).
FACTOR_PROXIES: dict[str, tuple[str, ...]] = {
    "value": ("factor_pe", "factor_pb", "factor_ps", "factor_ev_ebitda"),
    "momentum": ("factor_relative_return_12m", "factor_momentum_12_1"),
    "low_volatility": ("factor_realized_vol_63d", "factor_realized_vol_126d", "factor_beta_252d"),
    "quality": ("factor_roe", "factor_roic", "factor_operating_margin"),
    "growth": ("factor_eps_growth_yoy", "factor_sales_per_share_growth_yoy"),
}
# Factores del modelo de cinco de Fama-French que este panel no puede replicar con honestidad.
# Declararlos ausentes es preferible a sustituirlos por un sucedáneo que nadie pueda auditar.
UNAVAILABLE_FACTORS = {
    "size": "El panel no incluye capitalización bursátil ni número de acciones en circulación.",
    "investment": "El panel no incluye crecimiento de activos totales point-in-time.",
}
TERCILE = 3


def build_attribution(
    evidence: Path,
    prepared: Path,
    values: Mapping[str, Any],
    *,
    candidate_ic_series: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Construye el bloque completo de atribución sobre el ganador ya congelado."""
    equity = pd.read_parquet(evidence / "equity.parquet")
    diagnostics = pd.read_parquet(evidence / "rank_ic_diagnostics.parquet")
    features = pd.read_parquet(prepared / "features_point_in_time.parquet")
    prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
    targets = pd.read_parquet(prepared / "targets_forward.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    tail_path = evidence / "rank_tail_diagnostics.parquet"
    tail = pd.read_parquet(tail_path) if tail_path.exists() else pd.DataFrame()

    forward = _one_period_returns(prices)
    factors = _factor_returns(features, forward)
    meta_ic = diagnostics.loc[diagnostics["agent"].eq("meta_final")].copy()
    meta_ic["year"] = pd.to_datetime(meta_ic["prediction_date"]).dt.year
    return {
        "factor_regression": _factor_regression(equity, factors),
        "factor_proxies": {
            "method": "diferencial tercil superior menos tercil inferior, rebalanceado por snapshot",
            "available": sorted(factors.columns.tolist()),
            "unavailable": UNAVAILABLE_FACTORS,
        },
        "neutralized_rank_ic": _neutralized_rank_ic(scores, targets, features),
        "ic_significance": _ic_significance(meta_ic),
        "deflated_sharpe": _deflated_sharpe(equity, candidate_ic_series or []),
        "baselines": _baseline_rank_ic(prepared, targets),
        "universe_coverage": _coverage_by_year(features),
        "transfer": _transfer_curve(equity, tail),
    }


# --- Réplicas de factores ---------------------------------------------------------------------

def _one_period_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retorno de cada acción entre snapshots consecutivos, alineado con la cadencia de la cartera.

    Se usa el paso entre snapshots y no `forward_excess_return` porque el horizonte de la etiqueta
    (hasta 12 meses) no coincide con la frecuencia a la que la cartera realiza resultados; mezclar
    ambas frecuencias en una regresión produce betas sin interpretación.
    """
    frame = prices[["ticker", "snapshot_date", "price"]].copy()
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_date"])
    frame = frame.sort_values(["ticker", "snapshot_ts"])
    frame["next_price"] = frame.groupby("ticker")["price"].shift(-1)
    frame["forward_1p"] = frame["next_price"] / frame["price"] - 1
    return frame.loc[frame["price"].gt(0) & frame["forward_1p"].notna(),
                     ["ticker", "snapshot_date", "forward_1p"]]


def _factor_returns(features: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Retorno por snapshot de cada cartera réplica: tercil superior menos tercil inferior."""
    merged = features.merge(forward, on=["ticker", "snapshot_date"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    series: dict[str, pd.Series] = {}
    for name, columns in FACTOR_PROXIES.items():
        present = [column for column in columns if column in merged.columns]
        if not present:
            continue
        characteristic = merged[present].mean(axis=1, skipna=True)
        usable = merged.loc[characteristic.notna()].copy()
        usable["characteristic"] = characteristic.dropna()
        rows: dict[str, float] = {}
        for date, cohort in usable.groupby("snapshot_date", sort=True):
            clean = cohort[["characteristic", "forward_1p"]].dropna()
            if len(clean) < 3 * TERCILE:
                continue
            ordered = clean.sort_values("characteristic")
            size = len(ordered) // TERCILE
            low = ordered.head(size)["forward_1p"].mean()
            high = ordered.tail(size)["forward_1p"].mean()
            rows[str(date)] = float(high - low)
        if rows:
            series[name] = pd.Series(rows)
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()


# --- Regresión con Newey-West -----------------------------------------------------------------

def _newey_west_ols(y: np.ndarray, x: np.ndarray, lags: int) -> dict[str, Any]:
    """MCO con matriz de covarianzas Newey-West (HAC).

    Los retornos de una cartera con rotación y horizontes solapados están autocorrelacionados; con
    errores estándar clásicos, un alfa sin significación real aparece como significativo.
    """
    n, k = x.shape
    if n <= k + 1:
        return {"applicable": False, "reason": "observaciones insuficientes para la regresión"}
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    s = (x * residuals[:, None])
    omega = s.T @ s
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = s[lag:].T @ s[:-lag]
        omega += weight * (gamma + gamma.T)
    covariance = xtx_inv @ omega @ xtx_inv
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(standard_errors > 0, beta / standard_errors, np.nan)
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals ** 2).sum()) / total if total > 0 else None
    return {
        "applicable": True, "beta": beta.tolist(),
        "standard_errors": standard_errors.tolist(),
        "t_statistics": [None if not np.isfinite(t) else float(t) for t in t_stats],
        "n_observations": int(n), "lags": int(lags), "r_squared": r_squared,
    }


def _factor_regression(equity: pd.DataFrame, factors: pd.DataFrame) -> dict[str, Any]:
    """Regresa el exceso de la cartera sobre las réplicas de factores. El alfa es el intercepto."""
    if factors.empty or equity.empty:
        return {"applicable": False, "reason": "sin réplicas de factores disponibles"}
    frame = equity[["snapshot_date", "excess_return"]].copy()
    frame["snapshot_date"] = frame["snapshot_date"].astype(str)
    frame = frame.set_index("snapshot_date").join(factors, how="inner").dropna()
    years = pd.to_datetime(frame.index).year
    result: dict[str, Any] = {"factors": list(factors.columns)}
    for label, mask in (
        ("selection", years <= SELECTION_UNTIL_YEAR),
        ("confirmation", np.isin(years, KNOWN_STRESS_YEARS)),
        ("full", np.ones(len(frame), dtype=bool)),
    ):
        window = frame.loc[mask]
        if len(window) < len(factors.columns) + 3:
            result[label] = {"applicable": False, "reason": "observaciones insuficientes"}
            continue
        y = window["excess_return"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(window)), window[list(factors.columns)].to_numpy(dtype=float)])
        fitted = _newey_west_ols(y, design, lags=12)
        if fitted.get("applicable"):
            fitted["alpha_per_period"] = fitted["beta"][0]
            fitted["alpha_t_stat"] = fitted["t_statistics"][0]
            fitted["loadings"] = dict(zip(factors.columns, fitted["beta"][1:], strict=True))
            fitted["loading_t_stats"] = dict(zip(factors.columns, fitted["t_statistics"][1:], strict=True))
        result[label] = fitted
    result["interpretation"] = (
        "Un alfa positivo con t de Newey-West por encima de 2 indica que el exceso de la cartera no "
        "se explica por la exposición a las réplicas de valor, momentum, baja volatilidad, calidad y "
        "crecimiento. Un alfa no significativo con una carga alta en baja volatilidad indicaría que "
        "el sistema reproduce ese factor conocido en lugar de aprender una ordenación propia."
    )
    return result


# --- Rank-IC neutralizado ----------------------------------------------------------------------

def _neutralized_rank_ic(scores: pd.DataFrame, targets: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    """Rank-IC contra el retorno residualizado por las características de estilo.

    Si la capacidad predictiva sobrevive a retirar del objetivo lo explicable por valor, momentum,
    baja volatilidad y calidad, la ordenación aporta información propia.
    """
    controls = [column for group in FACTOR_PROXIES.values() for column in group if column in features.columns]
    if not controls:
        return {"applicable": False, "reason": "sin características de control en el panel"}
    from module.modeling.targets import normalize_target_columns

    frame = scores[["ticker", "snapshot_date", "meta_rank"]].merge(
        normalize_target_columns(targets)[["ticker", "snapshot_date", "forward_excess_return", "target_available"]],
        on=["ticker", "snapshot_date"], how="inner",
    ).merge(features[["ticker", "snapshot_date", *controls]], on=["ticker", "snapshot_date"], how="inner")
    frame = frame.loc[frame["target_available"].fillna(False)]
    frame["year"] = pd.to_datetime(frame["snapshot_date"]).dt.year
    raw_ics, residual_ics = [], []
    for _, cohort in frame.loc[frame["year"].le(SELECTION_UNTIL_YEAR)].groupby("snapshot_date"):
        clean = cohort[["meta_rank", "forward_excess_return", *controls]].dropna()
        if len(clean) < max(20, len(controls) + 5):
            continue
        y = clean["forward_excess_return"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(clean)), clean[controls].to_numpy(dtype=float)])
        residual = y - design @ (np.linalg.pinv(design.T @ design) @ design.T @ y)
        rank = clean["meta_rank"]
        raw_ics.append(float(rank.corr(clean["forward_excess_return"], method="spearman")))
        residual_ics.append(float(rank.corr(pd.Series(residual, index=clean.index), method="spearman")))
    if not residual_ics:
        return {"applicable": False, "reason": "sin cohortes con sección transversal suficiente"}
    return {
        "applicable": True,
        "controls": controls,
        "raw_mean_rank_ic": float(np.mean(raw_ics)),
        "neutralized_mean_rank_ic": float(np.mean(residual_ics)),
        "retained_fraction": float(np.mean(residual_ics) / np.mean(raw_ics)) if np.mean(raw_ics) else None,
        "n_cohorts": len(residual_ics),
    }


# --- Significación e IC-IR ---------------------------------------------------------------------

def _ic_significance(meta_ic: pd.DataFrame) -> dict[str, Any]:
    """IC-IR y t de Newey-West del Rank-IC medio, por ventana."""
    out: dict[str, Any] = {}
    for label, frame in (
        ("selection", meta_ic.loc[meta_ic["year"].le(SELECTION_UNTIL_YEAR)]),
        ("confirmation", meta_ic.loc[meta_ic["year"].isin(KNOWN_STRESS_YEARS)]),
    ):
        values = pd.to_numeric(frame.get("rank_ic"), errors="coerce").dropna().to_numpy()
        if len(values) < 2:
            out[label] = {"applicable": False, "n_cohorts": len(values)}
            continue
        fitted = _newey_west_ols(values, np.ones((len(values), 1)), lags=12)
        std = float(values.std(ddof=1))
        out[label] = {
            "applicable": True,
            "mean_rank_ic": float(values.mean()),
            "ic_ir": float(values.mean() / std) if std > 0 else None,
            "newey_west_t": fitted.get("t_statistics", [None])[0] if fitted.get("applicable") else None,
            "n_cohorts": len(values),
            # Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten 11 meses de
            # etiqueta: el número de observaciones independientes es mucho menor que el de cohortes.
            "effective_independent_observations": max(1, len(values) // 12),
        }
    return out


def _deflated_sharpe(equity: pd.DataFrame, candidate_series: list[list[float]]) -> dict[str, Any]:
    """Deflated Sharpe Ratio de Bailey y López de Prado.

    Corrige el Sharpe observado por el número de configuraciones probadas: buscar entre muchas
    alternativas produce un máximo alto aunque ninguna tenga capacidad real. Sin esta corrección, un
    p-valor obtenido tras 50 evaluaciones y 17 decisiones sobrestima la evidencia.

    **Aproximación declarada**: el Sharpe observado se calcula sobre los excesos de la cartera,
    pero la dispersión entre configuraciones se estima con las series de Rank-IC de los candidatos,
    porque no existe un backtest por candidato (solo el ganador congelado se traduce a cartera). La
    corrección exacta exigiría ejecutar la cartera para las 50 configuraciones. El resultado se lee
    como orden de magnitud del *haircut* por multiplicidad, no como una probabilidad exacta.
    """
    excess = pd.to_numeric(equity.get("excess_return"), errors="coerce").dropna().to_numpy()
    trials = max(len(candidate_series), 1)
    if len(excess) < 8:
        return {"applicable": False, "reason": "serie demasiado corta", "n_trials": trials}
    n = len(excess)
    std = float(excess.std(ddof=1))
    if std <= 0:
        return {"applicable": False, "reason": "varianza nula", "n_trials": trials}
    observed = float(excess.mean() / std)
    skew = float(pd.Series(excess).skew())
    kurtosis = float(pd.Series(excess).kurtosis()) + 3.0
    variances = [float(np.std(np.asarray(series, dtype=float), ddof=1)) for series in candidate_series
                 if len(series) > 1]
    means = [float(np.mean(np.asarray(series, dtype=float)) / value)
             for series, value in zip(candidate_series, variances, strict=False) if value > 0]
    spread = float(np.std(np.asarray(means), ddof=1)) if len(means) > 1 else 0.0
    euler = 0.5772156649015329
    if trials > 1 and spread > 0:
        threshold = spread * (
            (1 - euler) * _inverse_normal_cdf(1 - 1 / trials)
            + euler * _inverse_normal_cdf(1 - 1 / (trials * math.e))
        )
    else:
        threshold = 0.0
    denominator = math.sqrt(max(1e-12, 1 - skew * observed + (kurtosis - 1) / 4 * observed ** 2))
    statistic = (observed - threshold) * math.sqrt(n - 1) / denominator
    return {
        "applicable": True,
        "observed_sharpe_per_period": observed,
        "expected_max_sharpe_under_null": threshold,
        "deflated_sharpe_probability": float(_normal_cdf(statistic)),
        "n_trials": trials,
        "n_observations": n,
        "skew": skew,
        "kurtosis": kurtosis,
        "interpretation": (
            "La probabilidad es la confianza de que el Sharpe observado supere al máximo esperado "
            "por azar tras probar n_trials configuraciones. Por debajo de 0,95 el resultado no "
            "resiste la corrección por multiplicidad."
        ),
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _inverse_normal_cdf(probability: float) -> float:
    """Cuantil normal por bisección: evita una dependencia solo por esta función."""
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    low, high = -10.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2
        if _normal_cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


# --- Baselines deterministas --------------------------------------------------------------------

def _baseline_rank_ic(prepared: Path, targets: pd.DataFrame) -> dict[str, Any]:
    """Rank-IC de los baselines deterministas: la primera pregunta de cualquier tribunal.

    GARP y momentum puro son reglas sin aprendizaje. Si el sistema no los supera, el aparato de
    aprendizaje no está justificado. `module/data/baselines.py` los calculaba en cada build y su
    salida no la leía nadie.
    """
    path = prepared / "baseline_scores.parquet"
    if not path.exists():
        return {"applicable": False, "reason": "baseline_scores.parquet no materializado"}
    from module.modeling.targets import normalize_target_columns

    baselines = pd.read_parquet(path)
    merged = baselines.merge(
        normalize_target_columns(targets)[["ticker", "snapshot_date", "forward_excess_return", "target_available"]],
        on=["ticker", "snapshot_date"], how="inner",
    )
    merged = merged.loc[merged["target_available"].fillna(False)]
    merged["year"] = pd.to_datetime(merged["snapshot_date"]).dt.year
    merged = merged.loc[merged["year"].le(SELECTION_UNTIL_YEAR)]
    rows = []
    for column in sorted(name for name in baselines.columns if name.endswith("_rank")):
        cohort_ics = [
            float(cohort[column].corr(cohort["forward_excess_return"], method="spearman"))
            for _, cohort in merged.groupby("snapshot_date")
            if cohort[[column, "forward_excess_return"]].dropna().shape[0] >= 10
        ]
        finite = [value for value in cohort_ics if np.isfinite(value)]
        if finite:
            rows.append({
                "baseline": column.removesuffix("_rank"),
                "mean_rank_ic": float(np.mean(finite)),
                "positive_fraction": float(np.mean(np.asarray(finite) > 0)),
                "n_cohorts": len(finite),
            })
    return {"applicable": bool(rows), "baselines": rows}


# --- Cobertura y transferencia -------------------------------------------------------------------

def _coverage_by_year(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Cobertura efectiva del universo por año: convierte una limitación declarada en una medida.

    `usable_fraction` mide calidad **dentro** del panel y puede valer 100 % mientras falta media
    lista del índice, así que por sí sola no dice nada sobre cobertura. El denominador que sí lo
    dice son los miembros reales del S&P 500 de ese año (`members_at`): el índice ha tenido ~500
    componentes durante todo el periodo estudiado, de modo que la diferencia con `distinct_tickers`
    es cobertura perdida —empresas deslistadas cuyos fundamentales no se recuperan—, no un índice
    más pequeño. Confundir ambas cosas presenta un defecto medible como una característica del
    diseño.
    """
    if "snapshot_date" not in features:
        return []
    frame = features[["ticker", "snapshot_date"]].copy()
    frame["year"] = pd.to_datetime(frame["snapshot_date"]).dt.year
    fresh = features.get("is_price_fresh")
    frame["usable"] = fresh.fillna(False).to_numpy() if fresh is not None else True
    members_by_year = _index_members_by_year()
    rows = []
    for year, group in frame.groupby("year"):
        distinct = int(group["ticker"].nunique())
        members = members_by_year.get(int(year), 0)
        rows.append({
            "year": int(year),
            "distinct_tickers": distinct,
            "sp500_members": members,
            "panel_coverage_fraction": float(distinct / members) if members else None,
            "usable_rows": int(group["usable"].sum()),
            "total_rows": len(group),
            "usable_fraction": float(group["usable"].mean()),
        })
    return rows


def _index_members_by_year() -> dict[int, int]:
    """Miembros del S&P 500 en el último snapshot de composición de cada año natural.

    Sin el CSV de composición no hay denominador posible. Se degrada a vacío —la cobertura queda
    `None`, no 0— en vez de propagar el error: un panel sintético de test no tiene por qué traer la
    composición histórica, y perder el denominador no invalida el resto de la atribución.
    """
    try:
        return {
            int(as_of.year): len(members_at(as_of))
            for as_of in annual_membership_dates()
        }
    except FileNotFoundError:
        log.warning(
            "Sin composición histórica del S&P 500: la cobertura del panel se publica sin "
            "denominador y no puede compararse contra el tamaño real del índice."
        )
        return {}


def _transfer_curve(equity: pd.DataFrame, tail: pd.DataFrame) -> dict[str, Any]:
    """Relación entre rotación y alfa neto, y coeficiente de transferencia de la señal."""
    if equity.empty:
        return {"applicable": False}
    frame = equity.copy()
    frame["year"] = pd.to_datetime(frame["snapshot_date"]).dt.year
    frame = frame.loc[frame["year"].le(SELECTION_UNTIL_YEAR)]
    by_year = frame.groupby("year").agg(
        turnover=("turnover_pct", "sum"),
        excess=("excess_return", "sum"),
        cost=("cost_drag", "sum"),
    ).reset_index()
    result: dict[str, Any] = {
        "applicable": True,
        "by_year": by_year.to_dict("records"),
        "cost_per_unit_turnover": (
            float(by_year["cost"].sum() / by_year["turnover"].sum())
            if by_year["turnover"].sum() > 0 else None
        ),
    }
    if not tail.empty and "top_minus_bottom" in tail:
        spread = pd.to_numeric(tail["top_minus_bottom"], errors="coerce").dropna()
        if not spread.empty:
            result["signal_top_minus_bottom_mean"] = float(spread.mean())
            realized = float(frame["excess_return"].sum())
            result["realized_cumulative_excess"] = realized
    return result


def write_attribution(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
