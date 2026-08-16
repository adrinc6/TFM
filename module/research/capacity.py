"""Capacidad: ¿a partir de qué patrimonio deja la cartera de ser ejecutable?

Es la crítica que sigue al resultado económico: una cartera concentrada, de rotación alta y sobre
un universo de 500 valores puede batir al índice con un millón y ser irrealizable con mil. La
pregunta no se responde con el alfa, se responde con el volumen del mercado en el que hay que
operar.

**Qué se mide y qué no.** Se mide la *participación*: qué fracción del volumen negociado habitual de
una acción representaría cada orden de la cartera a un patrimonio dado. No se modela impacto de
mercado —cuánto movería el precio esa orden—, que es un trabajo aparte y exigiría supuestos que este
panel no puede sostener. La participación es observable; el impacto sería una hipótesis.

**Cómo se lee.** Por debajo del umbral declarado la cartera es ejecutable tal cual se simuló; por
encima, la simulación describe operaciones que el mercado no habría absorbido a esos precios, y el
resultado del backtest deja de ser alcanzable. Se publican dos umbrales porque no hay uno canónico:
el 5 % es la convención prudente de la industria y el 10 % el límite que rara vez se cruza sin mover
el precio.

**Volumen ausente no es liquidez infinita.** Si una orden cae sobre un ticker sin volumen medido, se
cuenta aparte como cobertura incompleta en vez de tratarla como ejecutable. Lo contrario dejaría que
los huecos del panel subieran la capacidad estimada, que es exactamente el sesgo que este
diagnóstico existe para no cometer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from module.common.utils import write_json
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR

log = logging.getLogger(__name__)

# Patrimonios sobre los que se evalúa la cartera, en dólares. Cubren desde una cartera personal
# hasta un fondo institucional, que es el rango en el que la respuesta cambia.
ASSETS_UNDER_MANAGEMENT = (1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9)

# Fracción del volumen diario habitual que una orden puede representar sin que la ejecución al
# precio simulado deje de ser creíble. Dos umbrales declarados, ninguno elegido por conveniencia.
PARTICIPATION_THRESHOLDS = (0.05, 0.10)

# Percentil de la distribución de participaciones que gobierna el veredicto. El máximo lo decidiría
# una sola orden atípica de un solo día; el percentil 95 describe el régimen habitual y sigue siendo
# exigente. Ambos se publican.
GOVERNING_PERCENTILE = 95

WINDOWS = ("selection", "confirmation")


def _orders_with_volume(
    orders: pd.DataFrame, equity: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    """Cada orden con la fracción de cartera que mueve y el volumen disponible ese día.

    ``notional`` viene en unidades del valor de la cartera simulada (que arranca en 100), así que
    dividirlo por el valor del periodo lo convierte en la **fracción de cartera** que se opera. Esa
    fracción sí es escalable a cualquier patrimonio; el nocional en bruto no.
    """
    if orders.empty:
        return pd.DataFrame()
    value = equity[["snapshot_date", "period_start_portfolio_value"]].copy()
    frame = orders.merge(value, on="snapshot_date", how="left")
    frame["portfolio_fraction"] = frame["notional"] / frame["period_start_portfolio_value"]
    columns = ["ticker", "snapshot_date", "median_dollar_volume_21d"]
    if "median_dollar_volume_21d" in prices.columns:
        frame = frame.merge(prices[columns], on=["ticker", "snapshot_date"], how="left")
    else:
        frame["median_dollar_volume_21d"] = np.nan
    frame["year"] = pd.to_datetime(frame["snapshot_date"]).dt.year
    return frame


def _window_slice(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "selection":
        return frame.loc[frame["year"].le(SELECTION_UNTIL_YEAR)]
    return frame.loc[frame["year"].isin(KNOWN_STRESS_YEARS)]


def participation(frame: pd.DataFrame, aum: float) -> pd.Series:
    """Participación de cada orden sobre el volumen mediano diario, a un patrimonio dado.

    Escala **linealmente** con el patrimonio por construcción: la misma cartera con el doble de
    dinero manda órdenes del doble de tamaño sobre el mismo mercado.
    """
    volume = pd.to_numeric(frame["median_dollar_volume_21d"], errors="coerce")
    traded = frame["portfolio_fraction"] * aum
    return (traded / volume.where(volume > 0)).replace([np.inf, -np.inf], np.nan)


def _window_block(frame: pd.DataFrame, aums: Sequence[float]) -> dict[str, Any]:
    """Participación por patrimonio y patrimonio máximo admisible por umbral, en una ventana."""
    total = len(frame)
    if not total:
        return {"available": False, "reason": "La ventana no contiene órdenes."}
    measured = frame.loc[pd.to_numeric(frame["median_dollar_volume_21d"], errors="coerce").gt(0)]
    coverage = len(measured) / total
    if measured.empty:
        return {
            "available": False, "orders": total, "volume_coverage": 0.0,
            "reason": "Ninguna orden de la ventana tiene volumen medido.",
        }
    ladder = []
    for aum in aums:
        values = participation(measured, float(aum)).dropna()
        ladder.append({
            "aum_usd": float(aum),
            "median_participation": float(values.median()),
            "p95_participation": float(np.percentile(values, GOVERNING_PERCENTILE)),
            "max_participation": float(values.max()),
        })
    return {
        "available": True,
        "orders": total,
        "orders_with_volume": len(measured),
        "volume_coverage": float(coverage),
        "ladder": ladder,
        "maximum_aum_usd": {
            f"{threshold:.0%}": _maximum_aum(measured, threshold) for threshold in PARTICIPATION_THRESHOLDS
        },
        "binding_names": _binding_names(measured),
    }


def _maximum_aum(frame: pd.DataFrame, threshold: float) -> float:
    """Patrimonio al que la participación gobernante alcanza justo el umbral.

    Como la participación es lineal en el patrimonio, no hace falta buscar: se resuelve en forma
    cerrada dividiendo el umbral por la participación gobernante medida a un patrimonio de
    referencia. Es exacto, no una interpolación sobre la escalera.
    """
    reference = 1e9
    values = participation(frame, reference).dropna()
    if values.empty:
        return float("nan")
    governing = float(np.percentile(values, GOVERNING_PERCENTILE))
    if governing <= 0:
        return float("inf")
    return reference * threshold / governing


def _binding_names(frame: pd.DataFrame, top: int = 10) -> list[dict[str, Any]]:
    """Las acciones que atan el límite: mucha orden sobre poco volumen."""
    reference = 1e9
    working = frame.assign(participation=participation(frame, reference))
    working = working.loc[working["participation"].notna()]
    if working.empty:
        return []
    grouped = (
        working.groupby("ticker")
        .agg(
            orders=("participation", "size"),
            max_participation=("participation", "max"),
            median_dollar_volume=("median_dollar_volume_21d", "median"),
        )
        .sort_values("max_participation", ascending=False)
        .head(top)
        .reset_index()
    )
    grouped["reference_aum_usd"] = reference
    return grouped.to_dict("records")


def build_capacity(
    evidence_dir: Path, prepared: Path, *, aums: Sequence[float] = ASSETS_UNDER_MANAGEMENT,
) -> dict[str, Any]:
    """Bloque de capacidad sobre la evidencia ya congelada del ganador."""
    orders = pd.read_parquet(evidence_dir / "orders.parquet")
    equity = pd.read_parquet(evidence_dir / "equity.parquet")
    prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
    frame = _orders_with_volume(orders, equity, prices)
    has_volume = "median_dollar_volume_21d" in prices.columns
    return {
        "volume_source": "asset_price_point_in_time.parquet::median_dollar_volume_21d",
        "volume_available": bool(has_volume),
        "participation_thresholds": [float(value) for value in PARTICIPATION_THRESHOLDS],
        "governing_percentile": GOVERNING_PERCENTILE,
        "aum_ladder_usd": [float(value) for value in aums],
        "windows": {
            window: _window_block(_window_slice(frame, window), aums) if not frame.empty
            else {"available": False, "reason": "El ganador no registró órdenes."}
            for window in WINDOWS
        },
        "caveats": [
            (
                "El nocional diario es aproximado: el precio está ajustado por splits y dividendos "
                "y el volumen solo por splits."
            ),
            (
                "Se mide participación sobre el volumen habitual, no impacto de mercado. Por debajo "
                "del umbral la ejecución es creíble; no se afirma que sea gratuita."
            ),
            (
                "Una orden sin volumen medido se cuenta como cobertura incompleta, nunca como "
                "ejecutable: un hueco del panel no puede subir la capacidad estimada."
            ),
        ],
    }


def write_capacity(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(dict(payload), path)
