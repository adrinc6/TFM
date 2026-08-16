"""Sensibilidad del resultado económico al supuesto de coste: hasta dónde aguanta.

Los costes del proyecto son constantes —comisión y slippage fijos— sobre una cartera de rotación
alta, y ésa es la cifra más atacable del capítulo económico. La limitación figura con severidad
Media y **sin ninguna cifra que la acote**. Lo que falta no es un supuesto de coste mejor, sino
decir hasta dónde aguanta el que hay. Este módulo lo mide.

Tres escenarios sobre la cartera que el TFM adopta —la ganadora del Portfolio Study, no la del
catálogo por defecto—: **bruto** (coste 0, cota superior que nadie realiza), **estándar** (el que ya
se reporta) y **equilibrio** (`c*`, el coste que anula el exceso geométrico contra el S&P 500). El
equilibrio se define contra el índice y no contra rentabilidad absoluta, porque la alternativa real
de un inversor es comprar el índice, no quedarse en efectivo.

**Los dos hechos que determinan el diseño.**

1. *El coste es exactamente ``turnover × tasa``.* En ``_price_orders`` el drag total es
   ``Σ(notional × tasa) / value``, y como ``notional = |Δw| × value`` y ``turnover = Σ|Δw|``, sale
   ``drag = turnover × tasa`` sin aproximación. ``equity.parquet`` ya persiste ``turnover_pct`` y
   ``cost_drag`` por snapshot, así que sobre la ruta de operaciones ya ejecutada la curva entera se
   obtiene en forma cerrada, con cero cómputo.
2. *Pero el coste entra dos veces.* Además de la contabilidad, alimenta los umbrales de decisión:
   ``round_trip`` en ``module/evaluation/portfolio.py`` fija ``entry_threshold`` y
   ``rotation_threshold``. Poner el coste a cero **no es la misma cartera sin comisiones**: los
   umbrales se desploman y operaría mucho más. Y al revés, con costes altos la cartera opera menos y
   se protege sola.

Por eso un solo número sería engañoso y se calculan **dos familias**: la de *ruta congelada* (mismas
decisiones, distinto coste), cuyo ``c*`` es conservador porque un gestor que pagase más operaría
menos; y la *resimulada*, en la que la cartera vuelve a decidir con cada coste, de modo que su
``c**`` es mayor o igual. La diferencia entre ambas es en sí misma un resultado: mide cuánto protege
la doctrina de umbrales económicos, que existe precisamente para que cada operación pague su coste.

Los costes **nunca seleccionan**. Elegir el coste sería elegir el mundo que más conviene a la
estrategia, y por eso `commission_bps` y `slippage_bps` quedan fuera del Portfolio Study. Este
módulo es diagnóstico: escribe su propio artefacto y no toca ninguna decisión.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from module.common.utils import write_json
from module.evaluation.backtest import annual_metrics, periods_per_year, window_metrics
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR

log = logging.getLogger(__name__)

BPS = 10_000.0

# Peldaños de coste **por operación** (comisión + slippage). Van de cero hasta bastante más allá del
# equilibrio esperado, porque una escalera que se quede corta no mide nada: solo dice que el
# equilibrio está «más arriba».
#
# La escalera es una constante de diagnóstico y no una variable del catálogo cerrado, con el mismo
# precedente que el ensemble de semillas y las iteraciones del bootstrap: no se optimiza, y el
# catálogo no podría expresarla —sus valores de coste van de 5 a 30 pb, ni el cero ni nada cercano
# al equilibrio esperado—. `settings_from_values` no valida contra el catálogo (la validación ocurre
# antes, al definir el study), así que basta pasar el coste en los valores del backtest.
COST_LADDER_BPS: tuple[float, ...] = (
    0.0, 5.0, 15.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0,
)

# Las dos ventanas se calculan y se publican **siempre por separado**. La de selección es la única
# que alimentó decisiones; 2025-2026 es confirmación. Un equilibrio calculado sobre la serie mezclada
# no respondería a ninguna pregunta contestable.
WINDOWS = ("selection", "confirmation")

# Métrica sobre la que se define el equilibrio: el exceso geométrico contra el benchmark.
BREAK_EVEN_METRIC = "geometric_excess_return"


def _repriced(equity: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Recompone la curva de patrimonio con otra tasa de coste, sobre las mismas operaciones.

    Reproduce paso a paso la contabilidad del motor —``valor × (1 + bruto) × (1 − drag)``— en vez de
    aproximarla, que es lo que permite que evaluada en el coste adoptado devuelva exactamente las
    cifras que ya reporta el ganador. Esa autoconsistencia es un test de contrato.
    """
    frame = equity.copy()
    gross = frame["gross_return"].to_numpy(dtype=float)
    drag = frame["turnover_pct"].to_numpy(dtype=float) * rate
    start = np.empty(len(frame), dtype=float)
    after = np.empty(len(frame), dtype=float)
    value = float(frame["period_start_portfolio_value"].iloc[0])
    for index in range(len(frame)):
        start[index] = value
        before = value * (1.0 + gross[index])
        value = before * (1.0 - drag[index])
        after[index] = value
    frame["period_start_portfolio_value"] = start
    frame["portfolio_value"] = after
    frame["cost_drag"] = drag
    frame["portfolio_return"] = after / start - 1.0
    frame["excess_return"] = frame["portfolio_return"] - frame["benchmark_return"]
    frame["cumulative_costs"] = np.cumsum(start * (1.0 + gross) * drag)
    return frame


def windowed_metrics(equity: pd.DataFrame, settings: Any) -> dict[str, dict[str, Any]]:
    """Métricas de las dos ventanas, con las mismas definiciones que usa el propio backtest."""
    annual = annual_metrics(equity, settings)
    periods = periods_per_year(settings)
    years = pd.to_datetime(equity["snapshot_date"]).dt.year
    return {
        "selection": window_metrics(
            equity.loc[years.le(SELECTION_UNTIL_YEAR)],
            annual.loc[annual["year"].le(SELECTION_UNTIL_YEAR)],
            periods,
        ),
        "confirmation": window_metrics(
            equity.loc[years.isin(KNOWN_STRESS_YEARS)],
            annual.loc[annual["year"].isin(KNOWN_STRESS_YEARS)],
            periods,
        ),
    }


def frozen_path_curve(
    equity: pd.DataFrame, settings: Any, ladder: Sequence[float] = COST_LADDER_BPS,
) -> list[dict[str, Any]]:
    """Familia de ruta congelada: mismas decisiones, distinto coste. Forma cerrada, sin simular."""
    rows: list[dict[str, Any]] = []
    for cost_bps in ladder:
        metrics = windowed_metrics(_repriced(equity, float(cost_bps) / BPS), settings)
        rows.append({"cost_bps": float(cost_bps), "family": "frozen_path", **_flatten(metrics)})
    return rows


def resimulated_curve(
    configuration: Mapping[str, Any], evidence_dir: Path, adopted_bps: float,
    ladder: Sequence[float] = COST_LADDER_BPS,
) -> list[dict[str, Any]]:
    """Familia resimulada: la cartera vuelve a decidir con cada coste.

    Cuesta unos segundos por peldaño porque reutiliza los scores ya congelados y solo rehace el
    backtest, sin reentrenar nada. Es el precio de medir el efecto del coste sobre los umbrales, que
    la forma cerrada no puede ver.
    """
    from module.studies.runner import run_profile_evaluation

    rows: list[dict[str, Any]] = []
    for cost_bps in ladder:
        commission, slippage = _split(float(cost_bps), configuration, adopted_bps)
        payload = run_profile_evaluation(
            {**dict(configuration), "commission_bps": commission, "slippage_bps": slippage},
            "balanced", evidence_dir,
        )
        summary = payload.get("summary", {})
        metrics = {
            "selection": {
                key: value for key, value in summary.items()
                if key not in {"confirmation", "full_curve"}
            },
            "confirmation": dict(summary.get("confirmation", {})),
        }
        rows.append({
            "cost_bps": float(cost_bps), "family": "resimulated",
            "commission_bps": commission, "slippage_bps": slippage, **_flatten(metrics),
        })
    return rows


def _split(cost_bps: float, configuration: Mapping[str, Any], adopted_bps: float) -> tuple[float, float]:
    """Reparte un peldaño entre comisión y slippage conservando la proporción adoptada.

    Al motor solo le importa la suma —tanto el drag como los umbrales usan ``comisión + slippage``—,
    así que el reparto no cambia ningún resultado. Se conserva la proporción para que las columnas
    de comisión y slippage de las órdenes sigan siendo legibles y no aparezca un slippage de cero
    que nadie ha decidido.
    """
    commission = float(configuration.get("commission_bps") or 0.0)
    if adopted_bps <= 0:
        return cost_bps, 0.0
    share = commission / adopted_bps
    return cost_bps * share, cost_bps * (1.0 - share)


def _flatten(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        f"{window}_{key}": value
        for window in WINDOWS
        for key, value in (metrics.get(window) or {}).items()
    }


def break_even_bps(rows: Sequence[Mapping[str, Any]], window: str) -> dict[str, Any]:
    """Coste al que el exceso geométrico contra el índice cruza cero, interpolado linealmente.

    Si el cruce cae más allá del último peldaño se declara así en vez de extrapolar: una
    extrapolación sobre una curva que solo se ha medido hasta cierto punto sería inventarse el tramo
    que importa. Y si el exceso ya es negativo sin coste alguno, no hay equilibrio que buscar.
    """
    key = f"{window}_{BREAK_EVEN_METRIC}"
    points = sorted(
        (
            (float(row["cost_bps"]), float(row[key]))
            for row in rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ),
        key=lambda point: point[0],
    )
    if not points:
        return {"available": False, "reason": "La ventana no tiene exceso medible."}
    if points[0][1] <= 0:
        return {
            "available": False, "never_positive": True,
            "reason": "El exceso ya es negativo con coste cero: no hay margen que agotar.",
        }
    for (low_cost, low_excess), (high_cost, high_excess) in pairwise(points):
        if high_excess <= 0:
            span = low_excess - high_excess
            crossing = low_cost if span <= 0 else low_cost + (high_cost - low_cost) * (low_excess / span)
            return {
                "available": True,
                "bps_per_trade": crossing,
                "pct_per_trade": crossing / 100.0,
                "round_trip_bps": crossing * 2.0,
                "bracket_bps": [low_cost, high_cost],
            }
    return {
        "available": False, "beyond_ladder": True,
        "last_cost_bps": points[-1][0], "last_excess": points[-1][1],
        "reason": "El exceso sigue siendo positivo en el último peldaño de la escalera.",
    }


# Salvedades que viajan **dentro** del propio artefacto: quien lo lea no tiene por qué haber leído
# la metodología, y un equilibrio sin ellas se cita como si fuera un margen realizable.
CAVEATS = [
    "El escenario bruto (coste cero) es una cota superior que ningún inversor realiza.",
    (
        "El equilibrio de ruta congelada es conservador por construcción: mantiene las decisiones "
        "tomadas con el coste adoptado, y un gestor que pagase más operaría menos."
    ),
    (
        "El exceso de la ventana de selección ya es una cota superior optimista —la cartera es la "
        "mejor de la rejilla—, así que el equilibrio hereda ese optimismo."
    ),
    (
        "Los costes no seleccionan nada: son un supuesto, no una decisión de gestión. Este artefacto "
        "es diagnóstico y no escribe en winner.json, decisions.json ni portfolio_winner.json."
    ),
]


def build_cost_sensitivity(
    evidence_dir: Path, configuration: Mapping[str, Any], *, simulation_dir: Path | None = None,
    ladder: Sequence[float] = COST_LADDER_BPS, resimulate: bool = True,
) -> dict[str, Any]:
    """Bloque completo de sensibilidad a costes sobre la evidencia ya congelada del ganador.

    ``evidence_dir`` aporta la curva de patrimonio ya ejecutada, que es todo lo que necesita la
    familia congelada. ``simulation_dir`` aporta los scores del modelo que la familia resimulada
    vuelve a llevar al backtest; se declara aparte en vez de asumir que ambos viven juntos, para que
    este módulo no dependa en silencio de que la evidencia del ganador tenga enlazados los
    artefactos de modelo.
    """
    from module.studies.config import settings_from_values

    settings = settings_from_values(dict(configuration), profile="balanced")
    adopted_bps = float(settings.commission_bps) + float(settings.slippage_bps)
    equity = pd.read_parquet(evidence_dir / "equity.parquet")

    frozen = frozen_path_curve(equity, settings, ladder)
    resimulated = (
        resimulated_curve(configuration, simulation_dir or evidence_dir, adopted_bps, ladder)
        if resimulate else []
    )
    payload: dict[str, Any] = {
        "adopted_cost_bps": adopted_bps,
        "commission_bps": float(settings.commission_bps),
        "slippage_bps": float(settings.slippage_bps),
        "ladder_bps": [float(step) for step in ladder],
        "break_even_metric": BREAK_EVEN_METRIC,
        "break_even_against": "benchmark",
        "frozen_path": frozen,
        "resimulated": resimulated,
        "break_even": {
            "frozen_path": {window: break_even_bps(frozen, window) for window in WINDOWS},
            "resimulated": {window: break_even_bps(resimulated, window) for window in WINDOWS},
        },
        "caveats": CAVEATS,
    }
    payload["margin_over_adopted"] = _margin(payload["break_even"], adopted_bps)
    return payload


def _margin(break_even: Mapping[str, Mapping[str, Any]], adopted_bps: float) -> dict[str, Any]:
    """Cuántas veces el coste adoptado cabe en el equilibrio. Es el titular del diagnóstico."""
    result: dict[str, Any] = {}
    for family, windows in break_even.items():
        for window, block in windows.items():
            crossing = block.get("bps_per_trade")
            if crossing is None or adopted_bps <= 0:
                continue
            result[f"{family}_{window}"] = float(crossing) / adopted_bps
    return result


def write_cost_sensitivity(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(dict(payload), path)
