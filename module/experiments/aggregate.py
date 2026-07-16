"""Agregación global del barrido y selección automática del sistema final.

Lee las filas de escenarios (comparison) y elige el CONFIG más estable y útil A LO LARGO DE MUCHAS
ERAS — no el que más bate al mercado. Dos piezas para que la elección sea honesta:

1. Métrica compuesta (era de DESARROLLO): premia la media del rank-IC OOS y su ESTABILIDAD (poca
   dispersión, muchos años positivos) y, en menor medida, el lift del top-N (el puente a "la cartera
   gana"). NO usa la alpha cruda: batir al mercado en bruto no es el criterio.
2. Protocolo dev/confirmación: el config se elige SOLO sobre la era de desarrollo
   (años < CONFIRMATION_ERA_START_YEAR) y se COMPRUEBA en la era de confirmación reservada, sin haber
   elegido sobre ella — así "el mejor config" no es un artefacto de sobreajuste por selección.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .metrics import CONFIRMATION_ERA_START_YEAR

log = logging.getLogger(__name__)

# Pesos de la métrica compuesta (suman 1; editable). Prioriza media y estabilidad del rank-IC
# (aprendizaje consistente) sobre el lift del top-N (utilidad); nada de alpha cruda.
COMPOSITE_WEIGHTS = {"mean": 0.40, "stability": 0.30, "positive": 0.20, "lift": 0.10}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Serie numérica de una columna, o una serie de NaN alineada si la columna no existe (tolera
    filas mínimas o de barridos antiguos sin las columnas por era)."""
    if name not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def _minmax(series: pd.Series) -> pd.Series:
    """Normaliza a [0,1] dentro del barrido. Sin dispersión (todos iguales) -> 0.5 neutro."""
    values = pd.to_numeric(series, errors="coerce")
    low, high = values.min(), values.max()
    if not pd.notna(low) or high == low:
        return pd.Series(0.5, index=values.index)
    return (values - low) / (high - low)


def aggregate_scenarios(rows, exp_dir: Path) -> dict:
    """Calcula la métrica compuesta por escenario (sobre desarrollo), elige el ganador y reporta su
    rendimiento en confirmación. Escribe `system_selection.csv` y `system_selection.json`; devuelve el
    veredicto (o {} si no hay métricas de desarrollo)."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    exp_dir = Path(exp_dir)
    if df.empty:
        return {}

    dev_pos = _col(df, "rank_ic_dev_positive_years")
    dev_n = _col(df, "rank_ic_dev_n_years").replace(0, pd.NA)
    df["dev_positive_fraction"] = dev_pos / dev_n

    eligible = df[_col(df, "rank_ic_dev_mean").notna()].copy()
    if eligible.empty:
        log.warning("Agregación: ningún escenario tiene métricas de desarrollo; no se elige sistema final.")
        return {}

    eligible["composite_dev"] = (
        COMPOSITE_WEIGHTS["mean"] * _minmax(_col(eligible, "rank_ic_dev_mean"))
        + COMPOSITE_WEIGHTS["stability"] * (1 - _minmax(_col(eligible, "rank_ic_dev_std")))
        + COMPOSITE_WEIGHTS["positive"] * _minmax(_col(eligible, "dev_positive_fraction"))
        + COMPOSITE_WEIGHTS["lift"] * _minmax(_col(eligible, "top_n_alpha_lift"))
    )
    eligible = eligible.sort_values("composite_dev", ascending=False)
    winner = eligible.iloc[0]

    columns = [
        "name", "block", "composite_dev", "rank_ic_dev_mean", "rank_ic_dev_std",
        "dev_positive_fraction", "rank_ic_conf_mean", "rank_ic_conf_positive_years",
        "rank_ic_conf_n_years", "top_n_alpha_lift", "cumulative_alpha", "information_ratio",
    ]
    selection = eligible[[c for c in columns if c in eligible.columns]]
    selection.to_csv(exp_dir / "system_selection.csv", index=False)

    conf_mean = pd.to_numeric(pd.Series([winner.get("rank_ic_conf_mean")]), errors="coerce").iloc[0]
    verdict = {
        "winner": str(winner["name"]),
        "composite_dev": float(winner["composite_dev"]),
        "rank_ic_dev_mean": float(pd.to_numeric(pd.Series([winner.get("rank_ic_dev_mean")]), errors="coerce").iloc[0]),
        "rank_ic_dev_std": float(pd.to_numeric(pd.Series([winner.get("rank_ic_dev_std")]), errors="coerce").iloc[0]),
        "rank_ic_conf_mean": None if pd.isna(conf_mean) else float(conf_mean),
        "confirmation_start_year": CONFIRMATION_ERA_START_YEAR,
        # Generaliza si en la era reservada el rank-IC sigue siendo positivo (la elección no era
        # un artefacto de la era de desarrollo).
        "generalizes": bool(pd.notna(conf_mean) and conf_mean > 0),
        "weights": COMPOSITE_WEIGHTS,
    }
    with open(exp_dir / "system_selection.json", "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, ensure_ascii=False, indent=2)
    log.info(
        "Sistema final elegido: %s (composite_dev=%.3f, rank-IC desarrollo=%.4f, confirmación=%s, generaliza=%s)",
        verdict["winner"], verdict["composite_dev"], verdict["rank_ic_dev_mean"],
        "n/a" if verdict["rank_ic_conf_mean"] is None else f"{verdict['rank_ic_conf_mean']:.4f}",
        verdict["generalizes"],
    )
    return verdict
