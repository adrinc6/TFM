"""Perfiles de inversor: selección de cartera por estilo, sobre las buenas del modelo.

El sistema explica por que cada accion esta arriba: cada agente (quality, value, growth, momentum,
risk) produce un rango por ticker (`quality_rank`, `value_rank`, `growth_rank`, `momentum_rank`,
`risk_rank`) y el meta los combina (`meta_rank`). Un perfil de inversor NO coge siempre el top-N del
meta: dentro de las **buenas** acciones (las del percentil alto del `meta_rank`), reordena segun su
estilo. Asi se puede medir como le habria ido a un inversor value, growth, momentum, contrarian,
defensivo, etc., usando las mismas señales.

Cada perfil produce un `meta_rank` alternativo (el `profile_score`) que la cartera consume igual
que el meta_rank normal. Es una transformacion determinista de los rangos ya existentes: no
reentrena nada, no mira al futuro.
"""

from __future__ import annotations

import pandas as pd

# Percentil del meta_rank a partir del cual una accion se considera "buena" y entra al universo
# elegible del perfil. Debajo de esto, el modelo no la recomienda y ningun perfil la elige.
GOOD_THRESHOLD = 0.60

# Cada perfil pondera los rangos de los agentes (quality, value, growth, momentum, risk) y, si
# quiere, el meta_rank como base. Los pesos pueden ser negativos cuando el estilo apuesta EN CONTRA
# de un agente (p.ej. el contrarian contra el momentum reciente). El score del perfil es la
# combinacion ponderada, re-rankeada entre las buenas. Nota sobre risk_rank: en el catalogo los
# factores de riesgo van en direccion inversa (rank alto = MENOS riesgo), asi que un peso positivo
# en risk_rank prioriza baja volatilidad y uno negativo tolera/busca volatilidad.
#
# El conjunto se diseño para representar arquetipos de inversor reconocibles con solapamiento minimo
# (correlacion esperada entre carteras baja; momentum y contrarian son polos opuestos por diseño).
PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    # El meta puro: referencia (sistema base sin sesgo de estilo).
    "balanced": {"meta_rank": 1.0},
    # Growth: crecimiento como motor, pero crecimiento DE CALIDAD; momentum confirma la tendencia.
    "growth": {"growth_rank": 0.60, "quality_rank": 0.25, "momentum_rank": 0.15},
    # Value: empresas infravaloradas; calidad y riesgo filtran las value traps.
    "value": {"value_rank": 0.70, "quality_rank": 0.15, "risk_rank": 0.15},
    # Quality compounder (tipo Buffett): negocios excelentes y duraderos; calidad manda.
    "quality": {"quality_rank": 0.70, "growth_rank": 0.20, "value_rank": 0.10},
    # Momentum trader: pura fuerza de precio y acepta la volatilidad que la acompaña (risk negativo).
    "momentum": {"momentum_rank": 0.75, "risk_rank": -0.25},
    # Contrarian: compra lo castigado, apuesta EN CONTRA del momentum reciente (peso negativo),
    # en nombres baratos y vigilando el riesgo para no comprar problemas estructurales.
    "contrarian": {"momentum_rank": -0.55, "value_rank": 0.30, "risk_rank": 0.15},
    # Defensivo / low-vol: preservacion de capital; baja volatilidad (risk_rank alto) y calidad.
    "defensive": {"risk_rank": 0.60, "quality_rank": 0.35, "value_rank": 0.05},
    # GARP (Growth At a Reasonable Price): crece pero con disciplina de precio (value junto a growth).
    "garp": {"growth_rank": 0.40, "value_rank": 0.35, "quality_rank": 0.25},
}

PROFILE_NAMES = tuple(PROFILE_WEIGHTS)


def apply_profile(
    scores: pd.DataFrame, profile: str, targets: pd.DataFrame | None = None,
    *, horizon_months: int,
) -> pd.DataFrame:
    """Devuelve `scores` con el `meta_rank` reemplazado por el ranking del perfil.

    - `balanced` devuelve el meta_rank tal cual (referencia).
    - Los demas: entre las acciones buenas (meta_rank >= GOOD_THRESHOLD) reordenan por la
      combinacion ponderada de los rangos de los agentes; las que no son buenas quedan por debajo
      (meta_rank del perfil = 0) para que nunca entren en la cartera.

    Cuando el perfil reordena, **la calibración se recalcula sobre el ranking nuevo**. El
    `expected_excess_return` es una función del rank: dejarlo como estaba lo referiría a un orden
    que ya no se usa, y como la cartera decide con umbrales en puntos básicos sobre esa columna, el
    perfil operaría con expectativas ajenas a sus propias posiciones.
    """
    if profile == "balanced" or profile not in PROFILE_WEIGHTS:
        return scores

    weights = PROFILE_WEIGHTS[profile]
    frame = scores.copy()

    # Un perfil pondera rangos de agente concretos; si falta la columna de alguno (p.ej. el finalista
    # se entreno sin el agente `quality`), saltarla en silencio deformaria el estilo sin aviso y el
    # resultado ya no representaría el arquetipo. Se falla en voz alta y el Study lo registra como
    # diagnóstico no aplicable.
    missing = [column for column in weights if column != "meta_rank" and column not in frame.columns]
    if missing:
        raise ValueError(
            f"El perfil '{profile}' necesita los rangos {missing}, ausentes en los scores. "
            "Evalua los perfiles sobre un modelo con los agentes correspondientes.")

    good = pd.to_numeric(frame["meta_rank"], errors="coerce") >= GOOD_THRESHOLD

    combined = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        if column in frame.columns:
            combined = combined + weight * pd.to_numeric(frame[column], errors="coerce").fillna(0.5)

    # Solo las buenas puntuan; el resto a 0. Re-rankear dentro de cada snapshot entre las buenas.
    combined = combined.where(good, other=float("-inf"))
    profile_rank = combined.groupby(frame["snapshot_date"]).rank(method="average", pct=True)
    # Las no-buenas (que tenian -inf) reciben rank alto por el pct; forzarlas a 0.
    profile_rank = profile_rank.where(good, other=0.0)
    frame["meta_rank"] = profile_rank
    if targets is None or "expected_excess_return" not in frame.columns:
        return frame
    from module.evaluation.signal_diagnostics import calibrated_alpha_path

    recalibrated = calibrated_alpha_path(frame, targets, horizon_months=horizon_months)
    return frame.drop(columns=["expected_excess_return"]).merge(
        recalibrated[["ticker", "snapshot_date", "expected_excess_return"]],
        on=["ticker", "snapshot_date"], how="left", validate="one_to_one",
    )
