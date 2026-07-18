"""Perfiles de inversor: selección de cartera por estilo, sobre las buenas del modelo.

El sistema explica por que cada accion esta arriba: cada agente (calidad, momentum, valor)
produce un rango por ticker (`quality_rank`, `momentum_rank`, `value_rank`) y el meta los combina
(`meta_rank`). Un perfil de inversor NO coge siempre el top-N del meta: dentro de las **buenas**
acciones (las del percentil alto del `meta_rank`), reordena segun su estilo. Asi se puede medir
como le habria ido a un inversor conservador, agresivo, value, etc., usando las mismas señales.

Cada perfil produce un `meta_rank` alternativo (el `profile_score`) que la cartera consume igual
que el meta_rank normal. Es una transformacion determinista de los rangos ya existentes: no
reentrena nada, no mira al futuro.
"""

from __future__ import annotations

import pandas as pd

# Percentil del meta_rank a partir del cual una accion se considera "buena" y entra al universo
# elegible del perfil. Debajo de esto, el modelo no la recomienda y ningun perfil la elige.
GOOD_THRESHOLD = 0.60

# Cada perfil pondera los rangos de los agentes. Los pesos suman 1. El score del perfil es la
# combinacion ponderada, re-rankeada entre las buenas. Todos parten del meta como base de calidad.
PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    # El meta puro: referencia (sistema base sin sesgo de estilo).
    "balanced": {"meta_rank": 1.0},
    # Conservador: calidad y estabilidad por encima de todo; algo de valor, nada de momentum.
    "conservative": {"quality_rank": 0.6, "value_rank": 0.3, "meta_rank": 0.1},
    # Agresivo: momentum y crecimiento; busca el que mas sube.
    "aggressive": {"momentum_rank": 0.7, "meta_rank": 0.3},
    # Value: barato y bueno (P/E, P/B bajos entre las de calidad).
    "value": {"value_rank": 0.7, "quality_rank": 0.2, "meta_rank": 0.1},
    # Calidad pura: solo el mejor negocio (ROE, margenes, poca deuda).
    "quality": {"quality_rank": 0.8, "meta_rank": 0.2},
    # Momentum puro: solo fuerza relativa reciente.
    "momentum": {"momentum_rank": 0.8, "meta_rank": 0.2},
    # GARP (growth at reasonable price): equilibrio valor + momentum de calidad.
    "garp": {"value_rank": 0.4, "quality_rank": 0.3, "momentum_rank": 0.3},
    # Contrarian: buenas del meta pero con momentum reciente BAJO (apuesta a reversion).
    "contrarian": {"quality_rank": 0.5, "value_rank": 0.3, "momentum_rank": -0.2, "meta_rank": 0.4},
}

PROFILE_NAMES = tuple(PROFILE_WEIGHTS)


def apply_profile(scores: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Devuelve `scores` con el `meta_rank` reemplazado por el ranking del perfil.

    - `balanced` devuelve el meta_rank tal cual (referencia).
    - Los demas: entre las acciones buenas (meta_rank >= GOOD_THRESHOLD) reordenan por la
      combinacion ponderada de los rangos de los agentes; las que no son buenas quedan por debajo
      (meta_rank del perfil = 0) para que nunca entren en la cartera.
    """
    if profile == "balanced" or profile not in PROFILE_WEIGHTS:
        return scores

    weights = PROFILE_WEIGHTS[profile]
    frame = scores.copy()
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
    return frame
