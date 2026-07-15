"""Aplica y restaura los overrides de un escenario sobre la configuración viva del proceso.

El proyecto liga sus parámetros de tres formas distintas, y cada una necesita un mecanismo de
override diferente (esto se verificó leyendo el código, no es incidental):

1. Campos del dataclass ``Settings`` (frozen)  -> se acumulan y se aplican con ``dataclasses.replace``.
2. Globales de un módulo, leídas en tiempo de llamada (``ml.AGENT_PRIOR_WEIGHTS``,
   ``sizing.MAX_POSITION_WEIGHT``, ``portfolio.MIN_ENTRY_SCORE``) -> ``setattr`` sobre ese módulo.
3. Constantes traídas con ``from environment import X`` (p.ej. ``MAX_PORTFOLIO_SIZE`` y
   ``MIN_ROTATION_ADVANTAGE`` en ``portfolio.py``): al importarse quedan ligadas POR NOMBRE en el
   namespace del consumidor, así que reescribir ``environment.X`` NO tiene efecto — hay que
   reescribir el nombre en ``portfolio``/``baselines``. Por eso el mapa de abajo apunta al módulo
   CONSUMIDOR, no a ``environment``.

``apply_overrides`` es un context manager: guarda los valores previos, aplica los del escenario, y
los restaura al salir. Así cada escenario corre aislado en el mismo proceso, sin fugas al siguiente.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any, Iterator

from environment import Settings
from module import ml
from module.backtest import baselines
from module.strategy import portfolio, sizing

from .scenario import Scenario


# strategy.NOMBRE -> lista de módulos donde ese nombre debe reescribirse para surtir efecto.
# Un nombre puede vivir en más de un módulo (p.ej. MAX_PORTFOLIO_SIZE lo importan portfolio y
# baselines); hay que actualizarlo en todos. Los nombres que son globales locales de su módulo
# (MIN_ENTRY_SCORE, MAX_POSITION_WEIGHT, ...) apuntan a un único módulo.
_STRATEGY_TARGETS: dict[str, list[Any]] = {
    "MAX_PORTFOLIO_SIZE": [portfolio, baselines],
    "MIN_PORTFOLIO_SIZE": [portfolio],
    "MIN_ROTATION_ADVANTAGE": [portfolio],
    "MIN_SCORE_ADVANTAGE_TO_REPLACE": [portfolio],
    "MIN_CONVICTION_ADVANTAGE": [portfolio],
    "MIN_OPPORTUNITY_COST_THRESHOLD": [portfolio],
    "MIN_ENTRY_SCORE": [portfolio],
    "MIN_ENTRY_QUALITY": [portfolio],
    "MIN_HOLD_SCORE": [portfolio],
    "MIN_BUY_TODAY_SCORE": [portfolio],
    "MIN_HOLD_MONTHS_BEFORE_ROTATION": [portfolio],
    "STOP_LOSS_THRESHOLD": [portfolio],
    "MAX_POSITION_WEIGHT": [sizing],
    "MIN_POSITION_WEIGHT": [sizing],
    "EQUAL_WEIGHT_SHARE": [sizing],
    "CONVICTION_WEIGHT_SHARE": [sizing],
    "CONVICTION_CONVEXITY": [sizing],
}

# Constantes de ml.py que un escenario puede overridar. Se listan explícitamente para fallar pronto
# ante un typo en vez de crear un atributo nuevo silenciosamente.
_ML_NAMES = {
    "AGENT_PRIOR_WEIGHTS",
    "LEARN_META_WEIGHTS",
    "RANDOM_STATE",
    "META_WEIGHT_FLOOR",
    "CONSISTENCY_LAMBDA",
    "N_CONSISTENCY_FOLDS",
    "LGBM_PARAM_GRID",
    "LABEL_HORIZON_CANDIDATES_MONTHS",
}

_SETTINGS_FIELDS = {f.name for f in dataclasses.fields(Settings)}


def split_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reparte los overrides en (settings, ml, strategy) validando nombres. Falla pronto ante un
    namespace o nombre desconocido para no dejar pasar un typo que no haría nada."""
    settings_ovr: dict[str, Any] = {}
    ml_ovr: dict[str, Any] = {}
    strategy_ovr: dict[str, Any] = {}
    for key, value in overrides.items():
        namespace, name = key.split(".", 1)
        if namespace == "settings":
            if name not in _SETTINGS_FIELDS:
                raise KeyError(f"'{name}' no es un campo de Settings.")
            settings_ovr[name] = value
        elif namespace == "ml":
            if name not in _ML_NAMES:
                raise KeyError(f"'ml.{name}' no está en la lista de constantes overridables de ml.py.")
            ml_ovr[name] = value
        elif namespace == "strategy":
            if name not in _STRATEGY_TARGETS:
                raise KeyError(f"'strategy.{name}' no está en el mapa de constantes de estrategia.")
            strategy_ovr[name] = value
        else:
            raise KeyError(f"Namespace desconocido en override '{key}'.")
    return settings_ovr, ml_ovr, strategy_ovr


@contextmanager
def apply_overrides(scenario: Scenario, base_settings: Settings) -> Iterator[Settings]:
    """Aplica los overrides del escenario y cede el ``Settings`` efectivo. Restaura todo al salir.

    Los overrides de ``Settings`` se materializan con ``dataclasses.replace`` sobre ``base_settings``
    (frozen), de modo que el ``Settings`` cedido lleva ya los campos cambiados. Los de ``ml`` y
    ``strategy`` se escriben sobre los módulos vivos y se revierten a su valor original en el
    ``finally``.
    """
    settings_ovr, ml_ovr, strategy_ovr = split_overrides(scenario.overrides)
    effective_settings = dataclasses.replace(base_settings, **settings_ovr) if settings_ovr else base_settings

    saved: list[tuple[Any, str, Any]] = []
    try:
        for name, value in ml_ovr.items():
            saved.append((ml, name, getattr(ml, name)))
            setattr(ml, name, value)
        for name, value in strategy_ovr.items():
            for module in _STRATEGY_TARGETS[name]:
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, value)
        # REVIEW_TOP_N se calcula en portfolio a partir de MAX_PORTFOLIO_SIZE en tiempo de import;
        # si el escenario cambia MAX_PORTFOLIO_SIZE hay que recomputarlo para que sea coherente.
        if "MAX_PORTFOLIO_SIZE" in strategy_ovr:
            saved.append((portfolio, "REVIEW_TOP_N", portfolio.REVIEW_TOP_N))
            portfolio.REVIEW_TOP_N = max(strategy_ovr["MAX_PORTFOLIO_SIZE"] * 3, 20)
        yield effective_settings
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)
