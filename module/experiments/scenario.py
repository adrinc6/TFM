"""Un escenario de experimento: un nombre, la hipótesis que contrasta y los overrides a aplicar.

Los overrides son un diccionario `"namespace.NOMBRE" -> valor`. El namespace indica QUÉ mecanismo
de configuración toca cada clave (ver overrides.py):

- ``settings.*``  -> un campo del dataclass ``Settings`` (p.ej. ``settings.transaction_cost_bps``).
- ``ml.*``        -> una constante global de ``module.ml`` (p.ej. ``ml.AGENT_PRIOR_WEIGHTS``).
- ``strategy.*``  -> una constante de estrategia (``sizing``/``portfolio``/``baselines``); el módulo
                     concreto lo resuelve el mapa de overrides.py.

Se declaran en un fichero Python (``experiments/escenarios_*.py``) como ``SCENARIOS: list[Scenario]``.
Elegir Python en vez de YAML evita añadir una dependencia y permite comentar el porqué de cada
escenario junto a él.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    why: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Un escenario necesita un nombre no vacío.")
        for key in self.overrides:
            namespace = key.split(".", 1)[0]
            if namespace not in {"settings", "ml", "strategy"}:
                raise ValueError(
                    f"Override '{key}' con namespace desconocido '{namespace}'. "
                    "Usa 'settings.', 'ml.' o 'strategy.'."
                )

    @property
    def is_baseline(self) -> bool:
        return not self.overrides
