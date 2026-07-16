"""Ejemplo de referencia — barrido del prior de pesos del meta-agente (AGENT_PRIOR_WEIGHTS).

Reconstruye de forma versionada y reproducible el barrido de pesos que el comentario de
module/ml.py menciona (el antiguo scripts/buscar_pesos_meta.py, ausente del repo). Traza el trade-off
estabilidad (dispersión del rank-IC entre años) vs. alpha realizada: tiltar más a calidad sube la
estabilidad pero cede algo de alpha; el 0.45/0.30/0.25 por defecto es el punto medio adoptado.

Cambian el scoring -> RE-ENTRENAN.

Lanzar con:
    python -m module.experiments run experiments/escenarios_pesos_meta.py
"""

from module.experiments import Scenario

_Q, _T, _A = "quality_probability", "timing_probability", "alpha_probability"


def _pesos(q: float, t: float, a: float) -> Scenario:
    return Scenario(
        name=f"pesos_{q:.2f}_{t:.2f}_{a:.2f}".replace(".", ""),
        why=f"Prior calidad/timing/alpha = {q}/{t}/{a}. Punto del trade-off estabilidad vs. alpha.",
        overrides={"ml.AGENT_PRIOR_WEIGHTS": {_Q: q, _T: t, _A: a}},
        block="pesos_meta",
    )


SCENARIOS = [
    Scenario(name="baseline", why="Prior por defecto 0.45/0.30/0.25 (punto medio adoptado). Referencia.", block="baseline"),
    _pesos(0.30, 0.35, 0.35),  # tilt a timing/alpha: más alpha, menos estable
    _pesos(0.60, 0.25, 0.15),  # tilt fuerte a calidad: más estable, cede alpha
]
