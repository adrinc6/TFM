"""Rejilla sistemática de escenarios — el barrido central del TFM.

En vez de escenarios sueltos escritos a mano, genera de forma programática el barrido de regímenes de
entrenamiento del sistema de IA: longitud de ventana, cadencia de reentreno, horizonte de etiqueta y
composición/pesos de agentes (incluidas ablations). Cada escenario reutiliza la maquinaria del runner
(`Scenario` + overrides + caché de scoring); no hay que tocar nada de `module/experiments/`.

Diseño del barrido (interpretable y acotado en cómputo):
- NÚCLEO (producto cartesiano ventana × cadencia): el eje central de la pregunta —¿con cuántos años y
  cada cuánto conviene reentrenar?—. Cada combinación RE-ENTRENA (scoring caro).
- HORIZONTE (una variación por horizonte sobre el baseline): ¿vive el edge a 3/6/12/24 meses?
- PESOS/ABLATIONS (una variación cada una sobre el baseline): apagar el meta-aprendizaje, repartir a
  partes iguales, o dejar un único agente — para aislar de dónde sale (o no) el aprendizaje.

El TAMAÑO DE CARTERA óptimo NO es un eje de la rejilla: se analiza dentro de cada escenario a partir
del ranking guardado (curva breadth top-N en model_walk_forward_diagnostics), sin multiplicar el
barrido.

Lanzar con:
    python -m module.experiments run experiments/rejilla.py

Coste: el núcleo son len(ventanas) x len(cadencias) scorings walk-forward completos; con las
variaciones OFAT, del orden de ~20 escenarios que re-entrenan. Conviene validar antes una subrejilla
(ver `subrejilla`) en DEV_MODE.
"""

from __future__ import annotations

from itertools import product

from module.experiments import Scenario

_Q, _T, _A = "quality_probability", "timing_probability", "alpha_probability"

# Ejes por defecto del barrido. Editar aquí para ampliarlo/reducirlo.
VENTANAS_ANIOS = [2, 4, 6, 8, 10]
CADENCIAS = ["Q", "A"]           # trimestral / anual (mensual se descarta: fundamentales trimestrales)
HORIZONTES_MESES = [3, 6, 24]    # 12 es el baseline; estos son las variaciones
# Presets de pesos del prior de agentes (calidad / timing / alpha). None = prior por defecto.
PESOS_PRESETS = {
    "equal": {_Q: 1 / 3, _T: 1 / 3, _A: 1 / 3},
    "solo_calidad": {_Q: 1.0, _T: 0.0, _A: 0.0},
    "solo_alpha": {_Q: 0.0, _T: 0.0, _A: 1.0},
}

_CADENCIA_NOMBRE = {"Q": "trim", "A": "anual", "M": "mensual"}


def _ventana(anios: int) -> dict:
    # min = max = N: ventana trailing REAL de N años (si min>max, ml.py cae a fallback degenerado).
    return {
        "settings.min_walk_forward_training_years": anios,
        "settings.max_walk_forward_training_years": anios,
    }


def ventana_cadencia(anios: int, cadencia: str) -> Scenario:
    return Scenario(
        name=f"v{anios}a_{_CADENCIA_NOMBRE[cadencia]}",
        why=f"Ventana de {anios} años, reentreno {_CADENCIA_NOMBRE[cadencia]}. Núcleo del barrido: "
            "¿con cuántos años de historia y cada cuánto reentrenar aprende de forma más estable?",
        overrides={**_ventana(anios), "settings.walk_forward_train_frequency": cadencia},
        block="ventana_cadencia",
    )


def horizonte(meses: int) -> Scenario:
    return Scenario(
        name=f"horizonte_{meses}m",
        why=f"Horizonte de etiqueta a {meses}m (baseline 12m). ¿Vive el edge a otro horizonte?",
        overrides={"settings.walk_forward_label_horizon_months": meses},
        block="horizonte",
    )


def pesos(nombre: str, preset: dict) -> Scenario:
    return Scenario(
        name=f"pesos_{nombre}",
        why=f"Prior de agentes = {nombre}. Ablación de composición: aísla de dónde viene el aprendizaje.",
        overrides={"ml.AGENT_PRIOR_WEIGHTS": preset},
        block="pesos_ablacion",
    )


def build_grid(
    ventanas: list[int] = VENTANAS_ANIOS,
    cadencias: list[str] = CADENCIAS,
    horizontes: list[int] = HORIZONTES_MESES,
    pesos_presets: dict[str, dict] = PESOS_PRESETS,
    incluir_ablacion_meta: bool = True,
) -> list[Scenario]:
    """Construye la lista de escenarios: baseline + núcleo (ventana × cadencia) + OFAT de horizonte y
    de pesos/ablations. El runner añade y deduplica el baseline igualmente, pero se incluye aquí para
    que el fichero sea autoexplicativo."""
    scenarios: list[Scenario] = [
        Scenario(name="baseline", why="Config por defecto (ventana 4a, reentreno trimestral, "
                 "horizonte 12m, prior calidad 0.45/0.30/0.25). Referencia del barrido.", block="baseline"),
    ]
    for anios, cadencia in product(ventanas, cadencias):
        scenarios.append(ventana_cadencia(anios, cadencia))
    scenarios.extend(horizonte(m) for m in horizontes)
    scenarios.extend(pesos(nombre, preset) for nombre, preset in pesos_presets.items())
    if incluir_ablacion_meta:
        scenarios.append(Scenario(
            name="sin_meta_aprendido",
            why="Fija los pesos al prior sin reaprenderlos por trimestre (los agentes siguen "
                "puntuando walk-forward). Si el rank-IC OOS apenas baja, el meta-agente no aporta.",
            overrides={"ml.LEARN_META_WEIGHTS": False},
            block="pesos_ablacion",
        ))
    return scenarios


def subrejilla() -> list[Scenario]:
    """Barrido reducido para validar end-to-end (rápido, ideal con DEV_MODE): dos ventanas, ambas
    cadencias, sin variaciones de horizonte ni pesos."""
    return build_grid(ventanas=[4, 8], cadencias=["Q", "A"], horizontes=[], pesos_presets={},
                      incluir_ablacion_meta=False)


SCENARIOS = build_grid()
