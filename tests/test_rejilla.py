"""Generador de la rejilla de escenarios (experiments/rejilla.py).

Fija que el barrido produce escenarios válidos y sin nombres duplicados, que el núcleo es el producto
cartesiano ventana × cadencia, y que todos los overrides son reconocidos por el runner.
"""

from __future__ import annotations

from experiments.rejilla import build_grid, subrejilla, CADENCIAS, VENTANAS_ANIOS
from module.experiments.overrides import split_overrides


def test_todos_los_overrides_son_validos_para_el_runner():
    for scenario in build_grid():
        # split_overrides revienta ante un namespace o nombre desconocido: valida toda la rejilla.
        split_overrides(scenario.overrides)


def test_nombres_unicos():
    names = [s.name for s in build_grid()]
    assert len(names) == len(set(names))


def test_nucleo_es_producto_cartesiano_ventana_por_cadencia():
    grid = build_grid()
    core = [s for s in grid if s.block == "ventana_cadencia"]
    assert len(core) == len(VENTANAS_ANIOS) * len(CADENCIAS)
    # Cada combinación fija min = max = N (ventana trailing real) y la cadencia.
    for scenario in core:
        assert scenario.overrides["settings.min_walk_forward_training_years"] == \
            scenario.overrides["settings.max_walk_forward_training_years"]
        assert scenario.overrides["settings.walk_forward_train_frequency"] in CADENCIAS


def test_subrejilla_es_reducida_y_valida():
    sub = subrejilla()
    assert 0 < len(sub) < len(build_grid())
    for scenario in sub:
        split_overrides(scenario.overrides)


if __name__ == "__main__":  # pragma: no cover
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
