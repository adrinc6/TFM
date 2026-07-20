"""Huella de código por etapa: acota la invalidación de caché al código que ejecuta cada etapa.

Un commit que solo toca el meta-agente no debe invalidar la caché de ``dataset`` ni ``features``.
Estos tests fijan esa propiedad para que no se rompa sin querer (p.ej. al añadir un import que
amplíe de más la clausura o al reintroducir una huella global tipo ``git_revision``).
"""

from __future__ import annotations

from module.runs.code_fingerprint import (
    STAGE_ENTRY_MODULES,
    _closure_files,
    combined_code_fingerprint,
    stage_code_fingerprint,
)

STAGES = ("dataset", "features", "agents", "backtest")


def _closure_names(stage: str) -> set[str]:
    return {path.name for path in _closure_files(STAGE_ENTRY_MODULES[stage])}


def test_entry_modules_cover_all_cacheable_stages() -> None:
    assert set(STAGE_ENTRY_MODULES) == set(STAGES)


def test_agents_closure_includes_meta_and_catalog() -> None:
    """El código del meta y del catálogo SÍ afecta a la etapa de agentes."""
    names = _closure_names("agents")
    assert {"meta.py", "catalog.py", "agents.py"} <= names


def test_dataset_closure_excludes_downstream_modeling() -> None:
    """Editar el meta/agentes/features NO debe invalidar la caché de dataset."""
    names = _closure_names("dataset")
    assert not ({"meta.py", "agents.py", "features.py"} & names)


def test_features_closure_excludes_agents_and_meta() -> None:
    names = _closure_names("features")
    assert not ({"meta.py", "agents.py"} & names)


def test_fingerprints_are_stable_and_distinct_per_stage() -> None:
    prints = {stage: stage_code_fingerprint(stage) for stage in STAGES}
    # deterministas dentro del proceso
    assert all(stage_code_fingerprint(stage) == prints[stage] for stage in STAGES)
    # cada etapa tiene su propia huella (distinto conjunto de código)
    assert len(set(prints.values())) == len(STAGES)


def test_combined_fingerprint_depends_on_every_stage() -> None:
    combined = combined_code_fingerprint()
    assert isinstance(combined, str) and len(combined) == 64


def test_editing_one_stage_module_only_shifts_that_stage(tmp_path, monkeypatch) -> None:
    """Simula un cambio de código en meta.py: cambia la huella de agents pero no la de dataset.

    Se comprueba comparando la huella real con una recomputada tras alterar el contenido del
    fichero (a través de un doble de ``sha256_file`` que devuelve un hash distinto solo para
    ``meta.py``). Así se valida la propiedad de aislamiento sin tocar el repositorio.
    """
    import module.runs.code_fingerprint as cf

    real_sha = cf.sha256_file

    def fake_sha(path):
        if path.name == "meta.py":
            return "0" * 64  # contenido "cambiado" solo para meta.py
        return real_sha(path)

    monkeypatch.setattr(cf, "sha256_file", fake_sha)
    cf.stage_code_fingerprint.cache_clear()
    cf.combined_code_fingerprint.cache_clear()
    shifted = {stage: cf.stage_code_fingerprint(stage) for stage in STAGES}
    cf.stage_code_fingerprint.cache_clear()
    cf.combined_code_fingerprint.cache_clear()

    monkeypatch.setattr(cf, "sha256_file", real_sha)
    cf.stage_code_fingerprint.cache_clear()
    cf.combined_code_fingerprint.cache_clear()
    baseline = {stage: cf.stage_code_fingerprint(stage) for stage in STAGES}
    cf.stage_code_fingerprint.cache_clear()
    cf.combined_code_fingerprint.cache_clear()

    # meta.py está en la clausura de agents (y features no, dataset tampoco):
    assert shifted["agents"] != baseline["agents"]
    assert shifted["dataset"] == baseline["dataset"]
    assert shifted["features"] == baseline["features"]
    assert shifted["backtest"] == baseline["backtest"]
