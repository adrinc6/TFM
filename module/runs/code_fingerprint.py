"""Huella de código por etapa: identifica la versión del código que ejecuta cada etapa.

Sustituye a ``git_revision()`` en las claves de caché. Un commit que solo toca el meta-agente
no debe invalidar la caché de ``dataset`` ni ``features``; con una huella acotada al código que
realmente ejecuta cada etapa, la reutilización sigue siendo exacta (si cambia el código de una
etapa, cambia su clave; si no, se reutiliza el artefacto idéntico) sin recomputar de más.

La huella se calcula como el sha256 del contenido de la clausura transitiva de imports locales
(``module.*``, ``escenarios.*`` y ``environment``) a partir del módulo de entrada de la etapa.
Es autosuficiente: añadir una dependencia nueva a una etapa la incorpora automáticamente, sin
mantener a mano listas frágiles de ficheros.
"""

from __future__ import annotations

import ast
import hashlib
from functools import lru_cache
from pathlib import Path

from environment import PROJECT_ROOT
from module.common.utils import sha256_file

# Módulo de entrada (por su nombre importable) que ejecuta cada etapa cacheable.
STAGE_ENTRY_MODULES: dict[str, str] = {
    "dataset": "module.data.dataset",
    "features": "module.modeling.features",
    "agents": "module.modeling.agents",
    "backtest": "module.evaluation.backtest",
}

# Prefijos de paquetes de primera parte cuyo código forma parte de la huella. Todo lo demás
# (stdlib, pandas, numpy, sklearn, lightgbm...) queda fuera: sus versiones se fijan por
# requirements y no forman parte del código del repositorio.
_FIRST_PARTY_PREFIXES = ("module.", "escenarios.", "environment")


def _module_to_path(module_name: str) -> Path | None:
    """Resuelve un nombre importable de primera parte a su fichero fuente dentro del repo."""
    relative = module_name.replace(".", "/")
    candidate = PROJECT_ROOT / f"{relative}.py"
    if candidate.exists():
        return candidate
    package_init = PROJECT_ROOT / relative / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def _is_first_party(module_name: str) -> bool:
    return module_name == "environment" or module_name.startswith(_FIRST_PARTY_PREFIXES)


def _iter_imports(source: str) -> set[str]:
    """Nombres de módulo importados en un fichero (import X / from X import ...)."""
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _closure_files(entry_module: str) -> list[Path]:
    """Clausura transitiva de ficheros de primera parte alcanzables desde ``entry_module``."""
    seen: set[str] = set()
    files: dict[str, Path] = {}
    pending = [entry_module]
    while pending:
        name = pending.pop()
        if name in seen or not _is_first_party(name):
            continue
        seen.add(name)
        path = _module_to_path(name)
        if path is None:
            continue
        files[name] = path
        pending.extend(_iter_imports(path.read_text(encoding="utf-8")))
    return [files[name] for name in sorted(files)]


@lru_cache(maxsize=None)
def stage_code_fingerprint(stage: str) -> str:
    """sha256 estable del código que ejecuta ``stage`` (clausura de imports de primera parte)."""
    entry = STAGE_ENTRY_MODULES.get(stage)
    if entry is None:
        raise ValueError(f"Etapa sin módulo de entrada registrado: {stage!r}")
    digest = hashlib.sha256()
    for path in _closure_files(entry):
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def combined_code_fingerprint() -> str:
    """Huella del código de todas las etapas: identidad de código a nivel de run completo."""
    digest = hashlib.sha256()
    for stage in sorted(STAGE_ENTRY_MODULES):
        digest.update(stage.encode("utf-8"))
        digest.update(stage_code_fingerprint(stage).encode("utf-8"))
    return digest.hexdigest()
