"""Feature list resolution helpers controlled by environment settings."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence


def _unique(seq: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in seq:
        s = str(item)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve_feature_columns(
    *,
    default_cols: Sequence[str],
    available_cols: Sequence[str],
    include_cols: Sequence[str] | None,
    exclude_cols: Sequence[str] | None,
    logger: logging.Logger,
    owner: str,
) -> list[str]:
    """Resolve final feature columns using include/exclude controls.

    Rules:
    - include_cols es la lista autoritativa cuando se proporciona (aunque esté vacía).
    - default_cols solo se usa por compatibilidad si include_cols es None.
    - exclude_cols es informativa (no altera selección de entrenamiento).
    - Only columns available in the input frame survive.
    """
    available = set(str(c) for c in available_cols)
    include = _unique(include_cols) if include_cols is not None else None
    exclude = set(_unique(exclude_cols or []))

    if include is not None:
        base = include
        missing = [c for c in base if c not in available]
        if missing:
            logger.info("[%s] Ignorando columnas include no disponibles: %s", owner, ", ".join(missing))
    else:
        base = _unique(default_cols)

    final = [c for c in base if c in available]

    listed_in_exclude = [c for c in base if c in exclude and c in available]
    if listed_in_exclude:
        logger.info("[%s] Columnas listadas en exclude (informativo): %s", owner, ", ".join(listed_in_exclude))

    return final
