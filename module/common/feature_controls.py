"""Feature list resolution helpers controlled by environment settings."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence


def _unique(seq: Iterable[str]) -> list[str]:
    """Returns a deduplicated list preserving insertion order.

    Args:
        seq (Iterable[str]): Sequence of strings to deduplicate.

    Returns:
        list[str]: Ordered list with duplicates removed.
    """
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
    """Resolves the final feature column list using include/exclude controls.

    Rules:
    - ``include_cols`` is mandatory and authoritative (even if empty).
    - ``default_cols`` is deprecated and not used for training selection.
    - ``exclude_cols`` is informational (does not alter training selection).
    - Only columns that exist in ``available_cols`` are returned.

    Args:
        default_cols (Sequence[str]): Deprecated fallback column list (ignored).
        available_cols (Sequence[str]): Columns present in the actual DataFrame.
        include_cols (Sequence[str] | None): Explicit list of columns to include.
            Must not be None; raise ValueError if None is passed.
        exclude_cols (Sequence[str] | None): Columns to log as excluded
            (informational only; does not remove columns from the result).
        logger (logging.Logger): Logger instance for diagnostic messages.
        owner (str): Name of the calling agent or component (used in log messages).

    Returns:
        list[str]: Deduplicated list of feature columns that are both in
            ``include_cols`` and present in ``available_cols``.

    Raises:
        ValueError: If ``include_cols`` is None.
    """
    available = set(str(c) for c in available_cols)
    # kept for backward compatibility: acknowledge the deprecated `default_cols` parameter
    _ = default_cols
    if include_cols is None:
        raise ValueError(
            f"[{owner}] include_cols cannot be None: define *_FEATURE_COLUMNS in environment.py"
        )
    include = _unique(include_cols)
    exclude = set(_unique(exclude_cols or []))

    base = include
    missing = [c for c in base if c not in available]
    if missing:
        logger.info("[%s] Ignoring include columns not available: %s", owner, ", ".join(missing))

    final = [c for c in base if c in available]

    # Log columns listed in exclude for informational purposes
    listed_in_exclude = [c for c in base if c in exclude and c in available]
    if listed_in_exclude:
        logger.info("[%s] Columns listed in exclude (informational): %s", owner, ", ".join(listed_in_exclude))

    return final
