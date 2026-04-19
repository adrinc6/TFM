"""Sector normalization utilities.

NOTE: SectorNormalizer has been deprecated with the introduction of
sector-specialized agents. Features are no longer normalized to sector means.
"""

from __future__ import annotations

from typing import Dict, Tuple


class SectorNormalizer:
    """
    DEPRECATED: Z-score relativo al sector.
    
    This class is no longer used since agents are now sector-specialized.
    Kept for backward compatibility only.
    """

    COLS = []  # Empty list - no features are normalized anymore

    def __init__(self, min_peers: int = 3):
        self.min_peers = min_peers
        self._stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
