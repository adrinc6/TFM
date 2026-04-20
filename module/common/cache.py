"""Simple file-based cache keyed by execution context."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any



def _canonicalize(value: Any) -> Any:
    """Converts values to stable, JSON-serializable structures for hashing.

    Recursively processes dicts (sorted by key), lists/tuples/sets, and
    special types like datetime, date, and Path.

    Args:
        value (Any): The value to canonicalise.

    Returns:
        Any: A JSON-serializable representation of the value.
    """
    if isinstance(value, dict):
        return {str(k): _canonicalize(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple, set)):
        return [_canonicalize(v) for v in list(value)]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


class CacheManager:
    """File-based cache with a deterministic key derived from a context dictionary.

    The cache key is a 16-character SHA-256 hex digest of the JSON-serialized
    context. Each unique context maps to a dedicated subdirectory under
    ``cache_dir / namespace / key``.

    Example:
        >>> cm = CacheManager("cache", {"tickers": ["AAPL"], "year": 2023})
        >>> cm.save_json("result", {"accuracy": 0.85})
        >>> data = cm.load_json("result")
    """

    def __init__(self, cache_dir: str | Path, context: dict[str, Any], namespace: str = "default") -> None:
        """Initialises the cache manager and creates the run directory.

        Args:
            cache_dir (str | Path): Root directory for all cache files.
            context (dict[str, Any]): Arbitrary context dictionary that uniquely
                identifies the computation. Used to derive the cache key.
            namespace (str): Logical grouping label for the cache entry.
        """
        self.namespace = str(namespace)
        self.cache_root = Path(cache_dir)
        self.context = _canonicalize(context)

        # Derive a short deterministic key from the JSON-serialized context
        payload = json.dumps(self.context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        self.run_dir = self.cache_root / self.namespace / self.key
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._write_manifest(payload)

    def _write_manifest(self, payload: str) -> None:
        """Writes a JSON manifest file recording the cache context.

        Args:
            payload (str): The raw JSON string of the context for record-keeping.
        """
        manifest = {
            "namespace": self.namespace,
            "key": self.key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": self.context,
            "context_json": payload,
        }
        self.save_json("manifest", manifest)

    def path(self, name: str, ext: str) -> Path:
        """Returns the full path for a named cache file.

        Args:
            name (str): Base name of the file (without extension).
            ext (str): File extension with or without leading dot.

        Returns:
            Path: Full path to the cache file.
        """
        suffix = ext if ext.startswith(".") else f".{ext}"
        return self.run_dir / f"{name}{suffix}"

    def exists(self, name: str, ext: str) -> bool:
        """Checks whether a named cache file exists on disk.

        Args:
            name (str): Base name of the file.
            ext (str): File extension.

        Returns:
            bool: True if the file exists.
        """
        return self.path(name, ext).exists()

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        """Serialises a dictionary to a JSON file in the cache directory.

        Args:
            name (str): Base name of the output file.
            payload (dict[str, Any]): Data to serialise.

        Returns:
            Path: Path to the written file.
        """
        out = self.path(name, "json")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return out

    def load_json(self, name: str) -> dict[str, Any] | None:
        """Loads a JSON file from the cache directory.

        Args:
            name (str): Base name of the file to load.

        Returns:
            dict[str, Any] | None: Parsed dictionary, or None if the file does
                not exist or cannot be parsed.
        """
        path = self.path(name, "json")
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_pickle(self, name: str, obj: Any) -> Path:
        """Saves an arbitrary Python object as a pickle file.

        Args:
            name (str): Base name of the output file.
            obj (Any): Object to serialise.

        Returns:
            Path: Path to the written file.
        """
        import pandas as pd

        out = self.path(name, "pkl")
        pd.to_pickle(obj, out)
        return out

    def load_pickle(self, name: str) -> Any | None:
        """Loads a pickle file from the cache directory.

        Args:
            name (str): Base name of the file to load.

        Returns:
            Any | None: Deserialized object, or None if the file does not exist
                or cannot be loaded.
        """
        import pandas as pd

        path = self.path(name, "pkl")
        if not path.exists():
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None
