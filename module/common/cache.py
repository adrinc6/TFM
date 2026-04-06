"""Simple file-based cache keyed by execution context."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any



def _canonicalize(value: Any) -> Any:
    """Convert values to stable JSON-serializable structures for hashing."""
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
    """File cache with deterministic key from a context dictionary."""

    def __init__(self, cache_dir: str | Path, context: dict[str, Any], namespace: str = "default") -> None:
        self.namespace = str(namespace)
        self.cache_root = Path(cache_dir)
        self.context = _canonicalize(context)

        payload = json.dumps(self.context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        self.run_dir = self.cache_root / self.namespace / self.key
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._write_manifest(payload)

    def _write_manifest(self, payload: str) -> None:
        manifest = {
            "namespace": self.namespace,
            "key": self.key,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "context": self.context,
            "context_json": payload,
        }
        self.save_json("manifest", manifest)

    def path(self, name: str, ext: str) -> Path:
        suffix = ext if ext.startswith(".") else f".{ext}"
        return self.run_dir / f"{name}{suffix}"

    def exists(self, name: str, ext: str) -> bool:
        return self.path(name, ext).exists()

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        out = self.path(name, "json")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return out

    def load_json(self, name: str) -> dict[str, Any] | None:
        path = self.path(name, "json")
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_pickle(self, name: str, obj: Any) -> Path:
        import pandas as pd

        out = self.path(name, "pkl")
        pd.to_pickle(obj, out)
        return out

    def load_pickle(self, name: str) -> Any | None:
        import pandas as pd

        path = self.path(name, "pkl")
        if not path.exists():
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None
