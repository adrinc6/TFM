"""Download registry for Finnhub/Yahoo fetches."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Registry:
    """
    Tracks which endpoints were already downloaded per ticker or group.
    Persisted under <base_dir>/_registry.json.
    """

    def __init__(self, base_dir: Path):
        self.path = base_dir / "_registry.json"
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def is_done(self, group: str, endpoint: str) -> bool:
        entry = self._data.get(group, {}).get(endpoint)
        if entry is None:
            return False
        # Legacy format: plain timestamp string means successful completion.
        if isinstance(entry, str):
            return True
        if isinstance(entry, dict):
            return bool(entry.get("status") == "ok")
        return False

    def get_endpoint_entry(self, group: str, endpoint: str):
        return self._data.get(group, {}).get(endpoint)

    def is_terminal_failure(self, group: str, endpoint: str) -> bool:
        entry = self.get_endpoint_entry(group, endpoint)
        if not isinstance(entry, dict):
            return False
        return bool(entry.get("status") == "failed" and entry.get("terminal", False))

    def should_skip_retry(self, group: str, endpoint: str, cooldown_hours: int | None = 24) -> bool:
        entry = self.get_endpoint_entry(group, endpoint)
        if not isinstance(entry, dict):
            return False
        if entry.get("status") != "failed":
            return False
        if bool(entry.get("terminal", False)):
            return True

        # Modo bloqueo indefinido: solo se libera con retry explícito.
        if cooldown_hours is None:
            return True

        updated_at = entry.get("updated_at")
        if not updated_at:
            return False
        try:
            ts = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        except Exception:
            return False

        age_seconds = (datetime.utcnow() - ts.replace(tzinfo=None)).total_seconds()
        return age_seconds < max(int(cooldown_hours), 1) * 3600

    def mark_done(self, group: str, endpoint: str) -> None:
        if group not in self._data:
            self._data[group] = {}
        self._data[group][endpoint] = {
            "status": "ok",
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.save()

    def mark_failed(
        self,
        group: str,
        endpoint: str,
        *,
        terminal: bool,
        reason: str,
        status_code: int | None = None,
    ) -> None:
        if group not in self._data:
            self._data[group] = {}
        payload = {
            "status": "failed",
            "terminal": bool(terminal),
            "reason": str(reason),
            "updated_at": datetime.utcnow().isoformat(),
        }
        if status_code is not None:
            payload["status_code"] = int(status_code)
        self._data[group][endpoint] = payload
        self.save()

    def clear(self, group: str | None = None, delete_file: bool = False) -> None:
        if group:
            self._data.pop(group, None)
        else:
            self._data = {}
        if delete_file and not group:
            try:
                if self.path.exists():
                    self.path.unlink()
            except Exception:
                # Fallback seguro: si no se puede borrar, al menos deja registro vacio.
                self.save()
            return
        self.save()
