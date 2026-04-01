"""Offline trace buffer for when platform is unreachable."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OfflineCache:
    """Local cache that buffers data when the platform is unreachable.

    Buffered items are flushed on the next successful connection.
    """

    def __init__(self, cache_dir: str = ".fastaiagent/cache/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        """Get cached data. Returns None if expired or missing."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            expires_at = data.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
                path.unlink(missing_ok=True)
                return None
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: dict, ttl_seconds: int = 3600) -> None:
        """Cache data with TTL."""
        path = self.cache_dir / f"{key}.json"
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        path.write_text(json.dumps({"value": value, "expires_at": expires_at}))

    def buffer_push(self, resource_type: str, data: dict) -> None:
        """Buffer a push for later retry when platform is reachable."""
        buffer_dir = self.cache_dir / "push_buffer"
        buffer_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        path = buffer_dir / f"{resource_type}_{ts}.json"
        path.write_text(json.dumps({"type": resource_type, "data": data}))

    def get_buffered_pushes(self) -> list[dict]:
        """Get all buffered pushes."""
        buffer_dir = self.cache_dir / "push_buffer"
        if not buffer_dir.exists():
            return []
        items = []
        for path in sorted(buffer_dir.glob("*.json")):
            try:
                items.append(json.loads(path.read_text()))
            except Exception:
                pass
        return items

    def clear_buffer(self) -> int:
        """Clear all buffered pushes. Returns count cleared."""
        buffer_dir = self.cache_dir / "push_buffer"
        if not buffer_dir.exists():
            return 0
        count = 0
        for path in buffer_dir.glob("*.json"):
            path.unlink(missing_ok=True)
            count += 1
        return count
