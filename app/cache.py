"""Bounded TTL cache for chat responses."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU with per-entry expiry.

    The bound matters more than the speed here: an unbounded dict keyed on user
    questions is a memory leak that only shows up in production.
    """

    def __init__(self, max_entries: int, ttl_seconds: float):
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._entries[key] = (time.monotonic() + self._ttl, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._entries), "hits": self.hits, "misses": self.misses}
