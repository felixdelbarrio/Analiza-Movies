# _wiki_payload + helpers
from __future__ import annotations

from typing import Any

from server.api.caching.file_cache import FileCache
from server.api.paths import WIKI_CACHE_PATH


def load_payload(cache: FileCache) -> dict[str, Any]:
    data = cache.load_json(WIKI_CACHE_PATH)
    if not isinstance(data, dict):
        raise ValueError("wiki_cache.json no es un objeto JSON")
    return data
