# _omdb_payload + endpoints helpers
from __future__ import annotations

from typing import Any

from server.api.caching.file_cache import FileCache
from server.api.paths import OMDB_CACHE_PATH

TEXT_COLUMNS = ["poster_url", "trailer_url", "omdb_json"]


def load_payload(cache: FileCache) -> dict[str, Any]:
    data = cache.load_json(OMDB_CACHE_PATH)
    if not isinstance(data, dict):
        raise ValueError("omdb_cache.json no es un objeto JSON")
    return data
