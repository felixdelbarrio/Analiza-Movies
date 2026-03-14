# _omdb_payload + endpoints helpers
from __future__ import annotations

from typing import Any

from server.api.caching.file_cache import FileCache
from server.api.paths import get_omdb_cache_path

TEXT_COLUMNS = ["poster_url", "trailer_url", "omdb_json"]


def load_payload(cache: FileCache, profile_id: str | None = None) -> dict[str, Any]:
    data = cache.load_json(get_omdb_cache_path(profile_id))
    if not isinstance(data, dict):
        raise ValueError("omdb_cache.json no es un objeto JSON")
    return data
