# _wiki_payload + helpers
from __future__ import annotations

from typing import Any

from server.api.caching.file_cache import FileCache
from server.api.paths import get_wiki_cache_path


def load_payload(cache: FileCache, profile_id: str | None = None) -> dict[str, Any]:
    data = cache.load_json(get_wiki_cache_path(profile_id))
    if not isinstance(data, dict):
        raise ValueError("wiki_cache.json no es un objeto JSON")
    return data
