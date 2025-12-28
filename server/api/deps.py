from __future__ import annotations

from server.api.caching.file_cache import FileCache
from server.api.settings import Settings

_SETTINGS = Settings.from_env()
_FILE_CACHE = FileCache(_SETTINGS)


def get_settings() -> Settings:
    return _SETTINGS


def get_file_cache() -> FileCache:
    return _FILE_CACHE