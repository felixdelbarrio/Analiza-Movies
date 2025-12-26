from __future__ import annotations

from typing import Final

from backend.config_base import (
    _cap_int,
    _get_env_bool,
    _get_env_enum_str,
    _get_env_int,
)

# ============================================================
# COLLECTION (orquestación por item): caches in-memory / trazas
# ============================================================

COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS: int = _cap_int(
    "COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS",
    _get_env_int("COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS", 1200),
    min_v=0,
    max_v=500_000,
)

COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS: int = _cap_int(
    "COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS",
    _get_env_int("COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS", 2400),
    min_v=0,
    max_v=1_000_000,
)

COLLECTION_TRACE_LINE_MAX_CHARS: int = _cap_int(
    "COLLECTION_TRACE_LINE_MAX_CHARS",
    _get_env_int("COLLECTION_TRACE_LINE_MAX_CHARS", 220),
    min_v=60,
    max_v=5000,
)

COLLECTION_ENABLE_LAZY_WIKI: bool = _get_env_bool("COLLECTION_ENABLE_LAZY_WIKI", True)

COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE: bool = _get_env_bool(
    "COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE",
    True,
)

_COLLECTION_OMDB_JSON_MODE_ALLOWED: Final[set[str]] = {"auto", "never", "always"}
COLLECTION_OMDB_JSON_MODE: str = _get_env_enum_str(
    "COLLECTION_OMDB_JSON_MODE",
    default="auto",
    allowed=_COLLECTION_OMDB_JSON_MODE_ALLOWED,
    normalize=True,
)

COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK: bool = _get_env_bool(
    "COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK",
    False,
)

COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE: bool = _get_env_bool(
    "COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE",
    True,
)

COLLECTION_TRACE_ALSO_DEBUG_CTX: bool = _get_env_bool("COLLECTION_TRACE_ALSO_DEBUG_CTX", True)