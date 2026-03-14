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

# Si True, el lazy-wiki se aplica a todo el catálogo (no solo DELETE/MAYBE/misidentified).
COLLECTION_LAZY_WIKI_FOR_ALL: bool = _get_env_bool(
    "COLLECTION_LAZY_WIKI_FOR_ALL",
    False,
)

COLLECTION_TRACE_ALSO_DEBUG_CTX: bool = _get_env_bool(
    "COLLECTION_TRACE_ALSO_DEBUG_CTX", True
)

# ============================================================
# DLNA-only: política de Wiki vs. __wiki embebido en OMDb cache
# (NO afecta Plex/otros porque sólo se aplicará si source=="dlna")
# ============================================================

# Cómo tratar __wiki embebido en OMDb cache cuando source=="dlna":
# - "use": usar __wiki si existe (comportamiento legacy)
# - "ignore": ignorarlo siempre y forzar path de Wiki real (mejor para bibliotecas no-en)
# - "use_if_fresh": usar si es "fresco" (por edad) y si no, refrescar
_DLNA_WIKI_EMBEDDED_MODE_ALLOWED: Final[set[str]] = {"use", "ignore", "use_if_fresh"}
DLNA_WIKI_EMBEDDED_MODE: str = _get_env_enum_str(
    "DLNA_WIKI_EMBEDDED_MODE",
    default="ignore",
    allowed=_DLNA_WIKI_EMBEDDED_MODE_ALLOWED,
    normalize=True,
)

# Ventana de frescura (solo si DLNA_WIKI_EMBEDDED_MODE == "use_if_fresh")
# Nota: requiere timestamp en __wiki (persistido en collection_analysis.py).
DLNA_WIKI_EMBEDDED_MAX_AGE_SECONDS: int = _cap_int(
    "DLNA_WIKI_EMBEDDED_MAX_AGE_SECONDS",
    _get_env_int("DLNA_WIKI_EMBEDDED_MAX_AGE_SECONDS", 14 * 24 * 60 * 60),
    min_v=0,
    max_v=365 * 24 * 60 * 60,
)

# Si True, en DLNA se permite intentar Wiki antes de OMDb cuando el título no está claro / no-inglés.
# (Esto NO toca Plex/otros; se aplicará solo en el flujo DLNA de collection_analysis).
DLNA_WIKI_PRE_OMDB_RESOLUTION: bool = _get_env_bool(
    "DLNA_WIKI_PRE_OMDB_RESOLUTION", True
)

# Si True, en DLNA se fuerza a que el lazy-wiki se ejecute incluso si hay __wiki embebido,
# respetando DLNA_WIKI_EMBEDDED_MODE.
DLNA_WIKI_FORCE_LAZY_EVEN_WITH_EMBEDDED: bool = _get_env_bool(
    "DLNA_WIKI_FORCE_LAZY_EVEN_WITH_EMBEDDED",
    True,
)
