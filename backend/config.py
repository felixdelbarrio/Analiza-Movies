"""
backend/config.py (fachada compat)

Re-exporta TODO para compatibilidad:
- El resto del proyecto puede seguir haciendo: from backend.config import X
- Los clientes nuevos pueden importar solo su submódulo: backend.config_wiki, etc.

Problema resuelto:
- Evitamos `from ... import *` porque puede “pisar” nombres `Final` (DATA_DIR, REPORTS_DIR_PATH, etc.)
  y Pylance lo interpreta como reasignación.
"""

from __future__ import annotations

from types import ModuleType

from backend import config_base as _config_base
from backend import config_collection as _config_collection
from backend import config_core as _config_core
from backend import config_dlna as _config_dlna
from backend import config_omdb as _config_omdb
from backend import config_plex as _config_plex
from backend import config_reports as _config_reports
from backend import config_scoring as _config_scoring
from backend import config_wiki as _config_wiki

# -----------------------------------------------------------------------------
# Re-export “seguro”: solo añadimos símbolos que NO existan ya en globals()
# así evitamos “pisar” Finals ya importados.
# Además, vamos acumulando los nombres exportados para construir __all__ estable.
# -----------------------------------------------------------------------------

_EXPORTED: list[str] = []


def _export_module(mod: ModuleType) -> None:
    public = getattr(mod, "__all__", None)
    if public is None:
        public = [n for n in dir(mod) if not n.startswith("_")]

    g = globals()
    for name in public:
        if name.startswith("_"):
            continue
        if name in g:
            # Ya existe: no lo sobrescribimos (evita reasignar Finals)
            continue
        g[name] = getattr(mod, name)
        _EXPORTED.append(name)


# Base primero
_export_module(_config_base)

# Luego el resto (sin pisar)
for _m in (
    _config_plex,
    _config_collection,
    _config_core,
    _config_omdb,
    _config_wiki,
    _config_dlna,
    _config_scoring,
    _config_reports,
):
    _export_module(_m)

# Construir __all__ (estable y solo con lo exportado)
__all__: list[str] = sorted(set(_EXPORTED))  # pyright: ignore[reportUnsupportedDunderAll]

# -----------------------------------------------------------------------------
# Debug dump (mismo comportamiento que antes, pero centralizado)
# -----------------------------------------------------------------------------
from backend.config_base import _log_config_debug, DEBUG_MODE, SILENT_MODE  # noqa: E402


def _dump_config_debug() -> None:
    _log_config_debug(
        "DEBUG_MODE", DEBUG_MODE, debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE
    )
    _log_config_debug(
        "SILENT_MODE", SILENT_MODE, debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE
    )
    _log_config_debug(
        "LOG_LEVEL",
        globals().get("LOG_LEVEL"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "HTTP_DEBUG",
        globals().get("HTTP_DEBUG"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )

    _log_config_debug(
        "BASE_DIR",
        globals().get("BASE_DIR"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "DATA_DIR",
        globals().get("DATA_DIR"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "REPORTS_DIR",
        str(globals().get("REPORTS_DIR_PATH")),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )

    _log_config_debug(
        "LOGGER_FILE_ENABLED",
        globals().get("LOGGER_FILE_ENABLED"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "LOGGER_FILE_DIR",
        str(globals().get("LOGGER_FILE_DIR")),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "LOGGER_FILE_PATH",
        (
            str(globals().get("LOGGER_FILE_PATH"))
            if globals().get("LOGGER_FILE_PATH")
            else None
        ),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )

    _log_config_debug(
        "PLEX_ANALYZE_WORKERS",
        globals().get("PLEX_ANALYZE_WORKERS"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "OMDB_HTTP_MAX_CONCURRENCY",
        globals().get("OMDB_HTTP_MAX_CONCURRENCY"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "WIKI_HTTP_MAX_CONCURRENCY",
        globals().get("WIKI_HTTP_MAX_CONCURRENCY"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "WIKI_CB_FAILURE_THRESHOLD",
        globals().get("WIKI_CB_FAILURE_THRESHOLD"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )
    _log_config_debug(
        "WIKI_WDQS_CB_FAILURE_THRESHOLD",
        globals().get("WIKI_WDQS_CB_FAILURE_THRESHOLD"),
        debug_mode=DEBUG_MODE,
        silent_mode=SILENT_MODE,
    )


_dump_config_debug()
