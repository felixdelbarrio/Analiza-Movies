from __future__ import annotations

"""
backend/config.py (fachada compat)

Re-exporta TODO para compatibilidad:
- El resto del proyecto puede seguir haciendo: from backend.config import X
- Los clientes nuevos pueden importar solo su submódulo: backend.config_wiki, etc.
"""

# Base primero (dotenv + helpers + flags + paths)
from backend.config_base import *  # noqa: F403,F401

# Módulos por área
from backend.config_plex import *  # noqa: F403,F401
from backend.config_collection import *  # noqa: F403,F401
from backend.config_core import *  # noqa: F403,F401
from backend.config_omdb import *  # noqa: F403,F401
from backend.config_wiki import *  # noqa: F403,F401
from backend.config_dlna import *  # noqa: F403,F401
from backend.config_scoring import *  # noqa: F403,F401
from backend.config_reports import *  # noqa: F403,F401

# Debug dump (mismo comportamiento que antes, pero centralizado)
from backend.config_base import _log_config_debug, DEBUG_MODE, SILENT_MODE  # noqa: E402

def _dump_config_debug() -> None:
    # Flags principales
    _log_config_debug("DEBUG_MODE", DEBUG_MODE, debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("SILENT_MODE", SILENT_MODE, debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("LOG_LEVEL", globals().get("LOG_LEVEL"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("HTTP_DEBUG", globals().get("HTTP_DEBUG"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)

    # Paths / reports
    _log_config_debug("BASE_DIR", globals().get("BASE_DIR"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("DATA_DIR", globals().get("DATA_DIR"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("REPORTS_DIR", str(globals().get("REPORTS_DIR_PATH")), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)

    # Logger file
    _log_config_debug("LOGGER_FILE_ENABLED", globals().get("LOGGER_FILE_ENABLED"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("LOGGER_FILE_DIR", str(globals().get("LOGGER_FILE_DIR")), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("LOGGER_FILE_PATH", str(globals().get("LOGGER_FILE_PATH")) if globals().get("LOGGER_FILE_PATH") else None, debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)

    # Unos cuantos “hot knobs”
    _log_config_debug("PLEX_ANALYZE_WORKERS", globals().get("PLEX_ANALYZE_WORKERS"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("OMDB_HTTP_MAX_CONCURRENCY", globals().get("OMDB_HTTP_MAX_CONCURRENCY"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("WIKI_HTTP_MAX_CONCURRENCY", globals().get("WIKI_HTTP_MAX_CONCURRENCY"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("WIKI_CB_FAILURE_THRESHOLD", globals().get("WIKI_CB_FAILURE_THRESHOLD"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)
    _log_config_debug("WIKI_WDQS_CB_FAILURE_THRESHOLD", globals().get("WIKI_WDQS_CB_FAILURE_THRESHOLD"), debug_mode=DEBUG_MODE, silent_mode=SILENT_MODE)

_dump_config_debug()