from __future__ import annotations

"""
backend/logger.py

Logger central del proyecto (fachada sobre `logging`).

API estable:
- debug / info / warning / error
- progress / progressf (siempre visible, sin timestamps)
- debug_ctx(tag, msg) (debug contextual alineado con SILENT/DEBUG)
- append_bounded_log(logs, line, ...) (logs por item acotados)

Política:
- SILENT_MODE=True: suprime debug/info/warning (salvo always=True). `error()` siempre emite.
- DEBUG_MODE=True: permite trazas útiles; en SILENT+DEBUG se emiten por `progress`.
- El logging nunca debe romper el pipeline.

Nota:
- No importamos `backend.config` directamente para evitar dependencias circulares.
  Leemos `backend.config` desde `sys.modules` si ya está importado.
"""

import logging
import sys
from typing import Final

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

LOGGER_NAME: Final[str] = "movies_cleaner"

# Logger interno cacheado (solo logger real; sin Any).
_LOGGER: logging.Logger | None = None

# Flag de inicialización idempotente
_CONFIGURED: bool = False


# ============================================================================
# UTILIDADES CONFIG / FLAGS (sin importar backend.config directamente)
# ============================================================================

def _safe_get_cfg() -> object | None:
    """Devuelve backend.config si ya ha sido importado; si no, None."""
    return sys.modules.get("backend.config")


def _cfg_bool(name: str, default: bool = False) -> bool:
    cfg = _safe_get_cfg()
    if cfg is None:
        return default
    try:
        return bool(getattr(cfg, name, default))
    except Exception:
        return default


def _cfg_int(name: str, default: int) -> int:
    cfg = _safe_get_cfg()
    if cfg is None:
        return default
    try:
        return int(getattr(cfg, name, default))
    except Exception:
        return default


def _cfg_str(name: str, default: str | None = None) -> str | None:
    cfg = _safe_get_cfg()
    if cfg is None:
        return default
    try:
        v = getattr(cfg, name, default)
        if v is None:
            return None
        s = str(v).strip()
        return s or default
    except Exception:
        return default


def is_silent_mode() -> bool:
    """SILENT_MODE global (si backend.config está cargado)."""
    return _cfg_bool("SILENT_MODE", False)


def is_debug_mode() -> bool:
    """DEBUG_MODE global (si backend.config está cargado)."""
    return _cfg_bool("DEBUG_MODE", False)


# ============================================================================
# RESOLUCIÓN DE LEVEL + EXTERNAL LOGGERS
# ============================================================================

def _resolve_level_from_config() -> int:
    """
    Determina el nivel del logging root.

    Prioridad:
      1) LOG_LEVEL explícito
      2) Flags de debug heredados
      3) INFO
    """
    if _safe_get_cfg() is None:
        return logging.INFO

    lvl = _cfg_str("LOG_LEVEL", None)
    if isinstance(lvl, str) and lvl.strip():
        name = lvl.strip().upper()
        mapped = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.CRITICAL,
        }.get(name)
        if mapped is not None:
            return mapped

    if (
        _cfg_bool("WIKI_DEBUG", False)
        or _cfg_bool("OMDB_DEBUG", False)
        or _cfg_bool("DEBUG", False)
        or _cfg_bool("DEBUG_MODE", False)
    ):
        return logging.DEBUG

    return logging.INFO


def _apply_root_level(level: int) -> None:
    """Aplica nivel al root y a sus handlers (si existen)."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        try:
            handler.setLevel(level)
        except Exception:
            pass


def _should_enable_http_debug(cfg: object | None) -> bool:
    """Si HTTP_DEBUG=True, no silenciamos urllib3/requests/plexapi."""
    if cfg is None:
        return False
    try:
        return bool(getattr(cfg, "HTTP_DEBUG", False))
    except Exception:
        return False


def _configure_external_loggers(*, level: int) -> None:
    """
    Baja el nivel de loggers externos ruidosos aunque el root esté en DEBUG,
    salvo que HTTP_DEBUG=True.
    """
    cfg = _safe_get_cfg()
    if _should_enable_http_debug(cfg):
        return

    noisy = (
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "requests.packages.urllib3",
        "plexapi",
    )

    for name in noisy:
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass


# ============================================================================
# INICIALIZACIÓN DEL LOGGER
# ============================================================================

def _ensure_configured() -> logging.Logger:
    """
    Inicializa logging de forma idempotente y devuelve el logger principal.

    - Si ya se configuró, re-sincroniza niveles (por si backend.config cambió).
    - Nunca lanza excepciones.
    """
    global _LOGGER, _CONFIGURED

    if _CONFIGURED and _LOGGER is not None:
        try:
            level = _resolve_level_from_config()
            _apply_root_level(level)
            _configure_external_loggers(level=level)
        except Exception:
            pass
        return _LOGGER

    level = _resolve_level_from_config()
    root = logging.getLogger()

    try:
        if not root.handlers:
            logging.basicConfig(
                level=level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
        else:
            _apply_root_level(level)
    except Exception:
        pass

    _configure_external_loggers(level=level)

    _LOGGER = logging.getLogger(LOGGER_NAME)
    _CONFIGURED = True
    return _LOGGER


def get_logger() -> logging.Logger:
    """Devuelve el logger principal, asegurando inicialización."""
    return _ensure_configured()


# ============================================================================
# CONTROL DE SILENT MODE (para logs con niveles)
# ============================================================================

def _should_log(*, always: bool = False) -> bool:
    """
    Decide si un mensaje debe emitirse (para debug/info/warning).

    - always=True -> siempre emite
    - SILENT_MODE=True -> suprime
    """
    if always:
        return True
    return not is_silent_mode()


# ============================================================================
# PROGRESO / HEARTBEAT (NO logging)
# ============================================================================

def progress(message: str) -> None:
    """
    Emite una línea siempre visible (ignora SILENT_MODE).

    No usa logging para evitar timestamps/levels y dar una “señal” limpia.
    """
    try:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
    except Exception:
        pass


def progressf(fmt: str, *args: object) -> None:
    """Formato estilo printf para progress."""
    try:
        msg = fmt % args if args else fmt
    except Exception:
        msg = fmt
    progress(msg)


# ============================================================================
# API PÚBLICA DE LOGGING
# ============================================================================

def debug(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    try:
        log.debug(msg, *args, **kwargs)
    except Exception:
        pass


def info(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    try:
        log.info(msg, *args, **kwargs)
    except Exception:
        pass


def warning(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    try:
        log.warning(msg, *args, **kwargs)
    except Exception:
        pass


def error(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    """
    ERROR siempre se emite (ignora SILENT_MODE). `always` se acepta por compat.
    """
    log = _ensure_configured()
    kwargs.pop("always", None)
    try:
        log.error(msg, *args, **kwargs)
    except Exception:
        try:
            print(msg)
        except Exception:
            pass


# ============================================================================
# DEBUG CONTEXTUAL (tag) + BOUNDED LOG BUFFER
# ============================================================================

_DEFAULT_LOGS_MAX_SILENT: Final[int] = 0
_DEFAULT_LOGS_MAX_SILENT_DEBUG: Final[int] = 12
_DEFAULT_LOGS_MAX_NORMAL: Final[int] = 50
_DEFAULT_LOG_LINE_MAX_CHARS: Final[int] = 500

_LOGS_TRUNCATED_SENTINEL: Final[str] = "[LOGS_TRUNCATED]"


def logs_limit() -> int:
    """
    Límite de logs acumulables por item, según modos.

    Ajustable desde config:
      - LOGGER_LOGS_MAX_SILENT
      - LOGGER_LOGS_MAX_SILENT_DEBUG
      - LOGGER_LOGS_MAX_NORMAL
    """
    if is_silent_mode():
        if is_debug_mode():
            return _cfg_int("LOGGER_LOGS_MAX_SILENT_DEBUG", _DEFAULT_LOGS_MAX_SILENT_DEBUG)
        return _cfg_int("LOGGER_LOGS_MAX_SILENT", _DEFAULT_LOGS_MAX_SILENT)
    return _cfg_int("LOGGER_LOGS_MAX_NORMAL", _DEFAULT_LOGS_MAX_NORMAL)


def truncate_line(text: str, max_chars: int | None = None) -> str:
    """Trunca una línea para evitar payloads enormes (JSON/HTML/etc.)."""
    limit = (
        int(max_chars)
        if isinstance(max_chars, int) and max_chars > 0
        else _cfg_int("LOGGER_LOG_LINE_MAX_CHARS", _DEFAULT_LOG_LINE_MAX_CHARS)
    )
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)] + " …(truncated)"


def append_bounded_log(
    logs: list[str],
    line: object,
    *,
    force: bool = False,
    tag: str | None = None,
) -> None:
    """
    Añade un log a `logs` respetando límites.

    - force=True: ignora el límite (pero trunca).
    - SILENT + !DEBUG -> normalmente no acumula (limit=0).
    """
    if not isinstance(logs, list):
        return

    prefix = f"[{tag}] " if isinstance(tag, str) and tag.strip() else ""
    msg = prefix + truncate_line(str(line))

    limit = logs_limit()

    if force:
        logs.append(msg)
        return

    if limit <= 0:
        return

    if len(logs) >= limit:
        if not logs or logs[-1] != _LOGS_TRUNCATED_SENTINEL:
            logs.append(_LOGS_TRUNCATED_SENTINEL)
        return

    logs.append(msg)


def debug_ctx(tag: str, msg: object) -> None:
    """
    Debug contextual con tag.

    - DEBUG_MODE=False -> no-op
    - DEBUG_MODE=True:
        * SILENT_MODE=True  -> progress("[TAG][DEBUG] ...")
        * SILENT_MODE=False -> info("[TAG][DEBUG] ...")
    """
    if not is_debug_mode():
        return

    t = (tag or "DEBUG").strip().upper()
    text = str(msg)

    try:
        if is_silent_mode():
            progress(f"[{t}][DEBUG] {text}")
        else:
            info(f"[{t}][DEBUG] {text}")
    except Exception:
        if not is_silent_mode():
            try:
                print(f"[{t}][DEBUG] {text}")
            except Exception:
                pass