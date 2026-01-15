"""
backend/logger.py

Logger central del proyecto (fachada sobre `logging`).

API estable
-----------
- debug / info / warning / error
- progress / progressf (siempre visible, sin timestamps)
- debug_ctx(tag, msg) (debug contextual alineado con SILENT/DEBUG)
- append_bounded_log(logs, line, ...) (logs por item acotados)

Política
--------
- SILENT_MODE=True: suprime debug/info/warning (salvo always=True). `error()` siempre emite.
- DEBUG_MODE=True: permite trazas útiles; en SILENT+DEBUG se emiten por `progress`.
- El logging nunca debe romper el pipeline.

Salida opcional a fichero (integrado con config.py)
---------------------------------------------------
Este módulo NO decide nombres. Solo "consume" variables desde backend.config (si ya está importado):

- LOGGER_FILE_ENABLED: bool
- LOGGER_FILE_PATH: Path | str | None

Notas técnicas importantes
--------------------------
- No importamos `backend.config` directamente (evitamos circular imports).
  Leemos `backend.config` desde `sys.modules` si ya está importado.
- Inicialización idempotente.
- Best-effort: si no se puede abrir/crear el fichero, el pipeline sigue solo con consola.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from types import ModuleType, TracebackType
from typing import Final, Mapping, TypedDict, cast

from typing_extensions import TypeAlias, Unpack

# ============================================================================
# TIPOS: kwargs seguros para logging
# ============================================================================

_ExcInfoTuple: TypeAlias = tuple[
    type[BaseException], BaseException, TracebackType | None
]
ExcInfo: TypeAlias = bool | _ExcInfoTuple | BaseException | None


class LogKwargs(TypedDict, total=False):
    """Subconjunto útil de kwargs soportados por logging.Logger.*."""

    exc_info: ExcInfo
    stack_info: bool
    stacklevel: int
    extra: Mapping[str, object] | None


def _filter_log_kwargs(kwargs: Mapping[str, object]) -> LogKwargs:
    """Best-effort: filtra kwargs no tipados a un conjunto seguro para logging."""
    out: LogKwargs = {}

    v = kwargs.get("exc_info")
    if "exc_info" in kwargs and (
        v is None or isinstance(v, (bool, BaseException, tuple))
    ):
        out["exc_info"] = cast(ExcInfo, v)

    v = kwargs.get("stack_info")
    if "stack_info" in kwargs and isinstance(v, bool):
        out["stack_info"] = v

    v = kwargs.get("stacklevel")
    if "stacklevel" in kwargs and isinstance(v, int):
        out["stacklevel"] = v

    v = kwargs.get("extra")
    if "extra" in kwargs and (v is None or isinstance(v, Mapping)):
        out["extra"] = cast(Mapping[str, object] | None, v)

    return out


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

LOGGER_NAME: Final[str] = "movies_cleaner"

# Logger NO opcional (evita “narrowing” raro en algunos typecheckers)
_LOGGER: Final[logging.Logger] = logging.getLogger(LOGGER_NAME)
_CONFIGURED: bool = False

_FILE_HANDLER_TAG: Final[str] = "_movies_cleaner_file_handler"
_PROGRESS_FILE_LOCK: Final[threading.Lock] = threading.Lock()


# ============================================================================
# UTILIDADES CONFIG / FLAGS (sin importar backend.config directamente)
# ============================================================================


def _safe_get_cfg() -> ModuleType | None:
    """Devuelve el módulo backend.config si ya ha sido importado (evita circular imports)."""
    mod = sys.modules.get("backend.config")
    return mod if isinstance(mod, ModuleType) else None


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
    """Determina el nivel del logging root."""
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
    """Aplica nivel al root y a sus handlers existentes (best-effort)."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        try:
            handler.setLevel(level)
        except Exception:
            pass


def _should_enable_http_debug(cfg: ModuleType | None) -> bool:
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
# FILE LOGGING (opcional, controlado por config)
# ============================================================================


def _file_logging_enabled() -> bool:
    return _cfg_bool("LOGGER_FILE_ENABLED", False)


def _file_logging_path() -> str | None:
    """
    Path del fichero de log.

    Prioridad:
      0) ENV LOGGER_FILE_PATH
      1) backend.config.LOGGER_FILE_PATH
      2) None
    """
    try:
        env_p = (os.getenv("LOGGER_FILE_PATH") or "").strip()
        if env_p:
            return env_p
    except Exception:
        pass

    cfg = _safe_get_cfg()
    if cfg is None:
        return None

    try:
        p = getattr(cfg, "LOGGER_FILE_PATH", None)
        if p is None:
            return None
        s = str(p).strip()
        return s or None
    except Exception:
        return None


def _has_our_file_handler(root: logging.Logger) -> bool:
    for h in root.handlers:
        try:
            if bool(getattr(h, _FILE_HANDLER_TAG, False)):
                return True
        except Exception:
            pass
    return False


def _ensure_file_handler(root: logging.Logger, *, level: int) -> None:
    if not _file_logging_enabled():
        return

    path = _file_logging_path()
    if not path:
        return

    if _has_our_file_handler(root):
        for h in root.handlers:
            try:
                if bool(getattr(h, _FILE_HANDLER_TAG, False)):
                    h.setLevel(level)
            except Exception:
                pass
        return

    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fh = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        setattr(fh, _FILE_HANDLER_TAG, True)
        root.addHandler(fh)

        if is_debug_mode() and not is_silent_mode():
            try:
                sys.stdout.write(f"[LOGGER] File logging enabled -> {path}\n")
                sys.stdout.flush()
            except Exception:
                pass
    except Exception:
        pass


def _append_progress_to_file(message: str) -> None:
    if not _file_logging_enabled():
        return

    path = _file_logging_path()
    if not path:
        return

    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with _PROGRESS_FILE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
    except Exception:
        pass


# ============================================================================
# INICIALIZACIÓN DEL LOGGER
# ============================================================================


def _ensure_configured() -> logging.Logger:
    """Inicializa logging de forma idempotente y devuelve el logger principal."""
    global _CONFIGURED

    level = _resolve_level_from_config()
    root = logging.getLogger()

    if not _CONFIGURED:
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
        _CONFIGURED = True
    else:
        try:
            _apply_root_level(level)
        except Exception:
            pass

    _configure_external_loggers(level=level)
    _ensure_file_handler(root, level=level)
    return _LOGGER


def get_logger() -> logging.Logger:
    return _ensure_configured()


# ============================================================================
# CONTROL DE SILENT MODE (para logs con niveles)
# ============================================================================


def _should_log(*, always: bool = False) -> bool:
    return bool(always) or (not is_silent_mode())


# ============================================================================
# PROGRESO / HEARTBEAT (NO logging)
# ============================================================================


def progress(message: str) -> None:
    try:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
    except Exception:
        pass
    _append_progress_to_file(message)


def progressf(fmt: str, *args: object) -> None:
    try:
        msg = fmt % args if args else fmt
    except Exception:
        msg = fmt
    progress(msg)


# ============================================================================
# API PÚBLICA DE LOGGING
# ============================================================================


def debug(
    msg: str, *args: object, always: bool = False, **kwargs: Unpack[LogKwargs]
) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    try:
        log.debug(msg, *args, **kwargs)
    except Exception:
        pass


def info(
    msg: str, *args: object, always: bool = False, **kwargs: Unpack[LogKwargs]
) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    try:
        log.info(msg, *args, **kwargs)
    except Exception:
        pass


def warning(
    msg: str, *args: object, always: bool = False, **kwargs: Unpack[LogKwargs]
) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    try:
        log.warning(msg, *args, **kwargs)
    except Exception:
        pass


def error(
    msg: str, *args: object, always: bool = False, **kwargs: Unpack[LogKwargs]
) -> None:
    """ERROR siempre se emite (ignora SILENT_MODE)."""
    log = _ensure_configured()
    try:
        log.error(msg, *args, **kwargs)
    except Exception:
        try:
            sys.stdout.write(f"{msg}\n")
            sys.stdout.flush()
        except Exception:
            pass


# ============================================================================
# COMPAT: llamadas legacy con **kwargs: object
# ============================================================================


def debug_any(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    try:
        safe = _filter_log_kwargs(cast(Mapping[str, object], kwargs))
        log.debug(msg, *args, **safe)
    except Exception:
        pass


def info_any(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    try:
        safe = _filter_log_kwargs(cast(Mapping[str, object], kwargs))
        log.info(msg, *args, **safe)
    except Exception:
        pass


def warning_any(
    msg: str, *args: object, always: bool = False, **kwargs: object
) -> None:
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    try:
        safe = _filter_log_kwargs(cast(Mapping[str, object], kwargs))
        log.warning(msg, *args, **safe)
    except Exception:
        pass


def error_any(msg: str, *args: object, always: bool = False, **kwargs: object) -> None:
    log = _ensure_configured()
    try:
        safe = _filter_log_kwargs(cast(Mapping[str, object], kwargs))
        log.error(msg, *args, **safe)
    except Exception:
        try:
            sys.stdout.write(f"{msg}\n")
            sys.stdout.flush()
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
    if is_silent_mode():
        if is_debug_mode():
            return _cfg_int(
                "LOGGER_LOGS_MAX_SILENT_DEBUG", _DEFAULT_LOGS_MAX_SILENT_DEBUG
            )
        return _cfg_int("LOGGER_LOGS_MAX_SILENT", _DEFAULT_LOGS_MAX_SILENT)
    return _cfg_int("LOGGER_LOGS_MAX_NORMAL", _DEFAULT_LOGS_MAX_NORMAL)


def truncate_line(text: str, max_chars: int | None = None) -> str:
    limit = (
        int(max_chars)
        if isinstance(max_chars, int) and max_chars > 0
        else _cfg_int("LOGGER_LOG_LINE_MAX_CHARS", _DEFAULT_LOG_LINE_MAX_CHARS)
    )
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)] + " …(truncated)"


def append_bounded_log(
    logs: object,  # ✅ importante: así el isinstance() NO es “unreachable”
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

    # Nos quedamos con una lista mutable de strings (best-effort)
    logs_list = cast(list[str], logs)

    prefix = f"[{tag}] " if isinstance(tag, str) and tag.strip() else ""
    msg = prefix + truncate_line(str(line))

    limit = logs_limit()

    if force:
        logs_list.append(msg)
        return

    if limit <= 0:
        return

    if len(logs_list) >= limit:
        if not logs_list or logs_list[-1] != _LOGS_TRUNCATED_SENTINEL:
            logs_list.append(_LOGS_TRUNCATED_SENTINEL)
        return

    logs_list.append(msg)


def debug_ctx(tag: str, msg: object) -> None:
    if not is_debug_mode():
        return

    t = (tag or "DEBUG").strip().upper()
    line = f"[{t}][DEBUG] {str(msg)}"

    try:
        if is_silent_mode():
            progress(line)
        else:
            info(line)
    except Exception:
        try:
            sys.stdout.write(f"{line}\n")
            sys.stdout.flush()
        except Exception:
            pass


__all__ = [
    "get_logger",
    "debug",
    "info",
    "warning",
    "error",
    "debug_any",
    "info_any",
    "warning_any",
    "error_any",
    "progress",
    "progressf",
    "debug_ctx",
    "append_bounded_log",
    "logs_limit",
    "truncate_line",
    "is_silent_mode",
    "is_debug_mode",
]
