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

Salida opcional a fichero (controlada por entorno)
--------------------------------------------------
Este módulo consume variables de entorno directamente:

- LOGGER_FILE_ENABLED: bool
- LOGGER_FILE_PATH: Path | str | None
- LOGGER_FILE_DIR / LOGGER_FILE_PREFIX / LOGGER_FILE_TIMESTAMP_FORMAT / LOGGER_FILE_INCLUDE_PID

Notas técnicas importantes
--------------------------
- Carga `.env` de forma best-effort para no depender de `config_base`.
- Inicialización idempotente.
- Best-effort: si no se puede abrir/crear el fichero, el pipeline sigue solo con consola.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Final, Mapping, TypedDict, cast

from dotenv import load_dotenv
from typing_extensions import TypeAlias, Unpack

load_dotenv(override=False)

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
_GENERATED_FILE_PATH: str | None = None


# ============================================================================
# UTILIDADES ENV / FLAGS
# ============================================================================


def _clean_env_raw(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text or None


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = _clean_env_raw(os.getenv(name))
    return default if raw is None else raw


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name, None)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_path(name: str) -> str | None:
    raw = _env_str(name, None)
    if raw is None:
        return None
    path = Path(raw)
    resolved = path if path.is_absolute() else (Path(__file__).resolve().parent / path)
    return str(resolved.resolve())


def _sanitize_filename_component(value: str) -> str:
    out_chars: list[str] = []
    for ch in value or "":
        if ch.isalnum() or ch in ("-", "_", ".", "@"):
            out_chars.append(ch)
        else:
            out_chars.append("_")
    cleaned = "".join(out_chars).strip("._-")
    return cleaned or "run"


def is_silent_mode() -> bool:
    return _env_bool("SILENT_MODE", False)


def is_debug_mode() -> bool:
    return _env_bool("DEBUG_MODE", False)


# ============================================================================
# RESOLUCIÓN DE LEVEL + EXTERNAL LOGGERS
# ============================================================================


def _resolve_level_from_config() -> int:
    """Determina el nivel del logging root."""
    lvl = _env_str("LOG_LEVEL", None)
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
        _env_bool("WIKI_DEBUG", False)
        or _env_bool("OMDB_DEBUG", False)
        or _env_bool("DEBUG", False)
        or _env_bool("DEBUG_MODE", False)
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


def _configure_external_loggers(*, level: int) -> None:
    """
    Baja el nivel de loggers externos ruidosos aunque el root esté en DEBUG,
    salvo que HTTP_DEBUG=True.
    """
    if _env_bool("HTTP_DEBUG", False):
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
    return _env_bool("LOGGER_FILE_ENABLED", False)


def _file_logging_path() -> str | None:
    """
    Path del fichero de log.

    Prioridad:
      0) ENV LOGGER_FILE_PATH
      1) Path generado localmente
      2) None
    """
    global _GENERATED_FILE_PATH

    env_path = _env_path("LOGGER_FILE_PATH")
    if env_path:
        return env_path

    if _GENERATED_FILE_PATH is not None:
        return _GENERATED_FILE_PATH

    if not _file_logging_enabled():
        return None

    base_dir = Path(__file__).resolve().parent
    dir_raw = _env_str("LOGGER_FILE_DIR", "logs") or "logs"
    target_dir = Path(dir_raw)
    if not target_dir.is_absolute():
        target_dir = base_dir / target_dir

    prefix = _sanitize_filename_component(
        _env_str("LOGGER_FILE_PREFIX", "run") or "run"
    )
    ts_format = _env_str("LOGGER_FILE_TIMESTAMP_FORMAT", "%Y-%m-%d_%H-%M-%S")
    timestamp = _sanitize_filename_component(
        datetime.now().strftime(ts_format or "%Y-%m-%d_%H-%M-%S")
    )
    include_pid = _env_bool("LOGGER_FILE_INCLUDE_PID", True)
    pid_suffix = f"_{os.getpid()}" if include_pid else ""

    resolved = (target_dir / f"{prefix}_{timestamp}{pid_suffix}.log").resolve()
    _GENERATED_FILE_PATH = str(resolved)
    os.environ["LOGGER_FILE_PATH"] = _GENERATED_FILE_PATH
    return _GENERATED_FILE_PATH


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
            return _env_int(
                "LOGGER_LOGS_MAX_SILENT_DEBUG", _DEFAULT_LOGS_MAX_SILENT_DEBUG
            )
        return _env_int("LOGGER_LOGS_MAX_SILENT", _DEFAULT_LOGS_MAX_SILENT)
    return _env_int("LOGGER_LOGS_MAX_NORMAL", _DEFAULT_LOGS_MAX_NORMAL)


def truncate_line(text: str, max_chars: int | None = None) -> str:
    limit = (
        int(max_chars)
        if isinstance(max_chars, int) and max_chars > 0
        else _env_int("LOGGER_LOG_LINE_MAX_CHARS", _DEFAULT_LOG_LINE_MAX_CHARS)
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
