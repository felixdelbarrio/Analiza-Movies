from __future__ import annotations

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

Salida opcional a fichero (✅ integrado con la última config.py)
--------------------------------------------------------------
Este módulo NO decide nombres. Solo "consume" variables desde backend.config (si ya está importado):

- LOGGER_FILE_ENABLED: bool
- LOGGER_FILE_PATH: Path | str | None

La config (ya actualizada) genera un LOGGER_FILE_PATH único por ejecución
(timestamp + pid) o respeta un path explícito si se define.

Comportamiento cuando LOGGER_FILE_ENABLED=True y LOGGER_FILE_PATH no es None:
1) Todo lo que pase por `logging` (debug/info/warning/error) se duplica a fichero
   vía FileHandler en el root logger.
2) Todo lo que pase por `progress/progressf` también se apendea al mismo fichero
   (sin timestamp, para mantener el “heartbeat” limpio).

Notas técnicas importantes
--------------------------
- No importamos `backend.config` directamente (evitamos circular imports).
  Leemos `backend.config` desde `sys.modules` si ya está importado.
- Inicialización idempotente.
- Best-effort: si no se puede abrir/crear el fichero, el pipeline sigue solo con consola.
"""

import logging
import sys
import threading
from typing import Final

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

LOGGER_NAME: Final[str] = "movies_cleaner"

# Logger interno cacheado (solo logger real; sin Any).
_LOGGER: logging.Logger | None = None

# Flag de inicialización idempotente
_CONFIGURED: bool = False

# Tag para identificar nuestro handler de fichero (evita duplicados).
_FILE_HANDLER_TAG: Final[str] = "_movies_cleaner_file_handler"

# Lock para evitar mezclar líneas del progress() cuando hay hilos.
_PROGRESS_FILE_LOCK = threading.Lock()


# ============================================================================
# UTILIDADES CONFIG / FLAGS (sin importar backend.config directamente)
# ============================================================================

def _safe_get_cfg() -> object | None:
    """
    Devuelve el módulo backend.config si ya ha sido importado.

    Motivo:
    - Evita dependencias circulares (config importando logger y logger importando config).
    """
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

    Nota:
    - Si backend.config aún no está importado, devolvemos INFO (default seguro).
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
    """Aplica nivel al root y a sus handlers existentes (best-effort)."""
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

    Nota:
    - `level` se mantiene por compat; no lo usamos directamente para externos.
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
    """
    Feature flag: habilita duplicación del logging a fichero.

    Debe venir de backend.config:
      LOGGER_FILE_ENABLED: bool
    """
    return _cfg_bool("LOGGER_FILE_ENABLED", False)


def _file_logging_path() -> str | None:
    """
    Path del fichero de log.

    Esperamos que config exponga LOGGER_FILE_PATH como Path | str | None.
    - None / vacío => None
    """
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
    """Detecta si ya existe nuestro FileHandler (idempotencia)."""
    for h in root.handlers:
        try:
            if getattr(h, _FILE_HANDLER_TAG, False):
                return True
        except Exception:
            continue
    return False


def _ensure_file_handler(root: logging.Logger, *, level: int) -> None:
    """
    Añade un FileHandler al root logger si procede y si no existe ya.

    Reglas:
    - Solo si LOGGER_FILE_ENABLED=True y LOGGER_FILE_PATH está definido.
    - Idempotente (no duplica handlers).
    - Best-effort: nunca rompe el pipeline si falla crear el fichero/dir.

    Por qué root logger:
    - Cubre todo el logging del proceso (incluyendo librerías internas si no se silencian).
    """
    if not _file_logging_enabled():
        return

    path = _file_logging_path()
    if not path:
        return

    # Si ya existe, re-sincronizamos nivel y salimos.
    if _has_our_file_handler(root):
        for h in root.handlers:
            try:
                if getattr(h, _FILE_HANDLER_TAG, False):
                    h.setLevel(level)
            except Exception:
                pass
        return

    try:
        # Creamos directorio si hace falta (best-effort).
        import os  # local import: reduce coste y evita cargar si no se usa

        dir_name = os.path.dirname(path)
        if dir_name:
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                pass

        fh = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        # Marcamos handler como nuestro para detección idempotente.
        setattr(fh, _FILE_HANDLER_TAG, True)

        root.addHandler(fh)

        # Señal discreta solo en debug no-silent (sin usar logging para evitar recursion).
        try:
            if is_debug_mode() and not is_silent_mode():
                sys.stdout.write(f"[LOGGER] File logging enabled -> {path}\n")
                sys.stdout.flush()
        except Exception:
            pass

    except Exception:
        # Si no puede abrir el fichero, seguimos solo con consola.
        return


def _append_progress_to_file(message: str) -> None:
    """
    Duplica a fichero lo que sale por `progress()` cuando el file logging está habilitado.

    Diseño:
    - Best-effort: si falla, no rompe ni hace ruido.
    - Abrimos en modo append por llamada: robusto ante forks/errores/rotación externa.
    - Lock para no mezclar líneas concurrentes.
    """
    if not _file_logging_enabled():
        return

    path = _file_logging_path()
    if not path:
        return

    try:
        import os  # local import

        dir_name = os.path.dirname(path)
        if dir_name:
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                pass

        with _PROGRESS_FILE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                # Importante: progress es “señal limpia” (sin timestamps).
                # En el fichero convivirá con logs con timestamp (FileHandler).
                f.write(f"{message}\n")
    except Exception:
        return


# ============================================================================
# INICIALIZACIÓN DEL LOGGER
# ============================================================================

def _ensure_configured() -> logging.Logger:
    """
    Inicializa logging de forma idempotente y devuelve el logger principal.

    - Si ya se configuró, re-sincroniza niveles (por si backend.config cambió).
    - Configura handler de fichero si procede (consumiendo LOGGER_FILE_* de config).
    - Nunca lanza excepciones.
    """
    global _LOGGER, _CONFIGURED

    if _CONFIGURED and _LOGGER is not None:
        try:
            level = _resolve_level_from_config()
            _apply_root_level(level)
            _configure_external_loggers(level=level)

            root = logging.getLogger()
            _ensure_file_handler(root, level=level)
        except Exception:
            pass
        return _LOGGER

    level = _resolve_level_from_config()
    root = logging.getLogger()

    # Config base a consola (solo si no hay handlers previos).
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

    # File handler opcional (best-effort).
    try:
        _ensure_file_handler(root, level=level)
    except Exception:
        pass

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

    ✅ Si LOGGER_FILE_ENABLED=True y LOGGER_FILE_PATH existe, también se persiste a fichero.
    """
    try:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        _append_progress_to_file(message)
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
    """
    Trunca una línea para evitar payloads enormes (JSON/HTML/etc.).

    - max_chars explícito tiene prioridad.
    - Si no, usa LOGGER_LOG_LINE_MAX_CHARS desde config, con fallback seguro.
    """
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
        * SILENT_MODE=True  -> progress("[TAG][DEBUG] ...")   (✅ también a fichero si está habilitado)
        * SILENT_MODE=False -> info("[TAG][DEBUG] ...")       (✅ también a fichero si está habilitado)
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