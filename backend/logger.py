from __future__ import annotations

import logging
import sys
from typing import Any, Final

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Nombre del logger principal de la aplicación.
# ⚠️ IMPORTANTE: los tests lo referencian explícitamente.
LOGGER_NAME: Final[str] = "plex_movies_cleaner"

# Logger interno cacheado.
# Puede ser:
#   - logging.Logger (ejecución real)
#   - FakeLogger (inyectado por tests)
_LOGGER: Any = None

# Flag para asegurar inicialización idempotente del logger
_CONFIGURED: bool = False


# ============================================================================
# UTILIDADES DE CONFIGURACIÓN
# ============================================================================

def _safe_get_cfg() -> Any | None:
    """
    Devuelve el módulo backend.config si ya ha sido importado.

    No lo importamos directamente para:
    - evitar dependencias circulares
    - permitir ejecución parcial (tests, herramientas, etc.)
    """
    return sys.modules.get("backend.config")


def _resolve_level_from_config() -> int:
    """
    Determina el nivel de logging raíz (root) en base a backend.config.

    Orden de prioridad:
      1) LOG_LEVEL explícito (string)
      2) Flags de debug (WIKI_DEBUG, OMDB_DEBUG, DEBUG, DEBUG_MODE)
      3) Fallback seguro: INFO
    """
    cfg = _safe_get_cfg()
    if cfg is None:
        return logging.INFO

    # ------------------------------------------------------------------
    # 1) LOG_LEVEL explícito (string)
    # ------------------------------------------------------------------
    try:
        lvl = getattr(cfg, "LOG_LEVEL", None)
    except Exception:
        lvl = None

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

    # ------------------------------------------------------------------
    # 2) Flags de debug “heredados”
    # ------------------------------------------------------------------
    def _flag(name: str) -> bool:
        try:
            return bool(getattr(cfg, name, False))
        except Exception:
            return False

    if (
        _flag("WIKI_DEBUG")
        or _flag("OMDB_DEBUG")
        or _flag("DEBUG")
        or _flag("DEBUG_MODE")
    ):
        return logging.DEBUG

    return logging.INFO


def _apply_root_level(level: int) -> None:
    """
    Aplica el nivel al logger root y a todos sus handlers.

    Esto es necesario porque:
    - algunos entornos ya crean handlers antes de que lleguemos aquí
    - cambiar solo el root no siempre actualiza los handlers existentes
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        try:
            handler.setLevel(level)
        except Exception:
            pass


def _should_enable_http_debug(cfg: Any | None) -> bool:
    """
    Decide si se permite el ruido HTTP de librerías externas.

    Si backend.config define:
        HTTP_DEBUG = True
    entonces NO se silencian urllib3 / requests / plexapi.

    Por defecto: False (evitamos spam).
    """
    if cfg is None:
        return False
    try:
        return bool(getattr(cfg, "HTTP_DEBUG", False))
    except Exception:
        return False


def _configure_external_loggers(*, level: int) -> None:
    """
    Ajusta el nivel de logging de dependencias ruidosas.

    Objetivo:
    - Aunque el root esté en DEBUG,
      evitar que urllib3 / requests / plexapi inunden la consola.

    Excepción:
    - Si HTTP_DEBUG=True → no tocamos nada.
    """
    cfg = _safe_get_cfg()
    if _should_enable_http_debug(cfg):
        return

    noisy_loggers = (
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "requests.packages.urllib3",
        "plexapi",
    )

    for name in noisy_loggers:
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass


# ============================================================================
# INICIALIZACIÓN DEL LOGGER
# ============================================================================

def _ensure_configured() -> Any:
    """
    Inicializa el sistema de logging una sola vez y devuelve el logger principal.

    Comportamiento clave:
    - Idempotente: puede llamarse muchas veces sin efectos secundarios.
    - Respeta loggers inyectados por tests.
    - Reaplica niveles si backend.config cambia dinámicamente.
    """
    global _LOGGER, _CONFIGURED

    # Ya inicializado → solo re-sincronizamos niveles
    if _CONFIGURED and _LOGGER is not None:
        try:
            level = _resolve_level_from_config()
            _apply_root_level(level)
            _configure_external_loggers(level=level)
        except Exception:
            pass
        return _LOGGER

    # Primera inicialización real
    level = _resolve_level_from_config()
    root = logging.getLogger()

    if not root.handlers:
        # Entorno limpio → configuramos logging básico
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        # Entorno ya configurado (IDE, tests, etc.)
        _apply_root_level(level)

    _configure_external_loggers(level=level)

    _LOGGER = logging.getLogger(LOGGER_NAME)
    _CONFIGURED = True
    return _LOGGER


# ============================================================================
# CONTROL DE SILENT MODE
# ============================================================================

def _should_log(*, always: bool = False) -> bool:
    """
    Decide si un mensaje de logging debe emitirse.

    Reglas:
      - always=True → siempre se emite
      - SILENT_MODE=True → se suprimen debug/info/warning
      - error() ignora este mecanismo (ver más abajo)
    """
    if always:
        return True

    cfg = sys.modules.get("backend.config")
    if cfg is None:
        return True

    try:
        silent = getattr(cfg, "SILENT_MODE", False)
    except Exception:
        return True

    return not bool(silent)


def get_logger() -> Any:
    """
    Devuelve el logger principal de la aplicación.

    Siempre asegura inicialización previa.
    """
    return _ensure_configured()


# ============================================================================
# PROGRESO / HEARTBEAT (NO LOGGING)
# ============================================================================

def progress(message: str) -> None:
    """
    Emite una línea de progreso SIEMPRE visible.

    Características:
    - Ignora SILENT_MODE
    - No usa logging (sin timestamps ni niveles)
    - Diseñado para feedback estructural:
        "(1/4) Películas"
        "Analizando biblioteca X"
    """
    try:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
    except Exception:
        # Nunca debe romper el flujo principal
        pass


def progressf(fmt: str, *args: Any) -> None:
    """
    Variante con formateo estilo printf para progreso.
    """
    try:
        msg = fmt % args if args else fmt
    except Exception:
        msg = fmt
    progress(msg)


# ============================================================================
# API PÚBLICA DE LOGGING
# ============================================================================

def debug(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """
    Logging DEBUG.

    - Respeta SILENT_MODE
    - always=True fuerza emisión
    """
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """
    Logging INFO.

    - Respeta SILENT_MODE
    - always=True fuerza emisión
    """
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """
    Logging WARNING.

    - Respeta SILENT_MODE
    - always=True fuerza emisión
    """
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """
    Logging ERROR.

    - SIEMPRE se emite
    - Ignora SILENT_MODE
    - `always` se acepta solo por compatibilidad con tests
    """
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.error(msg, *args, **kwargs)