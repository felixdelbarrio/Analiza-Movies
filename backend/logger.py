from __future__ import annotations

import logging
import sys
from typing import Any

# Nombre del logger principal (los tests lo usan explícitamente)
LOGGER_NAME: str = "plex_movies_cleaner"

# Logger interno y flag de configuración
_LOGGER: Any = None  # puede ser logging.Logger o un FakeLogger de tests
_CONFIGURED: bool = False


def _safe_get_cfg() -> Any | None:
    return sys.modules.get("backend.config")


def _resolve_level_from_config() -> int:
    """
    Decide el nivel de logging (root) en base a backend.config.

    Prioridad:
      1) config.LOG_LEVEL (str: "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL")
      2) flags de debug (p.ej. WIKI_DEBUG=True) -> DEBUG
      3) fallback -> INFO
    """
    cfg = _safe_get_cfg()
    if cfg is None:
        return logging.INFO

    # 1) LOG_LEVEL explícito
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

    # 2) Cualquier flag debug que quieras soportar
    def _flag(name: str) -> bool:
        try:
            return bool(getattr(cfg, name, False))
        except Exception:
            return False

    if _flag("WIKI_DEBUG") or _flag("OMDB_DEBUG") or _flag("DEBUG"):
        return logging.DEBUG

    return logging.INFO


def _apply_root_level(level: int) -> None:
    """
    Aplica nivel al root y a sus handlers (importante en algunos entornos).
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers:
        try:
            h.setLevel(level)
        except Exception:
            pass


def _ensure_configured() -> Any:
    """
    Inicializa el logger solo una vez y lo devuelve.

    - Si `_CONFIGURED` ya es True y `_LOGGER` no es None, devuelve `_LOGGER`
      tal cual (esto permite a los tests inyectar un FakeLogger).
    - Si no está configurado, inicializa logging básico y crea el logger
      con nombre LOGGER_NAME.
    """
    global _LOGGER, _CONFIGURED

    if _CONFIGURED and _LOGGER is not None:
        # OJO: aun así podemos ajustar el nivel si config cambió
        try:
            _apply_root_level(_resolve_level_from_config())
        except Exception:
            pass
        return _LOGGER

    level = _resolve_level_from_config()

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        # Si ya hay handlers (por ejemplo, el entorno), ajustamos el nivel igualmente
        _apply_root_level(level)

    _LOGGER = logging.getLogger(LOGGER_NAME)
    _CONFIGURED = True
    return _LOGGER


def _should_log(*, always: bool = False) -> bool:
    """
    Decide si se debe loguear en función de SILENT_MODE.

    Reglas:
      - Si always=True → siempre True.
      - Si existe backend.config y tiene SILENT_MODE:
          devuelve not SILENT_MODE.
      - Si no existe backend.config en sys.modules, o no tiene SILENT_MODE:
          devuelve True (por defecto se loguea).
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
    """Devuelve el logger principal, asegurando su configuración previa."""
    return _ensure_configured()


def debug(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """Debug, sujeto a SILENT_MODE salvo que always=True."""
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """Info, sujeto a SILENT_MODE salvo que always=True."""
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """Warning, sujeto a SILENT_MODE salvo que always=True."""
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """
    Error: **siempre** se loguea.

    - Ignora SILENT_MODE (siempre se escribe).
    - Acepta `always` solo por compatibilidad con tests, pero no lo usa.
    """
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.error(msg, *args, **kwargs)