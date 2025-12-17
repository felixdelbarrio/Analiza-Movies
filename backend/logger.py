# backend/logger.py
from __future__ import annotations

import logging
import sys
from typing import Any

# Nombre del logger principal (los tests lo usan explícitamente)
LOGGER_NAME: str = "plex_movies_cleaner"

# Logger interno y flag de configuración
_LOGGER: Any = None  # puede ser logging.Logger o un FakeLogger de tests
_CONFIGURED: bool = False


# ============================================================
# Configuración interna
# ============================================================


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
        return _LOGGER

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

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
        # Config no cargada o inexistente → por defecto, logueamos
        return True

    try:
        silent = getattr(cfg, "SILENT_MODE", False)
    except Exception:
        return True

    return not bool(silent)


def get_logger() -> Any:
    """Devuelve el logger principal, asegurando su configuración previa."""
    return _ensure_configured()


# ============================================================
# Wrappers públicos
# ============================================================


def debug(msg: str, *args: Any, always: bool = False, **kwargs: Any) -> None:
    """Debug, sujeto a SILENT_MODE salvo que always=True."""
    if not _should_log(always=always):
        return
    log = _ensure_configured()
    # Quitamos 'always' por si nos lo pasan también en **kwargs
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