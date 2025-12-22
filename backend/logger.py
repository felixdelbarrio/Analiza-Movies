from __future__ import annotations

"""
backend/logger.py

Logger central del proyecto.

Este módulo actúa como “fachada” (facade) sobre `logging` para que el resto del código
use una API única y consistente:

- info / warning / error / debug
- progress / progressf (salida “heartbeat” siempre visible, sin timestamps)
- utilidades para logging “debug contextual” y buffers de logs acotados:
    * debug_ctx(tag, msg)
    * append_bounded_log(logs, line, ...)

Por qué mover aquí la política de logs
--------------------------------------
Durante esta iteración hemos ido aplicando una filosofía común:

1) SILENT_MODE=True
   - Evitar ruido y, muy importante, evitar acumulación de strings (CPU/memoria).
   - Permitir “señales mínimas” de vida mediante `progress`.
2) DEBUG_MODE=True
   - Permitir trazas útiles, cortas y no voluminosas.
   - En SILENT+DEBUG: las trazas deben seguir siendo “mínimas” (usar progress o logs acotados).
3) No romper el pipeline
   - El logging no debe lanzar excepciones ni bloquear el flujo.

Este fichero implementa esa política en un solo lugar, evitando duplicación en:
- collection_analysis.py
- analiza_plex.py
- analiza_dlna.py
- omdb_client.py
- etc.

Compatibilidad
--------------
- Mantiene `LOGGER_NAME`, `get_logger()`, `debug/info/warning/error`,
  `progress/progressf` con la misma semántica.
- Los tests que referencian `LOGGER_NAME` siguen funcionando.

Notas de diseño
---------------
- No importamos `backend.config` directamente para evitar dependencias circulares.
  Leemos `backend.config` desde `sys.modules` si ya está importado.
- La configuración de logging es idempotente: `_ensure_configured()` puede llamarse
  repetidas veces.
- Añadimos helpers de *bounded logs* para que los orquestadores acumulen logs
  por item sin explotar memoria.

Uso recomendado
---------------
1) Logs normales:
   - logger.info("...") / logger.warning("...", always=True) / logger.error("...")
2) Señales mínimas:
   - logger.progress("[DLNA] ...")
3) Debug contextual (cuando DEBUG_MODE=True):
   - logger.debug_ctx("COLLECTION", "Fetched OMDb+Wiki ...")
   - logger.debug_ctx("OMDB", "Throttle sleeping ...")
4) Buffer de logs acotado (para devolver logs a capas superiores):
   - logger.append_bounded_log(logs, "[TRACE] ...")
   - logger.append_bounded_log(logs, "[ERROR] ...", force=True)

"""

import logging
import sys
from typing import Any, Final

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Nombre del logger principal de la aplicación.
# ⚠️ IMPORTANTE: los tests lo referencian explícitamente.
LOGGER_NAME: Final[str] = "movies_cleaner"

# Logger interno cacheado.
# Puede ser:
#   - logging.Logger (ejecución real)
#   - FakeLogger (inyectado por tests)
_LOGGER: Any = None

# Flag para asegurar inicialización idempotente del logger
_CONFIGURED: bool = False

# ============================================================================
# UTILIDADES CONFIG / FLAGS (sin importar backend.config directamente)
# ============================================================================


def _safe_get_cfg() -> Any | None:
    """
    Devuelve el módulo backend.config si ya ha sido importado.

    No lo importamos directamente para:
    - evitar dependencias circulares
    - permitir ejecución parcial (tests, herramientas, etc.)
    """
    return sys.modules.get("backend.config")


def _cfg_bool(name: str, default: bool = False) -> bool:
    """
    Lee un flag booleano desde backend.config de forma defensiva.
    """
    cfg = _safe_get_cfg()
    if cfg is None:
        return default
    try:
        return bool(getattr(cfg, name, default))
    except Exception:
        return default


def _cfg_int(name: str, default: int) -> int:
    """
    Lee un entero desde backend.config de forma defensiva.
    """
    cfg = _safe_get_cfg()
    if cfg is None:
        return default
    try:
        v = getattr(cfg, name, default)
        return int(v)
    except Exception:
        return default


def _cfg_str(name: str, default: str | None = None) -> str | None:
    """
    Lee un string desde backend.config de forma defensiva.
    """
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
    """
    SILENT_MODE global (si backend.config está cargado).
    """
    return _cfg_bool("SILENT_MODE", False)


def is_debug_mode() -> bool:
    """
    DEBUG_MODE global (si backend.config está cargado).
    """
    return _cfg_bool("DEBUG_MODE", False)


# ============================================================================
# RESOLUCIÓN DE LEVEL + EXTERNAL LOGGERS
# ============================================================================


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

    # 1) LOG_LEVEL explícito
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

    # 2) Flags de debug “heredados”
    if (
        _cfg_bool("WIKI_DEBUG", False)
        or _cfg_bool("OMDB_DEBUG", False)
        or _cfg_bool("DEBUG", False)
        or _cfg_bool("DEBUG_MODE", False)
    ):
        return logging.DEBUG

    return logging.INFO


def _apply_root_level(level: int) -> None:
    """
    Aplica el nivel al logger root y a todos sus handlers.

    Esto es necesario porque algunos entornos crean handlers antes de que lleguemos aquí.
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

    Si backend.config define HTTP_DEBUG=True, NO se silencian urllib3/requests/plexapi.
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
    - Aunque el root esté en DEBUG, evitar que urllib3/requests/plexapi inunden la consola.
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

    # Ya inicializado → re-sincronizamos niveles
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
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        _apply_root_level(level)

    _configure_external_loggers(level=level)

    _LOGGER = logging.getLogger(LOGGER_NAME)
    _CONFIGURED = True
    return _LOGGER


def get_logger() -> Any:
    """
    Devuelve el logger principal de la aplicación, asegurando inicialización previa.
    """
    return _ensure_configured()


# ============================================================================
# CONTROL DE SILENT MODE (para logs con niveles)
# ============================================================================


def _should_log(*, always: bool = False) -> bool:
    """
    Decide si un mensaje de logging debe emitirse.

    Reglas:
      - always=True → siempre se emite
      - SILENT_MODE=True → se suprimen debug/info/warning
      - error() ignora este mecanismo (siempre emite)
    """
    if always:
        return True
    return not is_silent_mode()


# ============================================================================
# PROGRESO / HEARTBEAT (NO logging)
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
# API PÚBLICA DE LOGGING (compat)
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

    - SIEMPRE se emite (ignora SILENT_MODE)
    - `always` se acepta solo por compatibilidad
    """
    log = _ensure_configured()
    kwargs.pop("always", None)
    log.error(msg, *args, **kwargs)


# ============================================================================
# NUEVO: DEBUG CONTEXTUAL (tag) + BOUNDED LOG BUFFER
# ============================================================================

# En ejecuciones grandes, acumular strings en memoria es caro.
# Unificamos aquí la política para todo el proyecto.

# Defaults conservadores; se pueden sobreescribir desde config:
#   LOGGER_LOGS_MAX_SILENT
#   LOGGER_LOGS_MAX_SILENT_DEBUG
#   LOGGER_LOGS_MAX_NORMAL
#   LOGGER_LOG_LINE_MAX_CHARS

_DEFAULT_LOGS_MAX_SILENT: Final[int] = 0
_DEFAULT_LOGS_MAX_SILENT_DEBUG: Final[int] = 12
_DEFAULT_LOGS_MAX_NORMAL: Final[int] = 50
_DEFAULT_LOG_LINE_MAX_CHARS: Final[int] = 500

_LOGS_TRUNCATED_SENTINEL: Final[str] = "[LOGS_TRUNCATED]"


def logs_limit() -> int:
    """
    Devuelve el límite de logs acumulables en una lista `logs` (por item),
    según SILENT_MODE/DEBUG_MODE.

    Se puede ajustar desde backend.config:
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
    Trunca una línea para evitar explosiones (por ejemplo JSON grande).
    """
    if not isinstance(text, str):
        text = str(text)

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
    Añade un log a una lista `logs` respetando límites por modo.

    - force=True:
        * Ignora límites (útil para errores críticos).
        * Aun así se trunca la línea para evitar payloads enormes.
    - tag:
        * Si se pasa, se prefija como "[TAG] ..."

    Política:
    - SILENT + !DEBUG -> limit=0 (no acumula nada).
    - SILENT + DEBUG  -> pocas líneas (útiles).
    - NO SILENT       -> más líneas, pero siempre capadas.
    """
    if not isinstance(logs, list):
        return

    prefix = f"[{tag}] " if isinstance(tag, str) and tag.strip() else ""
    msg = prefix + truncate_line(str(line))

    limit = logs_limit()

    if force:
        logs.append(msg)
        return

    # En silent con limit=0: no acumulamos logs.
    if limit <= 0:
        return

    if len(logs) >= limit:
        # Añadimos sentinel solo una vez.
        if not logs or logs[-1] != _LOGS_TRUNCATED_SENTINEL:
            logs.append(_LOGS_TRUNCATED_SENTINEL)
        return

    logs.append(msg)


def debug_ctx(tag: str, msg: object) -> None:
    """
    Debug contextual con “tag” consistente en todo el proyecto.

    Reglas:
    - DEBUG_MODE=False -> no-op
    - DEBUG_MODE=True:
        * SILENT_MODE=True  -> progress("[TAG][DEBUG] ...")  (señal mínima sin timestamps)
        * SILENT_MODE=False -> logger.info("[TAG][DEBUG] ...")
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
        # logging nunca debe romper el flujo
        if not is_silent_mode():
            try:
                print(f"[{t}][DEBUG] {text}")
            except Exception:
                pass