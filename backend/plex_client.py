from __future__ import annotations

"""
backend/plex_client.py

Cliente ligero y DEFENSIVO para Plex (plexapi) + helpers de extracción segura.

🧠 Contexto del bug
-------------------
En plexapi, acceder a ciertos atributos (p.ej. movie.originalTitle) puede disparar
un _reload() interno si el objeto está en modo lazy.

Ese reload implica una request HTTP contra Plex. Si Plex cierra la conexión sin
responder (RemoteDisconnected / ProtocolError), requests lanza ConnectionError y
el run entero puede romperse.

🎯 Objetivos del módulo
-----------------------
- Helpers robustos ante fallos de red y estructuras “raras”.
- Evitar que errores puntuales tumben el pipeline.
- Centralizar accesos lazy peligrosos (getattr defensivo).
- Mantener la política de logs definida en backend/logger.py.

✅ Métricas in-memory (thread-safe, O(1))
-----------------------------------------
Acumula contadores para:
- errores de red al leer atributos lazy,
- fallos generales de helpers,
- fallos al conectar.

Sin I/O a disco y sin datos sensibles (no IDs, no paths, no tokens).

Configuración (backend/config_plex.py)
--------------------------------------
Este archivo usa directamente variables del config:

- PLEX_METRICS_ENABLED: bool
- PLEX_METRICS_TOP_N: int
- PLEX_METRICS_LOG_ON_SILENT_DEBUG: bool
- PLEX_METRICS_LOG_EVEN_IF_ZERO: bool
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import requests  # type: ignore[import-untyped]
from plexapi.server import PlexServer  # type: ignore[import-not-found]

from backend import logger as _logger
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.config_plex import (
    BASEURL,
    EXCLUDE_PLEX_LIBRARIES,
    PLEX_PORT,
    PLEX_TOKEN,
    PLEX_METRICS_ENABLED,
    PLEX_METRICS_TOP_N,
    PLEX_METRICS_LOG_ON_SILENT_DEBUG,
    PLEX_METRICS_LOG_EVEN_IF_ZERO,
)

# ============================================================
#                        LOGGING
# ============================================================


def _log(msg: object) -> None:
    """
    Info normal (visible si SILENT_MODE=False).
    Preferimos backend/logger.py. Fallback a print solo si NO silent.
    """
    try:
        _logger.info(str(msg))
    except Exception:
        if not SILENT_MODE:
            print(str(msg))


def _log_always(msg: object) -> None:
    """
    Aviso importante, visible incluso en SILENT_MODE.
    Mantiene la semántica del proyecto: warning(always=True).
    """
    try:
        _logger.warning(str(msg), always=True)
    except Exception:
        print(str(msg))


def _log_debug(msg: object) -> None:
    """
    Debug contextual (respeta backend/config_base).

    - DEBUG_MODE=False => no emite.
    - DEBUG_MODE=True:
        * SILENT_MODE=True  -> progress
        * SILENT_MODE=False -> info
    """
    if not DEBUG_MODE:
        return

    text = str(msg)
    try:
        if SILENT_MODE:
            _logger.progress(f"[PLEX][DEBUG] {text}")
        else:
            _logger.info(f"[PLEX][DEBUG] {text}")
    except Exception:
        if not SILENT_MODE:
            print(text)


# ============================================================
#                 INT PARSING (mypy-friendly)
# ============================================================


def _int_or_none(value: object) -> int | None:
    """
    Convierte a int SOLO si es seguro.
    Evita: int(object) -> mypy call-overload.
    """
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            if not s:
                return None
            # acepta "12" o "12.0"
            try:
                f = float(s)
            except Exception:
                return None
            return int(f)
        return None
    except Exception:
        return None


def _to_int(value: object, default: int = 0) -> int:
    out = _int_or_none(value)
    return out if out is not None else default


# ============================================================
#                 MÉTRICAS (in-memory, thread-safe)
# ============================================================


@dataclass(slots=True)
class PlexMetrics:
    """
    Contadores agregados de eventos relevantes del cliente Plex.

    - Actualización concurrente (ThreadPool) => lock externo global.
    - Sin datos sensibles.
    """

    network_attr_errors_total: int = 0
    network_attr_errors_by_attr: Counter[str] = field(default_factory=Counter)
    network_attr_errors_by_exc: Counter[str] = field(default_factory=Counter)

    helper_failures_total: int = 0
    helper_failures_by_name: Counter[str] = field(default_factory=Counter)
    helper_failures_by_exc: Counter[str] = field(default_factory=Counter)

    connect_failures_total: int = 0
    connect_failures_by_exc: Counter[str] = field(default_factory=Counter)

    def snapshot(self) -> dict[str, object]:
        """Snapshot serializable (Counters -> dict)."""
        return {
            "network_attr_errors_total": self.network_attr_errors_total,
            "network_attr_errors_by_attr": dict(self.network_attr_errors_by_attr),
            "network_attr_errors_by_exc": dict(self.network_attr_errors_by_exc),
            "helper_failures_total": self.helper_failures_total,
            "helper_failures_by_name": dict(self.helper_failures_by_name),
            "helper_failures_by_exc": dict(self.helper_failures_by_exc),
            "connect_failures_total": self.connect_failures_total,
            "connect_failures_by_exc": dict(self.connect_failures_by_exc),
        }


_METRICS = PlexMetrics()
_METRICS_LOCK = Lock()


def _metrics_enabled() -> bool:
    return bool(PLEX_METRICS_ENABLED)


def _metrics_inc_network_attr_error(attr: str, exc: BaseException) -> None:
    if not _metrics_enabled():
        return
    exc_name = exc.__class__.__name__
    with _METRICS_LOCK:
        _METRICS.network_attr_errors_total += 1
        _METRICS.network_attr_errors_by_attr[str(attr)] += 1
        _METRICS.network_attr_errors_by_exc[exc_name] += 1


def _metrics_inc_helper_failure(helper_name: str, exc: BaseException) -> None:
    if not _metrics_enabled():
        return
    exc_name = exc.__class__.__name__
    with _METRICS_LOCK:
        _METRICS.helper_failures_total += 1
        _METRICS.helper_failures_by_name[str(helper_name)] += 1
        _METRICS.helper_failures_by_exc[exc_name] += 1


def _metrics_inc_connect_failure(exc: BaseException) -> None:
    if not _metrics_enabled():
        return
    exc_name = exc.__class__.__name__
    with _METRICS_LOCK:
        _METRICS.connect_failures_total += 1
        _METRICS.connect_failures_by_exc[exc_name] += 1


def get_plex_metrics_snapshot() -> dict[str, object]:
    """API pública: snapshot de métricas (estructura estable)."""
    if not _metrics_enabled():
        return {
            "network_attr_errors_total": 0,
            "network_attr_errors_by_attr": {},
            "network_attr_errors_by_exc": {},
            "helper_failures_total": 0,
            "helper_failures_by_name": {},
            "helper_failures_by_exc": {},
            "connect_failures_total": 0,
            "connect_failures_by_exc": {},
        }
    with _METRICS_LOCK:
        return _METRICS.snapshot()


def reset_plex_metrics() -> None:
    """API pública: resetea métricas (útil al inicio de cada run)."""
    global _METRICS
    with _METRICS_LOCK:
        _METRICS = PlexMetrics()


def log_plex_metrics(*, force: bool = False) -> None:
    """
    API pública: imprime un resumen de métricas.

    Política:
    - SILENT_MODE=True:
        - emite solo si force=True
        - o si DEBUG_MODE=True y PLEX_METRICS_LOG_ON_SILENT_DEBUG=True
        - usa _log_debug (progress en silent)
    - NO SILENT:
        - imprime si hay fallos
        - o si force=True
        - o si PLEX_METRICS_LOG_EVEN_IF_ZERO=True
    """
    snap = get_plex_metrics_snapshot()

    net_total = _to_int(snap.get("network_attr_errors_total", 0), 0)
    helper_total = _to_int(snap.get("helper_failures_total", 0), 0)
    conn_total = _to_int(snap.get("connect_failures_total", 0), 0)

    if (
        not force
        and not bool(PLEX_METRICS_LOG_EVEN_IF_ZERO)
        and net_total == 0
        and helper_total == 0
        and conn_total == 0
    ):
        return

    try:
        top_n = max(1, int(PLEX_METRICS_TOP_N))
    except Exception:
        top_n = 5

    def _top(d: object, n: int) -> list[tuple[str, int]]:
        if not isinstance(d, Mapping):
            return []
        items: list[tuple[str, int]] = []
        for k, v in d.items():
            iv = _int_or_none(v)
            if iv is None:
                continue
            items.append((str(k), iv))
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        return items[:n]

    top_attrs = _top(snap.get("network_attr_errors_by_attr"), top_n)
    top_exc = _top(snap.get("network_attr_errors_by_exc"), top_n)
    top_helpers = _top(snap.get("helper_failures_by_name"), top_n)
    top_helper_exc = _top(snap.get("helper_failures_by_exc"), top_n)
    top_conn_exc = _top(snap.get("connect_failures_by_exc"), top_n)

    lines: list[str] = []
    lines.append("[PLEX] Metrics summary")
    lines.append(f"  - connect_failures: {conn_total}")
    if top_conn_exc:
        lines.append(f"    * top_exc: {top_conn_exc}")

    lines.append(f"  - network_attr_errors: {net_total}")
    if top_attrs:
        lines.append(f"    * top_attrs: {top_attrs}")
    if top_exc:
        lines.append(f"    * top_exc: {top_exc}")

    lines.append(f"  - helper_failures: {helper_total}")
    if top_helpers:
        lines.append(f"    * top_helpers: {top_helpers}")
    if top_helper_exc:
        lines.append(f"    * top_exc: {top_helper_exc}")

    msg = "\n".join(lines)

    if SILENT_MODE:
        if force or (DEBUG_MODE and bool(PLEX_METRICS_LOG_ON_SILENT_DEBUG)):
            _log_debug(msg)
        return

    _log(msg)


# ============================================================
#               Helpers defensivos (plexapi / red)
# ============================================================


def _is_networkish_exception(exc: BaseException) -> bool:
    """
    Heurística para detectar errores típicos de red en plexapi.

    Cubre:
    - requests.exceptions.RequestException
    - OSError / ConnectionError
    - nombres típicos de http.client / urllib3 (RemoteDisconnected, ProtocolError, timeouts, resets)
    """
    try:
        if isinstance(exc, (requests.exceptions.RequestException, OSError, ConnectionError)):
            return True
    except Exception:
        pass

    name = exc.__class__.__name__.lower()
    if "remotedisconnected" in name:
        return True
    if "protocolerror" in name:
        return True
    if "connectionaborted" in name or "connectionreset" in name:
        return True
    if "readtimeout" in name or "connecttimeout" in name or "timeout" in name:
        return True

    return False


def _safe_getattr(obj: object, attr: str, default: Any = None) -> Any:
    """
    getattr() robusto:
    - Nunca lanza.
    - Captura fallos de red provocados por lazy reload de plexapi.
    - Devuelve default si algo va mal.

    Logs:
    - Error de red: warning always=True.
    - Otros errores: debug.
    """
    try:
        return getattr(obj, attr, default)
    except Exception as exc:
        if _is_networkish_exception(exc):
            _metrics_inc_network_attr_error(attr, exc)
            _log_always(f"[PLEX] Network error reading attribute {attr!r} (lazy reload skipped): {exc!r}")
            return default

        _metrics_inc_helper_failure("_safe_getattr", exc)
        _log_debug(f"_safe_getattr({attr!r}) failed: {exc!r}")
        return default


def _safe_getattr_str(obj: object, attr: str) -> str | None:
    """Lee un atributo string de forma segura (strip; vacío => None)."""
    val = _safe_getattr(obj, attr, None)
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    return None


# ============================================================
#                     CONEXIÓN A PLEX
# ============================================================


def _build_plex_base_url() -> str:
    """
    Construye la URL base para Plex:
        BASEURL="http://192.168.1.10"
        PLEX_PORT=32400
        -> "http://192.168.1.10:32400"
    """
    if not BASEURL or not str(BASEURL).strip():
        raise RuntimeError("BASEURL no está definido en el entorno (.env)")

    base = str(BASEURL).rstrip("/")
    try:
        port = int(PLEX_PORT)
    except Exception:
        port = 32400
    return f"{base}:{port}"


def connect_plex() -> PlexServer:
    """
    Conecta a Plex y devuelve PlexServer.

    - Faltan BASEURL/PLEX_TOKEN -> RuntimeError
    - Fallo de conexión -> se re-lanza (caller decide)
    """
    if not BASEURL or not str(BASEURL).strip() or not PLEX_TOKEN or not str(PLEX_TOKEN).strip():
        raise RuntimeError("Faltan BASEURL o PLEX_TOKEN en el .env")

    base_url = _build_plex_base_url()

    if DEBUG_MODE:
        _log_debug(f"Connecting to Plex at {base_url}")

    try:
        plex = PlexServer(base_url, str(PLEX_TOKEN))
    except Exception as exc:
        _metrics_inc_connect_failure(exc)
        _log_always(f"[PLEX] ERROR conectando a Plex ({base_url}): {exc!r}")
        raise

    if not SILENT_MODE:
        _log(f"[PLEX] Conectado a Plex: {base_url}")
    elif DEBUG_MODE:
        _log_debug("Connected")

    return plex


# ============================================================
#              BIBLIOTECAS A ANALIZAR
# ============================================================


def get_libraries_to_analyze(plex: PlexServer) -> list[object]:
    """
    Devuelve bibliotecas de Plex excluyendo EXCLUDE_PLEX_LIBRARIES.

    Contrato:
    - Nunca lanza.
    - En error -> [] y log always=True.
    """
    try:
        sections = plex.library.sections()
    except Exception as exc:
        _metrics_inc_helper_failure("get_libraries_to_analyze.sections", exc)
        _log_always(f"[PLEX] ERROR obteniendo secciones: {exc!r}")
        return []

    excluded = set(EXCLUDE_PLEX_LIBRARIES or [])
    selected: list[object] = []

    for section in sections:
        name = _safe_getattr_str(section, "title") or ""
        if name and name in excluded:
            if not SILENT_MODE:
                _log(f"[PLEX] Saltando biblioteca excluida: {name}")
            else:
                _log_debug(f"Skipping excluded library: {name}")
            continue
        selected.append(section)

    if DEBUG_MODE:
        _log_debug(f"Libraries selected: {len(selected)} (excluded={len(excluded)})")

    return selected


# ============================================================
#            INFO DE ARCHIVOS DE PELÍCULAS
# ============================================================


def get_movie_file_info(movie: object) -> tuple[str | None, int | None]:
    """
    Devuelve (ruta_principal, tamaño_total_en_bytes).

    - Ruta: primer part.file válido.
    - Tamaño: suma de part.size válidos.

    Best-effort: nunca lanza.
    """
    try:
        media_seq = _safe_getattr(movie, "media", None)
        if not isinstance(media_seq, Sequence) or not media_seq:
            return None, None

        best_path: str | None = None
        total_size: int = 0
        size_seen = False

        for media in media_seq:
            parts = _safe_getattr(media, "parts", None) or []
            if not isinstance(parts, Sequence):
                continue

            for part in parts:
                file_path = _safe_getattr(part, "file", None)
                size_val = _safe_getattr(part, "size", None)

                if best_path is None and isinstance(file_path, str) and file_path.strip():
                    best_path = file_path.strip()

                if isinstance(size_val, bool):
                    continue

                if isinstance(size_val, (int, float)):
                    iv = int(size_val)
                    if iv > 0:
                        total_size += iv
                        size_seen = True
                    continue

                # ✅ str/object (plexapi a veces devuelve str; evitamos Optional -> int)
                if isinstance(size_val, str):
                    iv2 = _int_or_none(size_val)
                    if iv2 is not None and iv2 > 0:
                        total_size += iv2
                        size_seen = True

        if best_path is None:
            return None, None

        return best_path, (total_size if size_seen else None)

    except Exception as exc:
        _metrics_inc_helper_failure("get_movie_file_info", exc)
        _log_debug(f"get_movie_file_info failed: {exc!r}")
        return None, None


# ============================================================
#           IMDB ID DESDE GUIDS PLEX
# ============================================================


def get_imdb_id_from_plex_guid(guid: str) -> str | None:
    """Extrae imdb_id (tt1234567) desde un guid de Plex (reconoce 'imdb://')."""
    if not isinstance(guid, str) or "imdb://" not in guid:
        return None
    try:
        after = guid.split("imdb://", 1)[1]
        imdb_id = after.split("?", 1)[0].strip()
        return imdb_id or None
    except Exception:
        return None


def get_imdb_id_from_movie(movie: object) -> str | None:
    """
    Intenta obtener imdb_id desde:
      1) movie.guids
      2) movie.guid (fallback)
    """
    try:
        guids = _safe_getattr(movie, "guids", None) or []
        if isinstance(guids, Sequence):
            for g in guids:
                gid = _safe_getattr(g, "id", None)
                if isinstance(gid, str):
                    imdb_id = get_imdb_id_from_plex_guid(gid)
                    if imdb_id:
                        return imdb_id
    except Exception as exc:
        _metrics_inc_helper_failure("get_imdb_id_from_movie.guids", exc)
        _log_debug(f"get_imdb_id_from_movie.guids failed: {exc!r}")

    guid_main = _safe_getattr(movie, "guid", None)
    if isinstance(guid_main, str):
        return get_imdb_id_from_plex_guid(guid_main)

    return None


# ============================================================
#        MEJOR TÍTULO PARA BÚSQUEDA (OMDb / Wiki)
# ============================================================


def get_best_search_title(movie: object) -> str | None:
    """
    Devuelve el mejor título para buscar en OMDb/Wiki.

    Prioridad:
      1) originalTitle
      2) title
    """
    t1 = _safe_getattr_str(movie, "originalTitle")
    if t1:
        return t1

    t2 = _safe_getattr_str(movie, "title")
    if t2:
        return t2

    return None


__all__ = [
    "connect_plex",
    "get_libraries_to_analyze",
    "get_movie_file_info",
    "get_imdb_id_from_movie",
    "get_imdb_id_from_plex_guid",
    "get_best_search_title",
    "get_plex_metrics_snapshot",
    "reset_plex_metrics",
    "log_plex_metrics",
]