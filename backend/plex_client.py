from __future__ import annotations

"""
backend/plex_client.py

Cliente ligero y DEFENSIVO para Plex (plexapi) + helpers de extracción segura.

🧠 Contexto del bug observado
----------------------------
En plexapi, acceder a ciertos atributos (p.ej. movie.originalTitle) puede disparar
un _reload() interno si el objeto está en modo lazy.

Ese reload implica una request HTTP contra Plex.
Si Plex cierra la conexión sin responder (RemoteDisconnected / ProtocolError),
requests lanza ConnectionError y el run entero puede romperse.

🎯 Objetivo de este módulo
-------------------------
- Proveer helpers que:
    ✔ sean robustos ante fallos de red
    ✔ NO tumben el pipeline por errores puntuales
    ✔ mantengan la semántica actual del proyecto
- Centralizar accesos lazy peligrosos.
- Mantener la política de logs definida en backend/logger.py.

✅ Métricas de fallos Plex (NUEVO)
---------------------------------
Este módulo acumula métricas in-memory (costo ~O(1)) para responder a preguntas como:
- ¿Cuántos fallos de red hubo leyendo atributos lazy?
- ¿Qué atributos fallan más (originalTitle/title/media/...)?
- ¿En qué helpers se concentran los problemas?

Características:
- No escribe a disco (no queremos I/O aquí).
- No expone datos sensibles.
- Permite dump manual (log_plex_metrics) y reset (reset_plex_metrics).
- Los errores de red se loguean (warning always=True) como antes, y además cuentan.

⚠️ Nota de tipado
-----------------
plexapi no expone typing estricto.
Usamos object / Any + getattr defensivo.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import requests  # type: ignore[import-not-found]
from plexapi.server import PlexServer  # type: ignore[import-not-found]

from backend import logger as _logger
from backend.config import (
    BASEURL,
    DEBUG_MODE,
    EXCLUDE_PLEX_LIBRARIES,
    PLEX_PORT,
    PLEX_TOKEN,
    SILENT_MODE,
)

# ============================================================
#                        LOGGING
# ============================================================

def _log(msg: object) -> None:
    """Info normal (visible si SILENT_MODE=False)."""
    try:
        _logger.info(str(msg))
    except Exception:
        if not SILENT_MODE:
            print(str(msg))


def _log_always(msg: object) -> None:
    """Aviso importante, visible incluso en SILENT_MODE."""
    try:
        _logger.warning(str(msg), always=True)
    except Exception:
        print(str(msg))


def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False: no hace nada
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
#                 MÉTRICAS (in-memory, thread-safe)
# ============================================================

@dataclass
class PlexMetrics:
    """
    Contadores agregados de eventos relevantes del cliente Plex.

    NOTAS DE DISEÑO
    --------------
    - Se actualiza desde distintos hilos (ThreadPool) => lock interno.
    - No guardamos IDs de películas ni paths (privacidad / ruido).
    - Guardamos "clase de error" y "atributo/helper" para diagnóstico.

    Campos:
    - network_attr_errors_total: fallos de red al leer atributos (lazy reload).
    - network_attr_errors_by_attr: top atributos que fallan (originalTitle, title...).
    - network_attr_errors_by_exc: top tipos de excepción.
    - helper_failures_total: fallos generales capturados por helpers (no necesariamente red).
    - helper_failures_by_name: top helpers con excepciones.
    - connect_failures_total: fallos conectando a Plex.
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
        """
        Devuelve una snapshot serializable (para logs/debug).
        (No exportamos Counters directamente para no sorprender en dumps.)
        """
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


def _metrics_inc_network_attr_error(attr: str, exc: BaseException) -> None:
    """Incrementa métricas de fallo de red en lectura de atributo."""
    exc_name = exc.__class__.__name__
    with _METRICS_LOCK:
        _METRICS.network_attr_errors_total += 1
        _METRICS.network_attr_errors_by_attr[attr] += 1
        _METRICS.network_attr_errors_by_exc[exc_name] += 1


def _metrics_inc_helper_failure(helper_name: str, exc: BaseException) -> None:
    """Incrementa métricas de fallo interno de helper (defensivo)."""
    exc_name = exc.__class__.__name__
    with _METRICS_LOCK:
        _METRICS.helper_failures_total += 1
        _METRICS.helper_failures_by_name[helper_name] += 1
        _METRICS.helper_failures_by_exc[exc_name] += 1


def _metrics_inc_connect_failure(exc: BaseException) -> None:
    """Incrementa métricas de fallo de conexión a Plex."""
    exc_name = exc.__class__.__name__
    with _METRICS_LOCK:
        _METRICS.connect_failures_total += 1
        _METRICS.connect_failures_by_exc[exc_name] += 1


def get_plex_metrics_snapshot() -> dict[str, object]:
    """
    API pública: snapshot de métricas.

    Útil si quieres incluirlo en un reporte final o test.
    """
    with _METRICS_LOCK:
        return _METRICS.snapshot()


def reset_plex_metrics() -> None:
    """API pública: resetea métricas (por ejemplo, al inicio de un run)."""
    global _METRICS
    with _METRICS_LOCK:
        _METRICS = PlexMetrics()


def log_plex_metrics(*, force: bool = False) -> None:
    """
    API pública: imprime un resumen de métricas.

    Política de logs:
    - Si SILENT_MODE=True:
        - solo emite si DEBUG_MODE=True o force=True
        - usa debug/progress (vía _log_debug)
    - Si SILENT_MODE=False:
        - si hay fallos, imprime resumen en INFO
        - si no hay fallos, no spamea (a menos que force=True)

    Uso recomendado:
        - al final del run / en flush_external_caches()
    """
    snap = get_plex_metrics_snapshot()

    net_total = int(snap.get("network_attr_errors_total") or 0)
    helper_total = int(snap.get("helper_failures_total") or 0)
    conn_total = int(snap.get("connect_failures_total") or 0)

    if not force and net_total == 0 and helper_total == 0 and conn_total == 0:
        return

    # Construimos un resumen pequeño y accionable.
    def _top(d: dict[str, int], n: int = 5) -> list[tuple[str, int]]:
        items = [(k, int(v)) for k, v in d.items()]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:n]

    top_attrs = _top(snap.get("network_attr_errors_by_attr") or {}, 5)
    top_exc = _top(snap.get("network_attr_errors_by_exc") or {}, 5)
    top_helpers = _top(snap.get("helper_failures_by_name") or {}, 5)
    top_conn_exc = _top(snap.get("connect_failures_by_exc") or {}, 5)

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

    msg = "\n".join(lines)

    if SILENT_MODE:
        if DEBUG_MODE or force:
            _log_debug(msg)
        return

    # No-silent: si hubo fallos relevantes, INFO es aceptable (aporta valor).
    _log(msg)


# ============================================================
#               Helpers defensivos (plexapi / red)
# ============================================================

def _is_networkish_exception(exc: BaseException) -> bool:
    """
    Heurística para detectar errores típicos de red en plexapi:
    - requests / urllib3
    - OSError
    - RemoteDisconnected / ProtocolError
    """
    if isinstance(exc, (requests.exceptions.RequestException, OSError, ConnectionError)):
        return True

    name = exc.__class__.__name__.lower()
    return "remotedisconnected" in name or "protocolerror" in name


def _safe_getattr(obj: object, attr: str, default: Any = None) -> Any:
    """
    getattr() robusto:

    - Nunca lanza.
    - Captura fallos de red provocados por lazy reload de plexapi.
    - Devuelve default si algo va mal.

    Métricas:
    - Si detectamos error de red -> incrementa contadores (atributo + tipo de error).
    """
    try:
        return getattr(obj, attr, default)
    except Exception as exc:
        if _is_networkish_exception(exc):
            _metrics_inc_network_attr_error(attr, exc)
            _log_always(
                f"[PLEX] Network error reading attribute {attr!r} "
                f"(lazy reload skipped): {exc!r}"
            )
            return default

        _log_debug(f"_safe_getattr({attr}) failed: {exc!r}")
        return default


def _safe_getattr_str(obj: object, attr: str) -> str | None:
    """
    Lee un atributo string de forma segura:

    - Devuelve str.strip() si existe y no está vacío
    - Devuelve None si falta / vacío / error de red / estructura inesperada
    """
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

        BASEURL = "http://192.168.1.10"
        PLEX_PORT = 32400
        -> "http://192.168.1.10:32400"
    """
    if not BASEURL or not str(BASEURL).strip():
        raise RuntimeError("BASEURL no está definido en el entorno (.env)")

    base = str(BASEURL).rstrip("/")
    return f"{base}:{int(PLEX_PORT)}"


def connect_plex() -> PlexServer:
    """
    Conecta a Plex y devuelve PlexServer.

    Reglas:
    - Faltan BASEURL/PLEX_TOKEN -> RuntimeError
    - Fallo de conexión -> se re-lanza (caller decide)

    Métricas:
    - Si falla la conexión, incrementa connect_failures.
    """
    if not BASEURL or not PLEX_TOKEN:
        raise RuntimeError("Faltan BASEURL o PLEX_TOKEN en el .env")

    base_url = _build_plex_base_url()

    if DEBUG_MODE:
        _log_debug(f"Connecting to Plex at {base_url}")

    try:
        plex = PlexServer(base_url, PLEX_TOKEN)
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

    - Nunca lanza.
    - En error -> devuelve [] y log always=True.
    - Métricas: helper_failures si no podemos listar secciones.
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

    Nunca lanza.
    Métricas:
    - Si la estructura es rara o hay excepciones inesperadas, cuenta helper_failures.
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

                if isinstance(size_val, int) and size_val > 0:
                    total_size += size_val
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
    """
    Extrae imdb_id (tt1234567) desde un guid de Plex.

    Nota:
    - Solo reconoce guids con 'imdb://'
    """
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

    Nunca lanza.
    Métricas:
    - Si hay excepción inesperada, cuenta helper_failures.
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
        # No spameamos en always: esto suele ser estructura/lazy; debug basta.
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

    - Devuelve None si no hay título usable.
    - Es 100% no-throw (captura errores de red vía _safe_getattr_str/_safe_getattr).

    Métricas:
    - Los fallos de red al leer atributos se contabilizan en _safe_getattr.
    """
    t1 = _safe_getattr_str(movie, "originalTitle")
    if t1:
        return t1

    t2 = _safe_getattr_str(movie, "title")
    if t2:
        return t2

    return None