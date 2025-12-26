from __future__ import annotations

"""
backend/collection_analiza_dlna.py  (antes: analiza_dlna.py)

Orquestador principal de análisis DLNA/UPnP (streaming).

Problema reportado (Plex DLNA / UPnP)
------------------------------------
1) Bucle “infinito” en el browse:
   - Causa típica: ciclos en el grafo de contenedores (virtual folders / alias / vistas)
   - Antes (histórico): stack sin visited => se puede reinsertar el mismo container id infinitamente.

2) Análisis múltiple del mismo fichero:
   - Causa típica: el mismo item aparece en múltiples contenedores (virtual views),
     o repeticiones por paginación/browse inconsistentes.
   - Antes (histórico): se acumulaban items sin dedupe global (por run).

Solución aplicada (robusta y conservadora)
------------------------------------------
A) Traversal seguro (anti-loop):
   - visited_containers: set[object_id]
   - límites duros configurables (fuses):
       * DLNA_TRAVERSE_MAX_DEPTH
       * DLNA_TRAVERSE_MAX_CONTAINERS
       * DLNA_TRAVERSE_MAX_ITEMS_TOTAL
       * DLNA_TRAVERSE_MAX_EMPTY_PAGES
       * DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER
   - stop conditions: depth, containers, items_total, empty_pages, pages_per_container

B) Dedupe de items (doble capa):
   - Dedupe LOCAL por contenedor raíz escaneado:
       * visited_item_urls_local
       * visited_item_ids_local
   - ✅ Dedupe GLOBAL por run (entre contenedores seleccionados):
       * seen_item_urls_global
       * seen_item_ids_global
     => evita analizar el mismo fichero aunque aparezca en varias carpetas/vistas.

C) Cache de endpoints ContentDirectory (por device.location):
   - Evita recalcular control_url/service_type en cada browse.
   - Reduce presión y latencia especialmente en Plex.

Logs (alineados con backend/logger.py)
-------------------------------------
- progress: “heartbeat” (siempre visible)
- info/warn/error: delega en logger.py (SILENT/DEBUG ya gestionado)
- debug_ctx("DLNA", ...): para contexto cuando DEBUG_MODE

"""

import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from typing import Protocol, TypeVar
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from backend import logger as logger
from backend.collection_analysis import analyze_movie, flush_external_caches
from backend.config_reports import (
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)

from backend.config_omdb import (
    OMDB_HTTP_MAX_CONCURRENCY,
    OMDB_HTTP_MIN_INTERVAL_SECONDS,
)

from backend.config_plex import (
    PLEX_ANALYZE_WORKERS,
)

from backend.config_base import (
    DEBUG_MODE,
    SILENT_MODE,
)
from backend.decision_logic import sort_filtered_rows
from backend.dlna_discovery import DLNADevice, discover_dlna_devices
from backend.movie_input import MovieInput
from backend.omdb_client import get_omdb_metrics_snapshot, reset_omdb_metrics
from backend.reporting import (
    open_all_csv_writer,
    open_filtered_csv_writer_only_if_rows,
    open_suggestions_csv_writer,
)

# ---------------------------------------------------------------------------
# Opcionales (resilience + métricas agregadas)
# ---------------------------------------------------------------------------

try:
    from backend.resilience import CircuitBreaker, call_with_resilience  # type: ignore
except Exception:  # pragma: no cover
    CircuitBreaker = None  # type: ignore[assignment]
    call_with_resilience = None  # type: ignore[assignment]

try:
    from backend.run_metrics import METRICS  # type: ignore
except Exception:  # pragma: no cover

    class _NoopMetrics:
        def incr(self, key: str, n: int = 1) -> None:
            return

        def observe_ms(self, key: str, ms: float) -> None:
            return

        def add_error(self, subsystem: str, action: str, *, endpoint: str | None, detail: str) -> None:
            return

        def snapshot(self) -> dict[str, object]:
            return {"counters": {}, "derived": {}, "timings_ms": {}, "errors": []}

    METRICS = _NoopMetrics()  # type: ignore[assignment]

# ============================================================================
# CONFIG (imports tolerantes para mantener compatibilidad)
# ============================================================================

try:
    from backend.config import DLNA_SCAN_WORKERS  # type: ignore
except Exception:  # pragma: no cover
    DLNA_SCAN_WORKERS = 2  # type: ignore

try:
    from backend.config import DLNA_BROWSE_MAX_RETRIES  # type: ignore
except Exception:  # pragma: no cover
    DLNA_BROWSE_MAX_RETRIES = 2  # type: ignore

try:
    from backend.config import DLNA_CB_FAILURE_THRESHOLD  # type: ignore
except Exception:  # pragma: no cover
    DLNA_CB_FAILURE_THRESHOLD = 5  # type: ignore

try:
    from backend.config import DLNA_CB_OPEN_SECONDS  # type: ignore
except Exception:  # pragma: no cover
    DLNA_CB_OPEN_SECONDS = 20.0  # type: ignore

# ✅ Traversal fuses (añadidos a config.py)
try:
    from backend.config import (  # type: ignore
        DLNA_TRAVERSE_MAX_CONTAINERS,
        DLNA_TRAVERSE_MAX_DEPTH,
        DLNA_TRAVERSE_MAX_EMPTY_PAGES,
        DLNA_TRAVERSE_MAX_ITEMS_TOTAL,
        DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER,
    )
except Exception:  # pragma: no cover
    # Fallbacks razonables (solo para backwards compatibility)
    DLNA_TRAVERSE_MAX_DEPTH = 30  # type: ignore
    DLNA_TRAVERSE_MAX_CONTAINERS = 20_000  # type: ignore
    DLNA_TRAVERSE_MAX_ITEMS_TOTAL = 300_000  # type: ignore
    DLNA_TRAVERSE_MAX_EMPTY_PAGES = 3  # type: ignore
    DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER = 20_000  # type: ignore

# Browse page size (constante deliberada: afecta estabilidad/latencia de algunos servers)
_BROWSE_PAGE_SIZE: int = 200

# Progreso / debug
_PROGRESS_EVERY_N_ITEMS: int = 100

# Workers caps (hard clamp)
_MAX_WORKERS_CAP: int = 64

# Bounded inflight (analyze)
_DEFAULT_MAX_INFLIGHT_FACTOR: int = 4

# Retry backoff defaults
_DLNA_RETRY_BASE_SECONDS: float = 0.35
_DLNA_RETRY_CAP_SECONDS: float = 6.0
_DLNA_RETRY_JITTER: float = 0.35

# ============================================================================
# TIPADO ESTRICTO (writers)
# ============================================================================

_Row = dict[str, object]


class _RowWriter(Protocol):
    """Interfaz mínima que exponen nuestros CSV writers del módulo reporting."""

    def write_row(self, row: _Row) -> None: ...


_WAll = TypeVar("_WAll", bound=_RowWriter)
_WSugg = TypeVar("_WSugg", bound=_RowWriter)

# ============================================================================
# MODELOS INTERNOS (DLNA)
# ============================================================================


@dataclass(frozen=True, slots=True)
class _DlnaContainer:
    """Contenedor DLNA (carpeta/categoría) accesible por ObjectID."""
    object_id: str
    title: str


@dataclass(frozen=True, slots=True)
class _DlnaVideoItem:
    """
    Item de vídeo DLNA.

    - title: título presentado por DIDL-Lite.
    - resource_url: URL del recurso (res).
    - size_bytes: tamaño aproximado si el server lo aporta.
    - item_id: atributo DIDL item@id si existe (útil para dedupe adicional).
    """
    title: str
    resource_url: str
    size_bytes: int | None
    item_id: str | None


@dataclass(frozen=True, slots=True)
class _TraversalLimits:
    """
    “Fuses” de traversal (anti-run-infinito / anti-catálogo virtual infinito).

    Importante:
    - Son límites duros; cuando se alcanzan, el traversal corta.
    - Deben ser suficientemente altos para librerías grandes,
      pero suficientemente bajos para evitar loops por estructuras virtuales.
    """
    max_depth: int
    max_containers: int
    max_items_total: int
    max_pages_per_container: int
    max_empty_pages: int


@dataclass(slots=True)
class _GlobalDedupeState:
    """
    Dedupe global por run (entre contenedores seleccionados).

    Objetivo:
    - Evitar analizar 2 veces el mismo fichero aunque aparezca en múltiples vistas/carpetas (Plex).
    - Este dedupe es independiente del dedupe local del traversal.
    """
    seen_item_urls: set[str]
    seen_item_ids: set[str]
    skipped_global: int = 0

# ============================================================================
# PARSEO DE TÍTULO / AÑO
# ============================================================================

_TITLE_YEAR_SUFFIX_RE: re.Pattern[str] = re.compile(
    r"(?P<base>.*?)"
    r"(?P<sep>\s*\.?\s*)"
    r"\(\s*(?P<year>\d{4})\s*\)\s*$"
)

# ============================================================================
# Circuit breaker global DLNA (por endpoint)
# ============================================================================

_DLNA_BREAKER = None
if CircuitBreaker is not None:
    try:
        _DLNA_BREAKER = CircuitBreaker(
            failure_threshold=int(DLNA_CB_FAILURE_THRESHOLD),
            open_seconds=float(DLNA_CB_OPEN_SECONDS),
            half_open_max_calls=1,
        )
    except Exception:  # pragma: no cover
        _DLNA_BREAKER = None

# ============================================================================
# OMDb metrics helpers (solo para SILENT+DEBUG)
# ============================================================================


def _metrics_get_int(m: Mapping[str, object], key: str) -> int:
    """Parse defensivo de métricas (por si cambian tipos)."""
    try:
        v = m.get(key, 0)
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    except Exception:
        pass
    return 0


def _metrics_diff(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, int]:
    """Diff parcial de métricas “interesantes” (evita ruido)."""
    keys = (
        "cache_hits",
        "cache_misses",
        "http_requests",
        "http_failures",
        "throttle_sleeps",
        "rate_limit_hits",
        "rate_limit_sleeps",
        "disabled_switches",
        "cache_store_writes",
        "cache_patch_writes",
        "candidate_search_calls",
    )
    out: dict[str, int] = {}
    for k in keys:
        out[k] = max(0, _metrics_get_int(after, k) - _metrics_get_int(before, k))
    return out


def _log_omdb_metrics(prefix: str, metrics: Mapping[str, object] | None = None) -> None:
    """Log ultra-compacto de métricas (solo SILENT+DEBUG)."""
    if not (SILENT_MODE and DEBUG_MODE):
        return

    m = metrics or get_omdb_metrics_snapshot()
    logger.progress(
        f"{prefix} OMDb metrics: "
        f"cache_hits={_metrics_get_int(m, 'cache_hits')} "
        f"cache_misses={_metrics_get_int(m, 'cache_misses')} "
        f"http_requests={_metrics_get_int(m, 'http_requests')} "
        f"http_failures={_metrics_get_int(m, 'http_failures')} "
        f"throttle_sleeps={_metrics_get_int(m, 'throttle_sleeps')} "
        f"rate_limit_hits={_metrics_get_int(m, 'rate_limit_hits')} "
        f"rate_limit_sleeps={_metrics_get_int(m, 'rate_limit_sleeps')} "
        f"disabled_switches={_metrics_get_int(m, 'disabled_switches')} "
        f"cache_store_writes={_metrics_get_int(m, 'cache_store_writes')} "
        f"cache_patch_writes={_metrics_get_int(m, 'cache_patch_writes')} "
        f"candidate_search_calls={_metrics_get_int(m, 'candidate_search_calls')}"
    )


def _compute_cost_score(delta: Mapping[str, int]) -> int:
    """Heurística de coste para rankings por contenedor."""
    http_requests = int(delta.get("http_requests", 0))
    rate_limit_sleeps = int(delta.get("rate_limit_sleeps", 0))
    http_failures = int(delta.get("http_failures", 0))
    rate_limit_hits = int(delta.get("rate_limit_hits", 0))
    throttle_sleeps = int(delta.get("throttle_sleeps", 0))

    score = 0
    score += http_requests * 1
    score += throttle_sleeps * 3
    score += rate_limit_hits * 10
    score += http_failures * 20
    score += rate_limit_sleeps * 50
    return score


def _rank_top_by_total_cost(deltas_by_group: dict[str, dict[str, int]], top_n: int = 5) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = [(group, _compute_cost_score(d)) for group, d in deltas_by_group.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _log_omdb_rankings(deltas_by_group: dict[str, dict[str, int]], *, min_groups: int = 2) -> None:
    """Rankings compactos para diagnosticar contenedores “caros”."""
    if not (SILENT_MODE and DEBUG_MODE):
        return
    if not deltas_by_group or len(deltas_by_group) < min_groups:
        return

    def _fmt(items: list[tuple[str, int]]) -> str:
        usable = [(name, val) for (name, val) in items if val > 0]
        return " | ".join([f"{i+1}) {name}: {val}" for i, (name, val) in enumerate(usable)])

    if (line := _fmt(_rank_top_by_total_cost(deltas_by_group, top_n=5))):
        logger.progress(f"[DLNA][DEBUG] Top containers by TOTAL_COST: {line}")


# ============================================================================
# WORKERS (cap por OMDb limiter)
# ============================================================================


def _compute_max_workers(requested: int) -> int:
    """Decide el número de workers para ANALYZE con caps defensivos."""
    max_workers = int(requested)
    if max_workers < 1:
        max_workers = 1
    if max_workers > _MAX_WORKERS_CAP:
        max_workers = _MAX_WORKERS_CAP

    # Cap adicional por OMDb (evita miles de hilos inútiles)
    omdb_cap = max(4, int(OMDB_HTTP_MAX_CONCURRENCY) * 8)
    max_workers = min(max_workers, omdb_cap)

    return max(1, max_workers)


def _compute_max_inflight(max_workers: int) -> int:
    """Nº de futures máximos en vuelo (ANALYZE) por contenedor."""
    inflight = max_workers * _DEFAULT_MAX_INFLIGHT_FACTOR
    return max(max_workers, inflight)


def _compute_scan_workers() -> int:
    """Workers para SCAN (Browse DLNA)."""
    try:
        w = int(DLNA_SCAN_WORKERS)
    except Exception:
        w = 2
    return max(1, min(8, w))


# ============================================================================
# FORMATEO (progreso por item)
# ============================================================================


def _format_human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 0

    while value >= 1024.0 and unit_index < (len(units) - 1):
        value /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_item_progress_line(*, index: int, total: int, title: str, year: int | None, file_size_bytes: int | None) -> str:
    base = title.strip() or "UNKNOWN"
    if year is not None:
        base = f"{base} ({year})"
    if DEBUG_MODE and file_size_bytes is not None and file_size_bytes >= 0:
        base = f"{base} [{_format_human_size(file_size_bytes)}]"
    return f"({index}/{total}) {base}"


# ============================================================================
# UTILIDADES GENERALES (DLNA SOAP + resiliencia)
# ============================================================================


def _is_plex_server(device: DLNADevice) -> bool:
    """Heurística: Plex Media Server suele anunciarse así en friendly_name."""
    return "plex media server" in device.friendly_name.lower()


def _xml_text(elem: ET.Element | None) -> str | None:
    """Extrae texto (strip) o None."""
    if elem is None or elem.text is None:
        return None
    val = elem.text.strip()
    return val or None


def _backoff_sleep(attempt: int) -> None:
    """Backoff exponencial con jitter para DLNA cuando no hay backend.resilience."""
    a = max(0, int(attempt))
    delay = min(_DLNA_RETRY_CAP_SECONDS, _DLNA_RETRY_BASE_SECONDS * (2 ** a))
    if _DLNA_RETRY_JITTER > 0:
        delay = max(0.0, delay * (1.0 + random.uniform(-_DLNA_RETRY_JITTER, _DLNA_RETRY_JITTER)))
    time.sleep(delay)


def _dlna_should_retry(exc: BaseException) -> bool:
    """
    Heurística ligera para retry.

    Nota: es deliberadamente permisiva; `DLNA_BROWSE_MAX_RETRIES` y circuit breaker
    actúan como control principal.
    """
    name = exc.__class__.__name__.lower()
    if "timeout" in name or "connection" in name or "remotedisconnected" in name or "protocolerror" in name:
        return True
    if "httperror" in name:
        return True
    return True


def _resilient_call(*, endpoint_key: str, action: str, fn):
    """
    Ejecuta una operación IO DLNA con:
    - circuit breaker si backend.resilience está disponible
    - retry/backoff si falla

    Devuelve el resultado o None.
    """
    t0 = time.monotonic()
    METRICS.incr(f"dlna.{action}.calls")

    if _DLNA_BREAKER is not None and callable(call_with_resilience):
        result, status = call_with_resilience(
            breaker=_DLNA_BREAKER,
            key=endpoint_key,
            fn=fn,
            should_retry=_dlna_should_retry,
            max_retries=int(DLNA_BROWSE_MAX_RETRIES),
        )
        METRICS.observe_ms(f"dlna.{action}.latency_ms", (time.monotonic() - t0) * 1000.0)

        if status == "ok":
            return result

        if status.startswith("circuit_open"):
            METRICS.incr(f"dlna.{action}.blocked_by_circuit")
            METRICS.add_error("dlna", action, endpoint=endpoint_key, detail=status)
            logger.warning(f"[DLNA] Circuit OPEN para {endpoint_key} ({action}) -> se omite temporalmente.", always=True)
            return None

        METRICS.incr(f"dlna.{action}.errors")
        METRICS.add_error("dlna", action, endpoint=endpoint_key, detail=status)
        return None

    # Fallback retry simple
    last_exc: BaseException | None = None
    for attempt in range(0, max(0, int(DLNA_BROWSE_MAX_RETRIES)) + 1):
        try:
            out = fn()
            METRICS.observe_ms(f"dlna.{action}.latency_ms", (time.monotonic() - t0) * 1000.0)
            return out
        except BaseException as exc:
            last_exc = exc
            METRICS.incr(f"dlna.{action}.errors")
            METRICS.add_error("dlna", action, endpoint=endpoint_key, detail=repr(exc))
            if attempt >= int(DLNA_BROWSE_MAX_RETRIES):
                break
            _backoff_sleep(attempt)

    METRICS.observe_ms(f"dlna.{action}.latency_ms", (time.monotonic() - t0) * 1000.0)
    if last_exc is not None:
        logger.error(f"[DLNA] Error {action} contra {endpoint_key}: {last_exc!r}", always=True)
    return None


def _fetch_xml_root(url: str, timeout_s: float = 5.0) -> ET.Element | None:
    """Descarga y parsea un XML (device description)."""
    endpoint_key = f"dlna:xml:{url}"

    def _do() -> ET.Element:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        return ET.fromstring(data)

    root = _resilient_call(endpoint_key=endpoint_key, action="xml_fetch", fn=_do)
    if root is None:
        logger.warning(f"[DLNA] No se pudo descargar/parsear XML {url}", always=True)
    return root


# ============================================================================
# ✅ Cache de endpoints ContentDirectory por device (evita recalcular)
# ============================================================================

_ENDPOINTS_CACHE: dict[str, tuple[str, str] | None] = {}


def _find_content_directory_endpoints(device_location: str) -> tuple[str, str] | None:
    """
    Localiza controlURL + serviceType para ContentDirectory.

    Cache:
    - Clave: device_location (LOCATION del device description)
    - Valor: (control_url, service_type) o None (negative cache)

    Esto reduce llamadas repetidas al device description (y evita presión en Plex).
    """
    key = (device_location or "").strip()
    if not key:
        return None

    if key in _ENDPOINTS_CACHE:
        return _ENDPOINTS_CACHE[key]

    root = _fetch_xml_root(device_location)
    if root is None:
        _ENDPOINTS_CACHE[key] = None
        return None

    for service in root.iter():
        if not (isinstance(service.tag, str) and service.tag.endswith("service")):
            continue

        service_type: str | None = None
        control_url: str | None = None

        for child in list(service):
            if not isinstance(child.tag, str):
                continue
            if child.tag.endswith("serviceType"):
                service_type = _xml_text(child)
            elif child.tag.endswith("controlURL"):
                control_url = _xml_text(child)

        if not service_type or not control_url:
            continue
        if "ContentDirectory" not in service_type:
            continue

        resolved = (urljoin(device_location, control_url), service_type)
        _ENDPOINTS_CACHE[key] = resolved
        return resolved

    _ENDPOINTS_CACHE[key] = None
    return None


def _soap_browse_direct_children(
    control_url: str,
    service_type: str,
    object_id: str,
    starting_index: int,
    requested_count: int,
) -> tuple[list[ET.Element], int] | None:
    """
    Ejecuta BrowseDirectChildren y devuelve:
        (children_elements, total_matches)

    Robustez:
    - Resiliente (retry/backoff + circuit breaker) sobre el POST SOAP.
    - DIDL-Lite puede venir escapado dentro de <Result>.
    - TotalMatches puede faltar -> fallback len(children).
    """
    endpoint_key = f"dlna:soap:{control_url}"

    body = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
        "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" "
        "s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">"
        "<s:Body>"
        f"<u:Browse xmlns:u=\"{service_type}\">"
        f"<ObjectID>{object_id}</ObjectID>"
        "<BrowseFlag>BrowseDirectChildren</BrowseFlag>"
        "<Filter>*</Filter>"
        f"<StartingIndex>{starting_index}</StartingIndex>"
        f"<RequestedCount>{requested_count}</RequestedCount>"
        "<SortCriteria></SortCriteria>"
        "</u:Browse>"
        "</s:Body>"
        "</s:Envelope>"
    )

    soap_action = f"\"{service_type}#Browse\""
    headers = {
        "Content-Type": 'text/xml; charset="utf-8"',
        "SOAPAction": soap_action,
        "SOAPACTION": soap_action,
    }

    def _do() -> bytes:
        req = Request(control_url, data=body.encode("utf-8"), headers=headers, method="POST")
        with urlopen(req, timeout=10) as resp:
            return resp.read()

    raw = _resilient_call(endpoint_key=endpoint_key, action="browse", fn=_do)
    if raw is None:
        return None

    try:
        envelope = ET.fromstring(raw)
    except Exception as exc:
        METRICS.incr("dlna.browse.parse_errors")
        METRICS.add_error("dlna", "browse_parse", endpoint=control_url, detail=repr(exc))
        logger.error(f"[DLNA] Respuesta SOAP inválida desde {control_url}: {exc!r}", always=True)
        return None

    result_text: str | None = None
    total_matches: int | None = None

    for elem in envelope.iter():
        if not isinstance(elem.tag, str):
            continue
        if elem.tag.endswith("Result"):
            result_text = _xml_text(elem)
        elif elem.tag.endswith("TotalMatches"):
            tm = _xml_text(elem)
            if tm and tm.isdigit():
                total_matches = int(tm)

    if result_text is None:
        METRICS.incr("dlna.browse.missing_result")
        return None

    try:
        didl = ET.fromstring(result_text)
    except Exception:
        # Caso típico DLNA: entidades escapadas dentro de <Result>
        unescaped = (
            result_text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
            .replace("&amp;", "&")
        )
        try:
            didl = ET.fromstring(unescaped)
        except Exception as exc:
            METRICS.incr("dlna.browse.didl_parse_errors")
            METRICS.add_error("dlna", "didl_parse", endpoint=control_url, detail=repr(exc))
            logger.error(f"[DLNA] No se pudo parsear DIDL-Lite desde {control_url}: {exc!r}", always=True)
            return None

    children = list(didl)
    return children, (total_matches or len(children))


# ============================================================================
# Discovery + navegación contenedores
# ============================================================================


def _extract_container_title_and_id(container: ET.Element) -> _DlnaContainer | None:
    """Extrae id + dc:title (o similar) de un <container>."""
    obj_id = container.attrib.get("id")
    if not obj_id:
        return None

    title: str | None = None
    for ch in list(container):
        if isinstance(ch.tag, str) and ch.tag.endswith("title"):
            title = _xml_text(ch)
            break

    if not title:
        return None

    return _DlnaContainer(object_id=obj_id, title=title)


def _is_likely_video_root_title(title: str) -> bool:
    """Heurística para detectar raíces de vídeo."""
    t = title.strip().lower()
    if not t:
        return False

    negative = (
        "music", "música", "audio",
        "photo", "photos", "foto", "fotos",
        "picture", "pictures", "imagen", "imágenes",
    )
    if any(n in t for n in negative):
        return False

    positive = ("video", "vídeo", "videos", "vídeos")
    return any(p in t for p in positive)


def _folder_browse_container_score(title: str) -> int:
    """Score para detectar “Browse by folder / Folders / Carpetas”."""
    t = title.strip().lower()
    if not t:
        return 0
    strong = ("by folder", "browse folders", "examinar carpetas", "carpetas", "por carpeta", "folders")
    for s in strong:
        if t == s or s in t:
            return 100
    return 0


def _is_plex_virtual_container_title(title: str) -> bool:
    """Plex expone “vistas virtuales” por DLNA; no queremos analizarlas."""
    t = title.strip().lower()
    if not t:
        return True

    plex_virtual_tokens = (
        "video channels", "channels", "shared video", "remote video",
        "watch later", "recommended", "preferences", "continue watching",
        "recently viewed", "recently added", "recently released",
        "by collection", "by edition", "by genre", "by year", "by decade",
        "by director", "by starring actor", "by country", "by content rating",
        "by rating", "by resolution", "by first letter",
    )
    return any(tok in t for tok in plex_virtual_tokens)


def _list_root_containers(device: DLNADevice) -> tuple[list[_DlnaContainer], tuple[str, str] | None]:
    """Lista contenedores bajo el ObjectID raíz "0"."""
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        logger.error(f"[DLNA] El dispositivo '{device.friendly_name}' no expone ContentDirectory.", always=True)
        return [], None

    control_url, service_type = endpoints
    root_children = _soap_browse_direct_children(control_url, service_type, "0", 0, 500)
    if root_children is None:
        METRICS.incr("dlna.scan.errors")
        return [], endpoints

    children, _ = root_children
    containers: list[_DlnaContainer] = []

    for elem in children:
        if not (isinstance(elem.tag, str) and elem.tag.endswith("container")):
            continue
        c = _extract_container_title_and_id(elem)
        if c is not None:
            containers.append(c)

    return containers, endpoints


def _list_video_root_containers(device: DLNADevice) -> list[_DlnaContainer]:
    """Filtra raíces que parecen de vídeo."""
    containers, _ = _list_root_containers(device)
    return [c for c in containers if _is_likely_video_root_title(c.title)]


def _list_child_containers(device: DLNADevice, parent_object_id: str) -> list[_DlnaContainer]:
    """Lista contenedores hijo de un parent ObjectID."""
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        return []

    control_url, service_type = endpoints
    children_resp = _soap_browse_direct_children(control_url, service_type, parent_object_id, 0, 500)
    if children_resp is None:
        METRICS.incr("dlna.scan.errors")
        return []

    children, _ = children_resp
    out: list[_DlnaContainer] = []

    for elem in children:
        if not (isinstance(elem.tag, str) and elem.tag.endswith("container")):
            continue
        c = _extract_container_title_and_id(elem)
        if c is not None:
            out.append(c)

    return out


def _auto_descend_folder_browse(device: DLNADevice, container: _DlnaContainer) -> _DlnaContainer:
    """
    Baja automáticamente hasta 3 niveles si detecta una carpeta “fuerte” tipo “Browse by folder”.

    Por qué existe:
    - Algunos servers exponen: Video -> (Browse by folder) -> (Folders) -> ...
    - Para el usuario, es más útil que “Video” apunte ya a la estructura de carpetas.
    """
    current = container
    for _ in range(3):
        children = _list_child_containers(device, current.object_id)
        if not children:
            return current

        best: _DlnaContainer | None = None
        best_score = 0
        for c in children:
            score = _folder_browse_container_score(c.title)
            if score > best_score:
                best_score = score
                best = c

        if best is None or best_score <= 0:
            return current
        current = best

    return current


# ============================================================================
# INTERACCIÓN (CLI)
# ============================================================================


def _ask_dlna_device() -> DLNADevice | None:
    """Selección interactiva de servidor (CLI)."""
    logger.info("\nBuscando servidores DLNA/UPnP en la red...\n", always=True)
    devices = discover_dlna_devices()

    if not devices:
        logger.error("[DLNA] No se han encontrado servidores DLNA/UPnP.", always=True)
        return None

    logger.info("Se han encontrado los siguientes servidores DLNA/UPnP:\n", always=True)
    for idx, dev in enumerate(devices, start=1):
        logger.info(f"  {idx}) {dev.friendly_name} ({dev.host}:{dev.port})", always=True)
        logger.info(f"      LOCATION: {dev.location}", always=True)

    while True:
        raw = input(f"\nSelecciona un servidor (1-{len(devices)}) o pulsa Enter para cancelar: ").strip()
        if raw == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if not raw.isdigit():
            logger.warning("Opción no válida. Debe ser un número (o Enter para cancelar).", always=True)
            continue
        num = int(raw)
        if not (1 <= num <= len(devices)):
            logger.warning("Opción fuera de rango.", always=True)
            continue
        chosen = devices[num - 1]
        logger.info(f"\nHas seleccionado: {chosen.friendly_name} ({chosen.host}:{chosen.port})\n", always=True)
        return chosen


def _parse_multi_selection(raw: str, max_value: int) -> list[int] | None:
    """Parse de '1,2,3' con validación y dedupe conservador."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None

    values: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        val = int(part)
        if not (1 <= val <= max_value):
            return None
        values.append(val)

    seen: set[int] = set()
    unique: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _select_folders_non_plex(base: _DlnaContainer, device: DLNADevice) -> list[_DlnaContainer] | None:
    """Selector CLI para servidores DLNA genéricos."""
    logger.info("\nMenú (Enter cancela):", always=True)
    logger.info("  0) Todas las carpetas de vídeo de DLNA", always=True)
    logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    while True:
        raw = input("Selecciona una opción (0/1) o pulsa Enter para cancelar: ").strip()
        if raw == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if raw not in ("0", "1"):
            logger.warning("Opción no válida. Introduce 0 o 1 (o Enter para cancelar).", always=True)
            continue

        if raw == "0":
            return [base]

        folders = _list_child_containers(device, base.object_id)
        if not folders:
            logger.error("[DLNA] No se han encontrado carpetas dentro del contenedor seleccionado.", always=True)
            return None

        logger.info("\nCarpetas detectadas (Enter cancela):", always=True)
        for idx, c in enumerate(folders, start=1):
            logger.info(f"  {idx}) {c.title}", always=True)

        raw_sel = input("Selecciona carpetas separadas por comas (ej: 1,2) o pulsa Enter para cancelar: ").strip()
        if raw_sel == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None

        selected = _parse_multi_selection(raw_sel, len(folders))
        if selected is None:
            logger.warning(
                f"Selección no válida. Usa números 1-{len(folders)} separados por comas (ej: 1,2).",
                always=True,
            )
            continue

        return [folders[i - 1] for i in selected]


def _select_folders_plex(base: _DlnaContainer, device: DLNADevice) -> list[_DlnaContainer] | None:
    """Selector CLI para Plex vía DLNA: filtra vistas virtuales."""
    logger.info("\nOpciones Plex (Enter cancela):", always=True)
    logger.info("  0) Todas las carpetas de vídeo de Plex Media Server", always=True)
    logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    while True:
        raw = input("Selecciona una opción (0/1) o pulsa Enter para cancelar: ").strip()
        if raw == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if raw not in ("0", "1"):
            logger.warning("Opción no válida. Introduce 0 o 1 (o Enter para cancelar).", always=True)
            continue

        if raw == "0":
            return [base]

        folders = _list_child_containers(device, base.object_id)
        folders = [c for c in folders if not _is_plex_virtual_container_title(c.title)]
        if not folders:
            logger.error(
                "[DLNA] No se han encontrado carpetas Plex seleccionables (tras filtrar vistas/servicios).",
                always=True,
            )
            return None

        logger.info("\nCarpetas detectadas en Plex (Enter cancela):", always=True)
        for idx, c in enumerate(folders, start=1):
            logger.info(f"  {idx}) {c.title}", always=True)

        raw_sel = input("Selecciona carpetas separadas por comas (ej: 1,2) o pulsa Enter para cancelar: ").strip()
        if raw_sel == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None

        selected = _parse_multi_selection(raw_sel, len(folders))
        if selected is None:
            logger.warning(
                f"Selección no válida. Usa números 1-{len(folders)} separados por comas (ej: 1,2).",
                always=True,
            )
            continue

        return [folders[i - 1] for i in selected]


# ============================================================================
# EXTRACCIÓN DE ITEMS DE VÍDEO (DIDL-Lite)
# ============================================================================


def _is_video_item(elem: ET.Element) -> bool:
    """Heurística: upnp:class o protocolInfo que sugiera vídeo."""
    upnp_class: str | None = None
    protocol_info: str | None = None

    for ch in list(elem):
        if not isinstance(ch.tag, str):
            continue
        if ch.tag.endswith("class"):
            upnp_class = _xml_text(ch)
        elif ch.tag.endswith("res"):
            protocol_info = ch.attrib.get("protocolInfo")

    if upnp_class and "videoItem" in upnp_class:
        return True
    if protocol_info and ":video" in protocol_info:
        return True
    return False


def _safe_parse_int(value: object) -> int | None:
    """Parse defensivo para size bytes."""
    try:
        if value is None:
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        s = str(value).strip()
        if not s or not s.isdigit():
            return None
        v = int(s)
        return v if v >= 0 else None
    except Exception:
        return None


def _normalize_resource_url(url: str) -> str:
    """
    Normaliza una resource URL para dedupe.

    Principio:
    - No tocamos query/params (en Plex puede haber tokens importantes).
    - Solo limpiamos whitespace y normalizamos a "string limpia".
    """
    return (url or "").strip()


def _extract_video_item(elem: ET.Element) -> _DlnaVideoItem | None:
    """Extrae título, resource_url, size y item_id (si existe) de un <item>."""
    title: str | None = None
    resource_url: str | None = None
    size_bytes: int | None = None
    item_id = elem.attrib.get("id") or None

    for ch in list(elem):
        if not isinstance(ch.tag, str):
            continue
        if ch.tag.endswith("title") and title is None:
            title = _xml_text(ch)
        elif ch.tag.endswith("res") and resource_url is None:
            resource_url = _xml_text(ch)
            size_bytes = _safe_parse_int(ch.attrib.get("size"))

    if not title or not resource_url:
        return None

    resource_url = _normalize_resource_url(resource_url)
    if not resource_url:
        return None

    return _DlnaVideoItem(title=title, resource_url=resource_url, size_bytes=size_bytes, item_id=item_id)


def _build_traversal_limits() -> _TraversalLimits:
    """
    Construye límites de traversal desde config y aplica clamps locales ultra defensivos.

    Nota:
    - config.py ya hace caps; aquí solo blindamos valores absurdos si alguien rompió config
      o si este módulo se reutiliza fuera del flujo normal.
    """
    max_depth = max(1, int(DLNA_TRAVERSE_MAX_DEPTH))
    max_containers = max(100, int(DLNA_TRAVERSE_MAX_CONTAINERS))
    max_items_total = max(1_000, int(DLNA_TRAVERSE_MAX_ITEMS_TOTAL))
    max_pages_per_container = max(10, int(DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER))
    max_empty_pages = max(1, int(DLNA_TRAVERSE_MAX_EMPTY_PAGES))

    return _TraversalLimits(
        max_depth=max_depth,
        max_containers=max_containers,
        max_items_total=max_items_total,
        max_pages_per_container=max_pages_per_container,
        max_empty_pages=max_empty_pages,
    )


def _iter_video_items_recursive(
    device: DLNADevice,
    root_object_id: str,
    *,
    limits: _TraversalLimits,
) -> tuple[list[_DlnaVideoItem], dict[str, int]]:
    """
    Recorre recursivamente un árbol de contenedores DLNA y devuelve items de vídeo.

    ✅ FIXES IMPORTANTES PARA PLEX/UPNP:
    - visited_containers evita ciclos (bucle infinito)
    - dedupe local de items evita repeticiones dentro del mismo traversal
    - fuses de seguridad evitan loops por paginación inconsistente

    Returns:
        (items, stats)
        stats es un dict pequeño para debug/telemetría (no se imprime salvo DEBUG).
    """
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        METRICS.incr("dlna.scan.errors")
        return [], {"containers_seen": 0, "cycle_skips": 0, "items": 0, "dedup_skips_local": 0}

    control_url, service_type = endpoints

    results: list[_DlnaVideoItem] = []

    # visited containers evita ciclos (CRÍTICO)
    visited_containers: set[str] = set()

    # dedupe LOCAL evita repetición por paginación/vistas dentro del mismo traversal
    visited_item_urls_local: set[str] = set()
    visited_item_ids_local: set[str] = set()

    # stack de (object_id, depth)
    stack: list[tuple[str, int]] = [(root_object_id, 0)]

    # Métricas / diagnóstico
    containers_seen = 0
    cycle_skips = 0
    dedup_skips_local = 0

    while stack:
        current_id, depth = stack.pop()

        if depth > limits.max_depth:
            logger.warning(
                f"[DLNA] Traverse depth cap alcanzado (max_depth={limits.max_depth}). "
                f"Se corta para evitar loops. root={root_object_id!r}",
                always=True,
            )
            break

        if current_id in visited_containers:
            cycle_skips += 1
            continue

        visited_containers.add(current_id)
        containers_seen += 1

        if containers_seen > limits.max_containers:
            logger.warning(
                f"[DLNA] Traverse container cap alcanzado (max_containers={limits.max_containers}). "
                "Se corta para evitar bucles/catálogos virtuales enormes.",
                always=True,
            )
            break

        start = 0
        total = 1

        pages = 0
        empty_pages = 0

        while start < total:
            pages += 1
            if pages > limits.max_pages_per_container:
                logger.warning(
                    f"[DLNA] Max pages por contenedor alcanzado (max_pages_per_container={limits.max_pages_per_container}). "
                    f"container_id={current_id!r}. Se corta este contenedor.",
                    always=True,
                )
                break

            browse = _soap_browse_direct_children(control_url, service_type, current_id, start, _BROWSE_PAGE_SIZE)
            if browse is None:
                METRICS.incr("dlna.scan.errors")
                break

            children, total_matches = browse
            total = max(total, int(total_matches))

            if not children:
                empty_pages += 1
                if empty_pages >= limits.max_empty_pages:
                    logger.warning(
                        f"[DLNA] Contenedor devuelve páginas vacías repetidas (max_empty_pages={limits.max_empty_pages}). "
                        f"container_id={current_id!r}. Se corta este contenedor.",
                        always=True,
                    )
                    break
            else:
                empty_pages = 0

            for elem in children:
                if not isinstance(elem.tag, str):
                    continue

                if elem.tag.endswith("container"):
                    cid = elem.attrib.get("id")
                    if cid and cid not in visited_containers:
                        stack.append((cid, depth + 1))
                    continue

                if not elem.tag.endswith("item"):
                    continue

                if not _is_video_item(elem):
                    continue

                item = _extract_video_item(elem)
                if item is None:
                    continue

                # ✅ Dedupe LOCAL por DIDL item_id si existe
                if item.item_id and item.item_id in visited_item_ids_local:
                    dedup_skips_local += 1
                    continue

                # ✅ Dedupe LOCAL por resource_url
                if item.resource_url in visited_item_urls_local:
                    dedup_skips_local += 1
                    continue

                if item.item_id:
                    visited_item_ids_local.add(item.item_id)
                visited_item_urls_local.add(item.resource_url)

                results.append(item)
                if len(results) >= limits.max_items_total:
                    logger.warning(
                        f"[DLNA] Max items total alcanzado (max_items_total={limits.max_items_total}). "
                        "Se corta para evitar runs infinitos/catálogos gigantes.",
                        always=True,
                    )
                    break

            if len(results) >= limits.max_items_total:
                break

            start += _BROWSE_PAGE_SIZE

        if len(results) >= limits.max_items_total:
            break

    stats = {
        "containers_seen": containers_seen,
        "cycle_skips": cycle_skips,
        "items": len(results),
        "dedup_skips_local": dedup_skips_local,
    }
    if DEBUG_MODE:
        logger.debug_ctx("DLNA", f"traverse_done: {stats} root={root_object_id!r}")

    return results, stats


# ============================================================================
# NORMALIZACIÓN PARA PIPELINE (título / file)
# ============================================================================


def _extract_year_from_title(title: str) -> tuple[str, int | None]:
    """Parsea sufijo “(YYYY)” al final si existe: 'Alien (1979)' -> ('Alien', 1979)."""
    raw = title.strip()
    if not raw:
        return title, None

    match = _TITLE_YEAR_SUFFIX_RE.match(raw)
    if not match:
        return title, None

    year_str = match.group("year")
    if not year_str.isdigit():
        return title, None

    year = int(year_str)
    if not (1900 <= year <= 2100):
        return title, None

    base = match.group("base").strip()
    base = base.rstrip(".").strip()
    return (base or title), year


def _extract_ext_from_resource_url(resource_url: str) -> str:
    """Intenta extraer extensión del último path segment del URL."""
    try:
        parsed = urlparse(resource_url)
        filename = parsed.path.rsplit("/", 1)[-1].strip()
        filename = unquote(filename)
        if "." not in filename:
            return ""
        ext = f".{filename.rsplit('.', 1)[-1].strip()}"
        if ext == ".":
            return ""
        if len(ext) > 8:
            return ""
        if not ext[1:].isalnum():
            return ""
        return ext
    except Exception:
        return ""


def _dlna_display_file(library: str, raw_title: str, resource_url: str) -> tuple[str, str]:
    """Genera un `file` friendly y devuelve también la URL del recurso."""
    ext = _extract_ext_from_resource_url(resource_url)
    base = raw_title.strip() or "UNKNOWN"

    if ext and not base.lower().endswith(ext.lower()):
        base = f"{base}{ext}"

    return f"{library}/{base}", resource_url


# ============================================================================
# ORQUESTACIÓN PRINCIPAL
# ============================================================================


def analyze_dlna_server(device: DLNADevice | None = None) -> None:
    """
    Entry-point principal.

    Flujo:
    - Descubre/selecciona un servidor DLNA.
    - Selecciona contenedores.
    - SCAN por contenedor (controlado por DLNA_SCAN_WORKERS).
    - ANALYZE por item (controlado por PLEX_ANALYZE_WORKERS + cap OMDb).
    - Escribe CSVs en streaming + filtered al final.
    - Flush de caches UNA vez al final (finally).

    ✅ Mejoras clave en esta versión:
    - Traversal con fuses desde config.py (DLNA_TRAVERSE_*)
    - Dedupe GLOBAL por run: evita análisis duplicado entre carpetas/vistas (Plex)
    - Scan concurrente en SILENT procesado "as completed" (menos latencia percibida)
    """
    t0 = time.monotonic()
    reset_omdb_metrics()

    # Dedupe global entre contenedores seleccionados (crítico en Plex)
    global_dedupe = _GlobalDedupeState(seen_item_urls=set(), seen_item_ids=set())

    # Limits (una vez por run)
    limits = _build_traversal_limits()

    try:
        if device is None:
            device = _ask_dlna_device()
            if device is None:
                return

        server_label = f"{device.friendly_name} ({device.host}:{device.port})"
        logger.progress(f"[DLNA] Servidor: {server_label}")
        logger.debug_ctx("DLNA", f"location={device.location!r}")

        # Mostrar límites de traversal en DEBUG (útil cuando hay loops/catálogos enormes)
        if DEBUG_MODE:
            logger.debug_ctx(
                "DLNA",
                "traverse_limits: "
                f"max_depth={limits.max_depth} max_containers={limits.max_containers} "
                f"max_items_total={limits.max_items_total} max_empty_pages={limits.max_empty_pages} "
                f"max_pages_per_container={limits.max_pages_per_container} page_size={_BROWSE_PAGE_SIZE}",
            )

        roots = _list_video_root_containers(device)
        if not roots:
            logger.error("[DLNA] No se han encontrado contenedores raíz de vídeo.", always=True)
            return

        if len(roots) == 1:
            chosen_root = roots[0]
        else:
            logger.info("\nDirectorios raíz de vídeo (Enter cancela):", always=True)
            for idx, c in enumerate(roots, start=1):
                logger.info(f"  {idx}) {c.title}", always=True)

            while True:
                raw = input(f"Selecciona un directorio de vídeo (1-{len(roots)}) o pulsa Enter para cancelar: ").strip()
                if raw == "":
                    logger.info("[DLNA] Operación cancelada.", always=True)
                    return
                if not raw.isdigit():
                    logger.warning("Opción no válida. Debe ser un número.", always=True)
                    continue
                n = int(raw)
                if not (1 <= n <= len(roots)):
                    logger.warning("Opción fuera de rango.", always=True)
                    continue
                chosen_root = roots[n - 1]
                break

        logger.progress(f"[DLNA] Raíz de vídeo: {chosen_root.title}")

        base = _auto_descend_folder_browse(device, chosen_root)
        if base.object_id != chosen_root.object_id:
            logger.debug_ctx("DLNA", f"auto_descend: {chosen_root.title!r} -> {base.title!r}")

        if _is_plex_server(device):
            selected_containers = _select_folders_plex(base, device)
        else:
            selected_containers = _select_folders_non_plex(base, device)

        if selected_containers is None:
            return

        if SILENT_MODE:
            titles = [c.title for c in selected_containers]
            logger.progress(f"[DLNA] Contenedores seleccionados: {len(titles)}")
            if DEBUG_MODE:
                logger.progress("[DLNA][DEBUG] " + " | ".join(titles))

        scan_workers = _compute_scan_workers()
        analyze_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS)
        max_inflight = _compute_max_inflight(analyze_workers)

        if SILENT_MODE:
            logger.progress(
                f"[DLNA] Concurrency: scan_workers={scan_workers} analyze_workers={analyze_workers} inflight_cap={max_inflight} | "
                f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
                f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
            )
        else:
            logger.debug_ctx(
                "DLNA",
                f"Concurrency: scan_workers={scan_workers} analyze_workers={analyze_workers} inflight_cap={max_inflight} "
                f"(cap por OMDb limiter, OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})",
            )

        filtered_rows: list[_Row] = []
        decisions_count: dict[str, int] = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}
        total_items_processed = 0
        total_items_errors = 0
        total_rows_written = 0
        total_suggestions_written = 0

        container_omdb_delta_analyze: dict[str, dict[str, int]] = {}

        analyze_snapshot_start: dict[str, object] | None = None
        if SILENT_MODE and DEBUG_MODE:
            analyze_snapshot_start = dict(get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: start:")

        def _maybe_print_item_logs(logs: list[str]) -> None:
            """Imprime logs por item respetando SILENT/DEBUG."""
            if not logs:
                return
            if not SILENT_MODE:
                for line in logs:
                    logger.info(line)
                return
            if DEBUG_MODE:
                for line in logs:
                    logger.info(line, always=True)

        def _tally_decision(row: Mapping[str, object]) -> None:
            """Acumula conteo de decisiones para resumen final."""
            d = row.get("decision")
            if d in ("KEEP", "MAYBE", "DELETE"):
                decisions_count[str(d)] += 1
            else:
                decisions_count["UNKNOWN"] += 1

        def _analyze_one(
            *,
            raw_title: str,
            resource_url: str,
            file_size: int | None,
            library: str,
        ) -> tuple[_Row | None, _Row | None, list[str]]:
            """Normaliza un item DLNA -> MovieInput y ejecuta analyze_movie()."""
            clean_title, extracted_year = _extract_year_from_title(raw_title)
            display_file, file_url = _dlna_display_file(library, raw_title, resource_url)

            movie_input = MovieInput(
                source="dlna",
                library=library,
                title=clean_title,
                year=extracted_year,
                file_path=display_file,
                file_size_bytes=file_size,
                imdb_id_hint=None,
                plex_guid=None,
                rating_key=None,
                thumb_url=None,
                extra={"source_url": file_url},
            )

            row, meta_sugg, logs = analyze_movie(movie_input, source_movie=None)

            if row:
                row["file"] = display_file
                row["file_url"] = file_url

            return row, meta_sugg, logs

        def _handle_result(
            res: tuple[_Row | None, _Row | None, list[str]],
            *,
            all_writer: _WAll,
            sugg_writer: _WSugg,
        ) -> None:
            nonlocal total_rows_written, total_suggestions_written

            row, meta_sugg, logs = res
            _maybe_print_item_logs(logs)

            if row:
                all_writer.write_row(row)
                total_rows_written += 1
                _tally_decision(row)
                if row.get("decision") in {"DELETE", "MAYBE"}:
                    filtered_rows.append(dict(row))

            if meta_sugg:
                sugg_writer.write_row(meta_sugg)
                total_suggestions_written += 1

        def _apply_global_dedupe(container_title: str, items: list[_DlnaVideoItem]) -> list[_DlnaVideoItem]:
            """
            Aplica dedupe GLOBAL por run sobre una lista de items escaneados.

            Por qué aquí (y no solo en traversal):
            - El traversal dedupe local evita repetición dentro del mismo root.
            - Plex suele duplicar los mismos items entre diferentes carpetas/vistas;
              esto solo se evita con un set global.
            """
            if not items:
                return items

            out: list[_DlnaVideoItem] = []
            for it in items:
                # Preferimos id si existe (más “semántico”), pero resource_url es clave principal para “mismo fichero”.
                if it.item_id and it.item_id in global_dedupe.seen_item_ids:
                    global_dedupe.skipped_global += 1
                    continue
                if it.resource_url in global_dedupe.seen_item_urls:
                    global_dedupe.skipped_global += 1
                    continue

                if it.item_id:
                    global_dedupe.seen_item_ids.add(it.item_id)
                global_dedupe.seen_item_urls.add(it.resource_url)
                out.append(it)

            if DEBUG_MODE and (len(out) != len(items)):
                logger.debug_ctx(
                    "DLNA",
                    f"global_dedupe: container={container_title!r} in={len(items)} out={len(out)} skipped={len(items)-len(out)}",
                )
            return out

        # =========================================================================
        # Writers globales (streaming)
        # =========================================================================
        with open_all_csv_writer(REPORT_ALL_PATH) as all_writer, open_suggestions_csv_writer(METADATA_FIX_PATH) as sugg_writer:

            def _scan_container(c: _DlnaContainer) -> tuple[str, list[_DlnaVideoItem]]:
                """
                Escanea un contenedor (raíz seleccionada) y devuelve (título, items).

                - Incluye métricas de latencia.
                - El dedupe GLOBAL se aplica fuera (para poder contar y loguear por run).
                """
                METRICS.incr("dlna.scan.containers")
                t_scan0 = time.monotonic()
                items, _stats = _iter_video_items_recursive(device, c.object_id, limits=limits)
                METRICS.observe_ms("dlna.scan.container_latency_ms", (time.monotonic() - t_scan0) * 1000.0)
                return c.title, items

            if not SILENT_MODE:
                # ---------------- NO SILENT (UX estable) ----------------
                candidates_by_container: dict[str, list[tuple[str, str, int | None, str]]] = {}
                total_candidates = 0

                total_containers = len(selected_containers)
                for idx, container in enumerate(selected_containers, start=1):
                    logger.progress(f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {container.title}")

                    items, _stats = _iter_video_items_recursive(device, container.object_id, limits=limits)
                    items = _apply_global_dedupe(container.title, items)

                    logger.debug_ctx("DLNA", f"scan {container.title!r} items={len(items)}")

                    bucket = candidates_by_container.setdefault(container.title, [])
                    for it in items:
                        bucket.append((it.title, it.resource_url, it.size_bytes, container.title))
                    total_candidates += len(items)

                if total_candidates == 0:
                    logger.progress("[DLNA] No se han encontrado items de vídeo.")
                    return

                logger.progress(f"[DLNA] Candidatos a analizar: {total_candidates}")
                analyzed_so_far = 0

                for container_title, items in candidates_by_container.items():
                    if not items:
                        continue

                    logger.progress(f"[DLNA] Analizando contenedor: {container_title} (items={len(items)})")

                    future_to_index: dict[Future[tuple[_Row | None, _Row | None, list[str]]], int] = {}
                    pending_by_index: dict[int, tuple[_Row | None, _Row | None, list[str]]] = {}
                    next_to_write = 1
                    inflight: set[Future[tuple[_Row | None, _Row | None, list[str]]]] = set()

                    def _drain_completed(*, drain_all: bool) -> None:
                        nonlocal next_to_write, total_items_processed, total_items_errors

                        while inflight:
                            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                            if not done:
                                return

                            for fut in done:
                                inflight.discard(fut)
                                idx_local = future_to_index.get(fut, -1)

                                try:
                                    res = fut.result()
                                except Exception as exc:  # pragma: no cover
                                    total_items_errors += 1
                                    METRICS.incr("dlna.analyze.future_errors")
                                    METRICS.add_error("dlna", "analyze_future", endpoint=None, detail=repr(exc))
                                    logger.error(f"[DLNA] Error analizando item (future): {exc!r}", always=True)
                                    total_items_processed += 1
                                    continue

                                if idx_local >= 1:
                                    pending_by_index[idx_local] = res

                                while next_to_write in pending_by_index:
                                    ready = pending_by_index.pop(next_to_write)
                                    _handle_result(ready, all_writer=all_writer, sugg_writer=sugg_writer)
                                    total_items_processed += 1
                                    next_to_write += 1

                            if not drain_all:
                                return

                    workers_here = min(analyze_workers, max(1, len(items)))
                    inflight_cap_here = min(max_inflight, max(1, len(items)))

                    with ThreadPoolExecutor(max_workers=workers_here) as executor:
                        for item_index, (raw_title, resource_url, file_size, library) in enumerate(items, start=1):
                            analyzed_so_far += 1

                            clean_title_preview, extracted_year_preview = _extract_year_from_title(raw_title)
                            display_title = raw_title if DEBUG_MODE else clean_title_preview

                            logger.info(
                                _format_item_progress_line(
                                    index=analyzed_so_far,
                                    total=total_candidates,
                                    title=display_title,
                                    year=extracted_year_preview,
                                    file_size_bytes=file_size,
                                )
                            )

                            fut = executor.submit(
                                _analyze_one,
                                raw_title=raw_title,
                                resource_url=resource_url,
                                file_size=file_size,
                                library=library,
                            )

                            future_to_index[fut] = item_index
                            inflight.add(fut)

                            if len(inflight) >= inflight_cap_here:
                                _drain_completed(drain_all=False)

                        _drain_completed(drain_all=True)

                    while next_to_write in pending_by_index:
                        ready = pending_by_index.pop(next_to_write)
                        _handle_result(ready, all_writer=all_writer, sugg_writer=sugg_writer)
                        total_items_processed += 1
                        next_to_write += 1

            else:
                # ---------------- SILENT (streaming + scan concurrente) ----------------
                total_containers = len(selected_containers)
                total_candidates = 0
                analyzed_so_far = 0

                with ThreadPoolExecutor(max_workers=scan_workers) as scan_pool, ThreadPoolExecutor(max_workers=analyze_workers) as analyze_pool:
                    scan_futs: list[Future[tuple[str, list[_DlnaVideoItem]]]] = []
                    for idx, c in enumerate(selected_containers, start=1):
                        logger.progress(f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {c.title}")
                        scan_futs.append(scan_pool.submit(_scan_container, c))

                    # ✅ Procesar en cuanto termine cada contenedor (mejor throughput/latencia)
                    for fut in as_completed(scan_futs):
                        try:
                            container_title, items = fut.result()
                        except Exception as exc:  # pragma: no cover
                            METRICS.incr("dlna.scan.errors")
                            METRICS.add_error("dlna", "scan_container", endpoint=None, detail=repr(exc))
                            logger.error(f"[DLNA] Error escaneando contenedor (future): {exc!r}", always=True)
                            continue

                        items = _apply_global_dedupe(container_title, items)

                        container_key = container_title or "<container>"
                        total_candidates += len(items)
                        logger.debug_ctx("DLNA", f"scan {container_title!r} items={len(items)}")

                        if not items:
                            continue

                        logger.progress(f"[DLNA] Analizando contenedor: {container_title} (items={len(items)})")

                        analyze_snap_start_container: dict[str, object] | None = None
                        if DEBUG_MODE:
                            analyze_snap_start_container = dict(get_omdb_metrics_snapshot())
                            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: analyze:start:")

                        inflight: set[Future[tuple[_Row | None, _Row | None, list[str]]]] = set()
                        inflight_cap_here = min(max_inflight, max(1, len(items)))

                        def _drain_completed(*, drain_all: bool) -> None:
                            nonlocal total_items_processed, total_items_errors

                            while inflight:
                                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                                if not done:
                                    return

                                for af in done:
                                    inflight.discard(af)
                                    try:
                                        res = af.result()
                                    except Exception as exc:  # pragma: no cover
                                        total_items_errors += 1
                                        METRICS.incr("dlna.analyze.future_errors")
                                        METRICS.add_error("dlna", "analyze_future", endpoint=None, detail=repr(exc))
                                        logger.error(f"[DLNA] Error analizando item (future): {exc!r}", always=True)
                                        total_items_processed += 1
                                        continue

                                    _handle_result(res, all_writer=all_writer, sugg_writer=sugg_writer)
                                    total_items_processed += 1

                                if not drain_all:
                                    return

                        for it in items:
                            analyzed_so_far += 1
                            if DEBUG_MODE and (analyzed_so_far % _PROGRESS_EVERY_N_ITEMS == 0):
                                logger.progress(f"[DLNA][DEBUG] Progreso: analizados {analyzed_so_far} items...")

                            inflight.add(
                                analyze_pool.submit(
                                    _analyze_one,
                                    raw_title=it.title,
                                    resource_url=it.resource_url,
                                    file_size=it.size_bytes,
                                    library=container_title,
                                )
                            )

                            if len(inflight) >= inflight_cap_here:
                                _drain_completed(drain_all=False)

                        _drain_completed(drain_all=True)

                        if DEBUG_MODE and analyze_snap_start_container is not None:
                            analyze_delta_container = _metrics_diff(analyze_snap_start_container, get_omdb_metrics_snapshot())
                            container_omdb_delta_analyze[container_key] = dict(analyze_delta_container)
                            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: analyze:delta:", metrics=analyze_delta_container)

                if total_candidates == 0 and METRICS.snapshot().get("counters", {}).get("dlna.scan.errors", 0) == 0:
                    logger.progress("[DLNA] No se han encontrado items de vídeo.")
                    return

                logger.progress(f"[DLNA] Candidatos detectados (streaming): {total_candidates}")

        # =========================================================================
        # filtered.csv (solo si hay filas) + resumen final
        # =========================================================================
        filtered_csv_status = "SKIPPED (0 rows)"
        filtered_len = 0

        if filtered_rows:
            filtered = sort_filtered_rows(filtered_rows)
            filtered_len = len(filtered)
            with open_filtered_csv_writer_only_if_rows(REPORT_FILTERED_PATH) as fw:
                for r in filtered:
                    fw.write_row(r)
            filtered_csv_status = f"OK ({filtered_len} rows)"

        elapsed = time.monotonic() - t0

        if SILENT_MODE and DEBUG_MODE and analyze_snapshot_start is not None:
            analyze_delta = _metrics_diff(analyze_snapshot_start, get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: delta:", metrics=analyze_delta)

        snap = METRICS.snapshot()
        counters: Mapping[str, object] = snap.get("counters", {}) if isinstance(snap, Mapping) else {}
        dlna_scan_errors = int(counters.get("dlna.scan.errors", 0) or 0) + int(counters.get("dlna.browse.errors", 0) or 0)
        dlna_circuit_blocks = int(counters.get("dlna.browse.blocked_by_circuit", 0) or 0) + int(counters.get("dlna.xml_fetch.blocked_by_circuit", 0) or 0)

        if SILENT_MODE:
            logger.progress(
                "[DLNA] Resumen final: "
                f"server={server_label} containers={len(selected_containers)} "
                f"scan_workers={_compute_scan_workers()} analyze_workers={_compute_max_workers(PLEX_ANALYZE_WORKERS)} inflight_cap={_compute_max_inflight(_compute_max_workers(PLEX_ANALYZE_WORKERS))} "
                f"time={elapsed:.1f}s | "
                f"scan_errors={dlna_scan_errors} circuit_blocks={dlna_circuit_blocks} analysis_errors={total_items_errors} | "
                f"items={total_items_processed} rows={total_rows_written} "
                f"(KEEP={decisions_count['KEEP']} MAYBE={decisions_count['MAYBE']} "
                f"DELETE={decisions_count['DELETE']} UNKNOWN={decisions_count['UNKNOWN']}) | "
                f"filtered_rows={filtered_len} filtered_csv={filtered_csv_status} "
                f"suggestions={total_suggestions_written}"
            )

            # ✅ Insight específico del bug “análisis múltiple” (solo info útil)
            if global_dedupe.skipped_global > 0:
                logger.progress(f"[DLNA] Dedupe global: evitados {global_dedupe.skipped_global} items duplicados entre contenedores.")

            logger.progress(
                "[DLNA] CSVs: "
                f"all={REPORT_ALL_PATH} | suggestions={METADATA_FIX_PATH} | filtered={REPORT_FILTERED_PATH}"
            )

            _log_omdb_metrics(prefix="[DLNA][DEBUG] Global:")

            if DEBUG_MODE and container_omdb_delta_analyze:
                logger.progress("[DLNA][DEBUG] Rankings (analyze deltas):")
                _log_omdb_rankings(container_omdb_delta_analyze, min_groups=2)

        logger.info("[DLNA] Análisis completado.", always=True)

    finally:
        # ✅ Flush agregado (una vez por run) aunque haya returns o excepciones
        try:
            flush_external_caches()
        except Exception as exc:  # pragma: no cover
            logger.debug_ctx("DLNA", f"flush_external_caches failed: {exc!r}")