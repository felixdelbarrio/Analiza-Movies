from __future__ import annotations

"""
backend/dlna_client.py

Cliente DLNA/UPnP: discovery + navegación + traversal seguro (ContentDirectory).

Por qué existe
--------------
El orquestador DLNA había crecido demasiado mezclando:
- IO de red DLNA (SOAP, device description XML, browse/paginación)
- Lógica de UX (selecciones CLI)
- Traversal anti-loop
- Orquestación de análisis / reporting

Este módulo extrae el “cliente” DLNA para:
- Reutilización y testabilidad (aislar IO DLNA).
- Robustez (anti-loops, fuses, retry/backoff, circuit breaker opcional).
- Rendimiento (cache de endpoints por device.location, early exits, parse tolerante DIDL).

Política de logs (alineada con backend/logger.py)
------------------------------------------------
- Menús/prompt y validación de input: SIEMPRE visibles -> logger.info(..., always=True)
- Hitos de flujo (selecciones, pasos principales): logger.progress(...)
- Debug contextual: logger.debug_ctx("DLNA", ...) cuando DEBUG_MODE=True (el logger decide visibilidad).
- Errores de IO/pipeline DLNA: logger.error(..., always=True)
- Nunca lanzar por logging (best-effort).

Garantías / guardrails
----------------------
- Traversal con visited_containers para evitar ciclos.
- Fuses configurables para evitar catálogos virtuales/paginación infinita:
    * DLNA_TRAVERSE_MAX_DEPTH
    * DLNA_TRAVERSE_MAX_CONTAINERS
    * DLNA_TRAVERSE_MAX_ITEMS_TOTAL
    * DLNA_TRAVERSE_MAX_EMPTY_PAGES
    * DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER
- Parseo DIDL-Lite tolerante (Result escapado).
- Retry/backoff y circuit breaker opcional (backend.resilience si existe).
- No depende de OMDb / scoring / reporting: solo navegación DLNA.

API (alto nivel)
----------------
- DLNAClient.discover() -> list[DLNADevice]
- DLNAClient.ask_user_to_select_device() -> DLNADevice | None

- DLNAClient.list_root_containers(device) -> list[DlnaContainer]
- DLNAClient.list_video_root_containers(device) -> list[DlnaContainer]
- DLNAClient.list_child_containers(device, parent_object_id) -> list[DlnaContainer]
- DLNAClient.auto_descend_folder_browse(device, container, max_levels=3) -> DlnaContainer
- DLNAClient.iter_video_items_recursive(device, root_object_id, limits=...) -> (items, stats)

UX de contenedores (centralizada aquí)
--------------------------------------
- DLNAClient.ask_user_to_select_video_containers(device)
    -> (chosen_video_root, selected_containers) | None

Contadores agregados (para resumen del orquestador)
---------------------------------------------------
Este módulo incrementa contadores agregados para que el resumen en analiza_dlna.py
cuadre sin tener que conocer detalles internos:
- dlna.browse.errors
- dlna.xml_fetch.errors
Además de:
- dlna.{action}.calls / dlna.{action}.errors / dlna.{action}.latency_ms
- dlna.{action}.blocked_by_circuit (si hay circuit breaker)
"""

import random
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Final
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from backend import logger as logger
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.dlna_discovery import DLNADevice, discover_dlna_devices

# ---------------------------------------------------------------------------
# Opcionales: resilience (circuit breaker + retry policies)
# ---------------------------------------------------------------------------
try:
    from backend.resilience import CircuitBreaker, call_with_resilience  # type: ignore
except Exception:  # pragma: no cover
    CircuitBreaker = None  # type: ignore[assignment]
    call_with_resilience = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Opcionales: run_metrics agregadas
# ---------------------------------------------------------------------------
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

    METRICS = _NoopMetrics()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Config best-effort (preferimos backend.config agregador)
# ---------------------------------------------------------------------------
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

try:
    from backend.config import (  # type: ignore
        DLNA_TRAVERSE_MAX_CONTAINERS,
        DLNA_TRAVERSE_MAX_DEPTH,
        DLNA_TRAVERSE_MAX_EMPTY_PAGES,
        DLNA_TRAVERSE_MAX_ITEMS_TOTAL,
        DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER,
    )
except Exception:  # pragma: no cover
    DLNA_TRAVERSE_MAX_DEPTH = 30  # type: ignore
    DLNA_TRAVERSE_MAX_CONTAINERS = 20_000  # type: ignore
    DLNA_TRAVERSE_MAX_ITEMS_TOTAL = 300_000  # type: ignore
    DLNA_TRAVERSE_MAX_EMPTY_PAGES = 3  # type: ignore
    DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER = 20_000  # type: ignore

# ---------------------------------------------------------------------------
# Constantes locales deliberadas (estabilidad con servers DLNA “quirky”)
# ---------------------------------------------------------------------------
_BROWSE_PAGE_SIZE: Final[int] = 200

# Retry backoff defaults (fallback si no hay backend.resilience)
_DLNA_RETRY_BASE_SECONDS: Final[float] = 0.35
_DLNA_RETRY_CAP_SECONDS: Final[float] = 6.0
_DLNA_RETRY_JITTER: Final[float] = 0.35

# HTTP timeouts (deliberados: navegación DLNA debe “fallar rápido”)
_XML_FETCH_TIMEOUT_S: Final[float] = 5.0
_SOAP_TIMEOUT_S: Final[float] = 10.0

# User-Agent: algunos servers DLNA son sensibles a headers muy minimalistas
_USER_AGENT: Final[str] = "AnalizaMovies-DLNAClient/1.0"


# =============================================================================
# Modelos públicos
# =============================================================================

@dataclass(frozen=True, slots=True)
class DLNAEndpoints:
    """Endpoints ContentDirectory necesarios para ejecutar Browse."""
    control_url: str
    service_type: str


@dataclass(frozen=True, slots=True)
class DlnaContainer:
    """Contenedor DLNA (carpeta/categoría) accesible por ObjectID."""
    object_id: str
    title: str


@dataclass(frozen=True, slots=True)
class DlnaVideoItem:
    """
    Item de vídeo DLNA.

    - title: título DIDL-Lite.
    - resource_url: URL del recurso (res).
    - size_bytes: tamaño aproximado si el server lo aporta.
    - item_id: DIDL item@id si existe (útil para dedupe adicional local).
    """
    title: str
    resource_url: str
    size_bytes: int | None
    item_id: str | None


@dataclass(frozen=True, slots=True)
class TraversalLimits:
    """
    Fuses de traversal anti-loop / anti-catálogo virtual infinito.

    Son límites duros; al alcanzarlos, el traversal corta.
    """
    max_depth: int
    max_containers: int
    max_items_total: int
    max_pages_per_container: int
    max_empty_pages: int


# =============================================================================
# Circuit breaker global (por endpoint)
# =============================================================================

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


# =============================================================================
# Logging centralizado (best-effort)
# =============================================================================

def _dbg(msg: object) -> None:
    """
    Debug contextual DLNA. No decide visibilidad; delega en logger.debug_ctx.
    """
    if not DEBUG_MODE:
        return
    try:
        if hasattr(logger, "debug_ctx"):
            logger.debug_ctx("DLNA", msg)  # type: ignore[attr-defined]
        else:
            logger.debug(str(msg))
    except Exception:
        return


# =============================================================================
# Helpers XML / HTTP
# =============================================================================

def _xml_text(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    v = elem.text.strip()
    return v or None


def _backoff_sleep(attempt: int) -> None:
    a = max(0, int(attempt))
    delay = min(_DLNA_RETRY_CAP_SECONDS, _DLNA_RETRY_BASE_SECONDS * (2 ** a))
    if _DLNA_RETRY_JITTER > 0:
        delay = max(0.0, delay * (1.0 + random.uniform(-_DLNA_RETRY_JITTER, _DLNA_RETRY_JITTER)))
    time.sleep(delay)


def _dlna_should_retry(exc: BaseException) -> bool:
    """
    Heurística ligera para retry.
    Deliberadamente permisiva: los fuses + max retries son el control real.
    """
    name = exc.__class__.__name__.lower()
    if "timeout" in name or "connection" in name or "remotedisconnected" in name or "protocolerror" in name:
        return True
    if "httperror" in name:
        return True
    return True


def _incr_aggregate_error_counter(action: str) -> None:
    """
    Contadores agregados (para resumen del orquestador).
    Centralizamos aquí para no duplicar lógica en varios sitios.
    """
    try:
        if action == "browse":
            METRICS.incr("dlna.browse.errors")
        elif action == "xml_fetch":
            METRICS.incr("dlna.xml_fetch.errors")
    except Exception:
        return


def _resilient_call(*, endpoint_key: str, action: str, fn):
    """
    Ejecuta IO DLNA con:
    - circuit breaker si backend.resilience está disponible
    - retry/backoff si no

    Devuelve resultado o None.
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

        if isinstance(status, str) and status.startswith("circuit_open"):
            METRICS.incr(f"dlna.{action}.blocked_by_circuit")
            METRICS.add_error("dlna", action, endpoint=endpoint_key, detail=str(status))
            logger.warning(f"[DLNA] Circuit OPEN para {endpoint_key} ({action}) -> se omite temporalmente.", always=True)
            return None

        METRICS.incr(f"dlna.{action}.errors")
        _incr_aggregate_error_counter(action)
        METRICS.add_error("dlna", action, endpoint=endpoint_key, detail=str(status))
        return None

    last_exc: BaseException | None = None
    for attempt in range(0, max(0, int(DLNA_BROWSE_MAX_RETRIES)) + 1):
        try:
            out = fn()
            METRICS.observe_ms(f"dlna.{action}.latency_ms", (time.monotonic() - t0) * 1000.0)
            return out
        except BaseException as exc:
            last_exc = exc
            METRICS.incr(f"dlna.{action}.errors")
            _incr_aggregate_error_counter(action)
            METRICS.add_error("dlna", action, endpoint=endpoint_key, detail=repr(exc))
            if attempt >= int(DLNA_BROWSE_MAX_RETRIES):
                break
            _backoff_sleep(attempt)

    METRICS.observe_ms(f"dlna.{action}.latency_ms", (time.monotonic() - t0) * 1000.0)
    if last_exc is not None:
        logger.error(f"[DLNA] Error {action} contra {endpoint_key}: {last_exc!r}", always=True)
    return None


def _fetch_xml_root(url: str, timeout_s: float = _XML_FETCH_TIMEOUT_S) -> ET.Element | None:
    endpoint_key = f"dlna:xml:{url}"

    def _do() -> ET.Element:
        req = Request(url, method="GET", headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=float(timeout_s)) as resp:
            data = resp.read()
        return ET.fromstring(data)

    root = _resilient_call(endpoint_key=endpoint_key, action="xml_fetch", fn=_do)
    if root is None:
        # xml_fetch.errors ya se incrementa en _resilient_call
        logger.warning(f"[DLNA] No se pudo descargar/parsear XML {url}", always=True)
    return root


# =============================================================================
# Cache endpoints ContentDirectory (thread-safe)
# =============================================================================

_ENDPOINTS_CACHE: dict[str, DLNAEndpoints | None] = {}
_ENDPOINTS_LOCK: Final[threading.RLock] = threading.RLock()


def clear_endpoints_cache() -> None:
    """Vacía el cache de endpoints (útil en tests o runs largos)."""
    with _ENDPOINTS_LOCK:
        _ENDPOINTS_CACHE.clear()


def _find_content_directory_endpoints(device_location: str) -> DLNAEndpoints | None:
    """
    Localiza controlURL + serviceType para ContentDirectory.

    Cache:
    - Clave: device_location (LOCATION del device description)
    - Valor: DLNAEndpoints o None (negative cache)
    """
    key = (device_location or "").strip()
    if not key:
        return None

    with _ENDPOINTS_LOCK:
        if key in _ENDPOINTS_CACHE:
            return _ENDPOINTS_CACHE[key]

    root = _fetch_xml_root(key)
    if root is None:
        with _ENDPOINTS_LOCK:
            _ENDPOINTS_CACHE[key] = None
        return None

    found: DLNAEndpoints | None = None

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

        found = DLNAEndpoints(control_url=urljoin(key, control_url), service_type=service_type)
        break

    with _ENDPOINTS_LOCK:
        _ENDPOINTS_CACHE[key] = found
    return found


# =============================================================================
# SOAP Browse
# =============================================================================

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
        "User-Agent": _USER_AGENT,
    }

    def _do() -> bytes:
        req = Request(control_url, data=body.encode("utf-8"), headers=headers, method="POST")
        with urlopen(req, timeout=float(_SOAP_TIMEOUT_S)) as resp:
            return resp.read()

    raw = _resilient_call(endpoint_key=endpoint_key, action="browse", fn=_do)
    if raw is None:
        # browse.errors ya se incrementa en _resilient_call
        return None

    try:
        envelope = ET.fromstring(raw)
    except Exception as exc:
        METRICS.incr("dlna.browse.parse_errors")
        METRICS.add_error("dlna", "browse_parse", endpoint=control_url, detail=repr(exc))
        _incr_aggregate_error_counter("browse")
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
        _incr_aggregate_error_counter("browse")
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
            _incr_aggregate_error_counter("browse")
            logger.error(f"[DLNA] No se pudo parsear DIDL-Lite desde {control_url}: {exc!r}", always=True)
            return None

    children = list(didl)
    return children, (total_matches or len(children))


# =============================================================================
# Heurísticas DLNA (Plex / roots / carpetas)
# =============================================================================

def is_plex_server(device: DLNADevice) -> bool:
    """Heurística: Plex Media Server suele anunciarse así en friendly_name."""
    try:
        return "plex media server" in (device.friendly_name or "").lower()
    except Exception:
        return False


def _is_likely_video_root_title(title: str) -> bool:
    """Heurística para detectar raíces de vídeo."""
    t = (title or "").strip().lower()
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
    t = (title or "").strip().lower()
    if not t:
        return 0
    strong = ("by folder", "browse folders", "examinar carpetas", "carpetas", "por carpeta", "folders")
    for s in strong:
        if t == s or s in t:
            return 100
    return 0


def is_plex_virtual_container_title(title: str) -> bool:
    """Plex expone “vistas virtuales” por DLNA; normalmente no son buenas para análisis/borrado."""
    t = (title or "").strip().lower()
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


# =============================================================================
# Extracción DIDL video items
# =============================================================================

def _is_video_item(elem: ET.Element) -> bool:
    """Heurística: upnp:class o protocolInfo sugieren vídeo."""
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
    Normaliza resource URL para dedupe local.
    - No tocamos query/params (Plex puede incluir tokens).
    """
    return (url or "").strip()


def _extract_video_item(elem: ET.Element) -> DlnaVideoItem | None:
    """Extrae (title, resource_url, size, item_id) de un <item>."""
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

    return DlnaVideoItem(title=title, resource_url=resource_url, size_bytes=size_bytes, item_id=item_id)


# =============================================================================
# Limits
# =============================================================================

def build_traversal_limits() -> TraversalLimits:
    """
    Construye límites desde config + clamps ultra defensivos.
    """
    max_depth = max(1, int(DLNA_TRAVERSE_MAX_DEPTH))
    max_containers = max(100, int(DLNA_TRAVERSE_MAX_CONTAINERS))
    max_items_total = max(1_000, int(DLNA_TRAVERSE_MAX_ITEMS_TOTAL))
    max_pages_per_container = max(10, int(DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER))
    max_empty_pages = max(1, int(DLNA_TRAVERSE_MAX_EMPTY_PAGES))

    return TraversalLimits(
        max_depth=max_depth,
        max_containers=max_containers,
        max_items_total=max_items_total,
        max_pages_per_container=max_pages_per_container,
        max_empty_pages=max_empty_pages,
    )


# =============================================================================
# CLI helpers (selección de carpetas / contenedores)
# =============================================================================

def _parse_multi_selection(raw: str, max_value: int) -> list[int] | None:
    """
    Parsea selección múltiple "1,2,5" (1-indexed).
    Devuelve lista unique preservando orden, o None si inválida.
    """
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


def _select_folders_non_plex(client: "DLNAClient", base: DlnaContainer, device: DLNADevice) -> list[DlnaContainer] | None:
    logger.info("\nMenú (Enter cancela):", always=True)
    logger.info("  0) Todas las carpetas de vídeo de DLNA", always=True)
    logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    while True:
        raw = input("Opción (0/1) o Enter cancela: ").strip()
        if raw == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if raw not in ("0", "1"):
            logger.warning("Opción no válida. Introduce 0 o 1 (o Enter para cancelar).", always=True)
            continue

        if raw == "0":
            return [base]

        folders = client.list_child_containers(device, base.object_id)
        if not folders:
            logger.error("[DLNA] No se han encontrado carpetas dentro del contenedor seleccionado.", always=True)
            return None

        logger.info("\nCarpetas detectadas (Enter cancela):", always=True)
        for idx, c in enumerate(folders, start=1):
            logger.info(f"  {idx}) {c.title}", always=True)

        raw_sel = input("Carpetas (ej: 1,2) o Enter cancela: ").strip()
        if raw_sel == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None

        selected = _parse_multi_selection(raw_sel, len(folders))
        if selected is None:
            logger.warning(f"Selección no válida. Usa números 1-{len(folders)} separados por comas.", always=True)
            continue

        return [folders[i - 1] for i in selected]


def _select_folders_plex(client: "DLNAClient", base: DlnaContainer, device: DLNADevice) -> list[DlnaContainer] | None:
    logger.info("\nOpciones Plex (Enter cancela):", always=True)
    logger.info("  0) Todas las carpetas de vídeo de Plex Media Server", always=True)
    logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    while True:
        raw = input("Opción (0/1) o Enter cancela: ").strip()
        if raw == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if raw not in ("0", "1"):
            logger.warning("Opción no válida. Introduce 0 o 1 (o Enter para cancelar).", always=True)
            continue

        if raw == "0":
            return [base]

        folders = client.list_child_containers(device, base.object_id)
        folders = [c for c in folders if not is_plex_virtual_container_title(c.title)]
        if not folders:
            logger.error("[DLNA] No se han encontrado carpetas Plex seleccionables (filtradas).", always=True)
            return None

        logger.info("\nCarpetas detectadas en Plex (Enter cancela):", always=True)
        for idx, c in enumerate(folders, start=1):
            logger.info(f"  {idx}) {c.title}", always=True)

        raw_sel = input("Carpetas (ej: 1,2) o Enter cancela: ").strip()
        if raw_sel == "":
            logger.info("[DLNA] Operación cancelada.", always=True)
            return None

        selected = _parse_multi_selection(raw_sel, len(folders))
        if selected is None:
            logger.warning(f"Selección no válida. Usa números 1-{len(folders)} separados por comas.", always=True)
            continue

        return [folders[i - 1] for i in selected]


# =============================================================================
# Cliente DLNA
# =============================================================================

class DLNAClient:
    """
    Cliente DLNA/UPnP orientado a navegación.

    Nota:
    - Instancia ligera; el cache de endpoints está a nivel de módulo (thread-safe).
    """

    # -----------------------------
    # Discovery / selección
    # -----------------------------

    def discover(self) -> list[DLNADevice]:
        """Descubre servidores DLNA/UPnP en red (wrapping backend.dlna_discovery)."""
        try:
            return discover_dlna_devices()
        except Exception as exc:
            logger.error(f"[DLNA] discover_dlna_devices falló: {exc!r}", always=True)
            return []

    def ask_user_to_select_device(self) -> DLNADevice | None:
        """Selección interactiva CLI de servidor (Enter cancela)."""
        logger.info("\nBuscando servidores DLNA/UPnP en la red...\n", always=True)
        devices = self.discover()

        if not devices:
            logger.error("[DLNA] No se han encontrado servidores DLNA/UPnP.", always=True)
            return None

        logger.info("Servidores DLNA/UPnP encontrados:\n", always=True)
        for idx, dev in enumerate(devices, start=1):
            if SILENT_MODE:
                logger.info(f"  {idx}) {dev.friendly_name}", always=True)
                if DEBUG_MODE:
                    logger.info(f"      {dev.host}:{dev.port}", always=True)
                    logger.info(f"      LOCATION: {dev.location}", always=True)
            else:
                logger.info(f"  {idx}) {dev.friendly_name} ({dev.host}:{dev.port})", always=True)
                logger.info(f"      LOCATION: {dev.location}", always=True)

        while True:
            prompt = (
                f"\nServidor (1-{len(devices)}) o Enter cancela: "
                if SILENT_MODE
                else f"\nSelecciona un servidor (1-{len(devices)}) o pulsa Enter para cancelar: "
            )
            raw = input(prompt).strip()

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
            logger.info(f"\nHas seleccionado: {chosen.friendly_name}\n", always=True)
            if DEBUG_MODE and not SILENT_MODE:
                logger.info(f"    {chosen.host}:{chosen.port}", always=True)
                logger.info(f"    LOCATION: {chosen.location}", always=True)
            return chosen

    def ask_user_to_select_video_containers(self, device: DLNADevice) -> tuple[DlnaContainer, list[DlnaContainer]] | None:
        """
        UX completa de navegación:
        - selecciona root de vídeo si hay varios
        - auto_descend_folder_browse (si aplica)
        - menú Plex/no-Plex y selección final de contenedores
        """
        roots = self.list_video_root_containers(device)
        if not roots:
            logger.error("[DLNA] No se han encontrado contenedores raíz de vídeo.", always=True)
            return None

        if len(roots) == 1:
            chosen_root = roots[0]
        else:
            logger.info("\nDirectorios raíz de vídeo (Enter cancela):", always=True)
            for idx, c in enumerate(roots, start=1):
                logger.info(f"  {idx}) {c.title}", always=True)

            while True:
                raw = input(f"Directorio (1-{len(roots)}) o Enter cancela: ").strip()
                if raw == "":
                    logger.info("[DLNA] Operación cancelada.", always=True)
                    return None
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

        base = self.auto_descend_folder_browse(device, chosen_root)
        if DEBUG_MODE and base.object_id != chosen_root.object_id:
            _dbg(f"auto_descend: {chosen_root.title!r} -> {base.title!r}")

        # Plex vs no-Plex se decide aquí; el orquestador no debería conocerlo.
        selected = _select_folders_plex(self, base, device) if is_plex_server(device) else _select_folders_non_plex(self, base, device)
        if selected is None:
            return None

        if SILENT_MODE:
            logger.progress(f"[DLNA] Contenedores seleccionados: {len(selected)}")
            if DEBUG_MODE and selected:
                logger.progress("[DLNA][DEBUG] " + " | ".join([c.title for c in selected]))

        return chosen_root, selected

    # -----------------------------
    # Contenedores
    # -----------------------------

    def list_root_containers(self, device: DLNADevice) -> list[DlnaContainer]:
        """Lista contenedores bajo ObjectID raíz '0'."""
        endpoints = _find_content_directory_endpoints(device.location)
        if endpoints is None:
            logger.error(f"[DLNA] El dispositivo '{device.friendly_name}' no expone ContentDirectory.", always=True)
            return []

        root_children = _soap_browse_direct_children(endpoints.control_url, endpoints.service_type, "0", 0, 500)
        if root_children is None:
            METRICS.incr("dlna.scan.errors")
            return []

        children, _ = root_children
        out: list[DlnaContainer] = []

        for elem in children:
            if not (isinstance(elem.tag, str) and elem.tag.endswith("container")):
                continue
            c = self._extract_container_title_and_id(elem)
            if c is not None:
                out.append(c)

        return out

    def list_video_root_containers(self, device: DLNADevice) -> list[DlnaContainer]:
        """Filtra raíces que parecen de vídeo."""
        roots = self.list_root_containers(device)
        return [c for c in roots if _is_likely_video_root_title(c.title)]

    def list_child_containers(self, device: DLNADevice, parent_object_id: str) -> list[DlnaContainer]:
        """Lista contenedores hijo de un parent ObjectID."""
        endpoints = _find_content_directory_endpoints(device.location)
        if endpoints is None:
            return []

        children_resp = _soap_browse_direct_children(endpoints.control_url, endpoints.service_type, parent_object_id, 0, 500)
        if children_resp is None:
            METRICS.incr("dlna.scan.errors")
            return []

        children, _ = children_resp
        out: list[DlnaContainer] = []

        for elem in children:
            if not (isinstance(elem.tag, str) and elem.tag.endswith("container")):
                continue
            c = self._extract_container_title_and_id(elem)
            if c is not None:
                out.append(c)

        return out

    def auto_descend_folder_browse(self, device: DLNADevice, container: DlnaContainer, *, max_levels: int = 3) -> DlnaContainer:
        """
        Baja automáticamente hasta `max_levels` niveles si detecta una carpeta “fuerte”
        tipo “Browse by folder”.
        """
        current = container
        for _ in range(max(0, int(max_levels))):
            children = self.list_child_containers(device, current.object_id)
            if not children:
                return current

            best: DlnaContainer | None = None
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

    @staticmethod
    def _extract_container_title_and_id(container: ET.Element) -> DlnaContainer | None:
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

        return DlnaContainer(object_id=obj_id, title=title)

    # -----------------------------
    # Traversal
    # -----------------------------

    def iter_video_items_recursive(
        self,
        device: DLNADevice,
        root_object_id: str,
        *,
        limits: TraversalLimits | None = None,
    ) -> tuple[list[DlnaVideoItem], dict[str, int]]:
        """
        Recorre recursivamente el árbol de contenedores DLNA desde root_object_id y devuelve items de vídeo.

        Returns:
            (items, stats)
            stats: {containers_seen, cycle_skips, items, dedup_skips_local}
        """
        lim = limits or build_traversal_limits()

        endpoints = _find_content_directory_endpoints(device.location)
        if endpoints is None:
            METRICS.incr("dlna.scan.errors")
            return [], {"containers_seen": 0, "cycle_skips": 0, "items": 0, "dedup_skips_local": 0}

        results: list[DlnaVideoItem] = []
        visited_containers: set[str] = set()
        visited_item_urls_local: set[str] = set()
        visited_item_ids_local: set[str] = set()

        stack: list[tuple[str, int]] = [(root_object_id, 0)]

        containers_seen = 0
        cycle_skips = 0
        dedup_skips_local = 0

        while stack:
            current_id, depth = stack.pop()

            if depth > lim.max_depth:
                logger.warning(
                    f"[DLNA] Traverse depth cap alcanzado (max_depth={lim.max_depth}). "
                    f"Se corta para evitar loops. root={root_object_id!r}",
                    always=True,
                )
                break

            if current_id in visited_containers:
                cycle_skips += 1
                continue

            visited_containers.add(current_id)
            containers_seen += 1

            if containers_seen > lim.max_containers:
                logger.warning(
                    f"[DLNA] Traverse container cap alcanzado (max_containers={lim.max_containers}). "
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
                if pages > lim.max_pages_per_container:
                    logger.warning(
                        f"[DLNA] Max pages por contenedor alcanzado (max_pages_per_container={lim.max_pages_per_container}). "
                        f"container_id={current_id!r}. Se corta este contenedor.",
                        always=True,
                    )
                    break

                browse = _soap_browse_direct_children(
                    endpoints.control_url,
                    endpoints.service_type,
                    current_id,
                    start,
                    _BROWSE_PAGE_SIZE,
                )
                if browse is None:
                    METRICS.incr("dlna.scan.errors")
                    break

                children, total_matches = browse
                total = max(total, int(total_matches))

                if not children:
                    empty_pages += 1
                    if empty_pages >= lim.max_empty_pages:
                        logger.warning(
                            f"[DLNA] Contenedor devuelve páginas vacías repetidas (max_empty_pages={lim.max_empty_pages}). "
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

                    if item.item_id and item.item_id in visited_item_ids_local:
                        dedup_skips_local += 1
                        continue
                    if item.resource_url in visited_item_urls_local:
                        dedup_skips_local += 1
                        continue

                    if item.item_id:
                        visited_item_ids_local.add(item.item_id)
                    visited_item_urls_local.add(item.resource_url)

                    results.append(item)
                    if len(results) >= lim.max_items_total:
                        logger.warning(
                            f"[DLNA] Max items total alcanzado (max_items_total={lim.max_items_total}). "
                            "Se corta para evitar runs infinitos/catálogos gigantes.",
                            always=True,
                        )
                        break

                if len(results) >= lim.max_items_total:
                    break

                start += _BROWSE_PAGE_SIZE

            if len(results) >= lim.max_items_total:
                break

        stats = {
            "containers_seen": containers_seen,
            "cycle_skips": cycle_skips,
            "items": len(results),
            "dedup_skips_local": dedup_skips_local,
        }
        _dbg(f"traverse_done: {stats} root={root_object_id!r}")
        return results, stats

    # -----------------------------
    # Utilidades (opcionales)
    # -----------------------------

    @staticmethod
    def extract_ext_from_resource_url(resource_url: str) -> str:
        """Extrae extensión del último segmento path del URL (best-effort)."""
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


__all__ = [
    "DLNAClient",
    "DLNAEndpoints",
    "DlnaContainer",
    "DlnaVideoItem",
    "TraversalLimits",
    "build_traversal_limits",
    "clear_endpoints_cache",
    "is_plex_server",
    "is_plex_virtual_container_title",
]