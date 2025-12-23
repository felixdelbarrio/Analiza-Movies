from __future__ import annotations

"""
backend/collection_analiza_dlna.py  (antes: analiza_dlna.py)

Orquestador principal de análisis DLNA/UPnP (streaming).

✅ Cambio de esta iteración (patch mínimo)
-----------------------------------------
- Este orquestador sigue llamando a `flush_external_caches()` EXACTAMENTE igual que antes
  (una vez al final del run), pero ahora ese flush ya NO depende de ningún wrapper de Wiki
  (p.ej. `get_wiki_client()`), porque `collection_analysis.py` ya llama a wiki/omdb de forma
  directa y expone el flush agregado.

📌 Qué hace este módulo
-----------------------
1) Descubre servidores DLNA/UPnP en la LAN (o recibe uno ya elegido).
2) Localiza el servicio ContentDirectory (SOAP) y navega el árbol de contenedores.
3) Extrae items de vídeo (título, URL recurso, size opcional).
4) Normaliza cada item -> MovieInput y ejecuta el pipeline por-item (collection_analysis.analyze_movie).
5) Escribe CSVs:
   - report_all.csv        (streaming)
   - metadata_fix.csv      (streaming; aquí suele ser None)
   - report_filtered.csv   (al final; solo si hay filas con DELETE/MAYBE)

Optimización clave (performance + memoria)
------------------------------------------
- Bounded in-flight futures:
  No se encolan todos los futures de golpe: se limita el nº en vuelo por contenedor.
  Esto reduce RAM y acelera “time to first output”.

- Orden estable en NO SILENT sin lista gigante:
  Se mantiene el orden de salida dentro de cada contenedor usando:
    pending_by_index + next_to_write
  sin pre-crear una lista enorme del tamaño total.

- Modo SILENT como streaming “real”:
  Se escribe cada resultado en cuanto se completa y se reporta progreso por contenedor.

Robustez DLNA (puntos dolorosos habituales)
-------------------------------------------
- DIDL-Lite escapado dentro de <Result>.
- Namespaces raros o tags con sufijos.
- size ausente o no numérico.
- Algunos servers no devuelven TotalMatches -> usamos len(children) como fallback.

Logs (centralizados)
--------------------
Este módulo NO implementa política propia de logs.
Se apoya en backend/logger.py:

- logger.progress(...)               -> SIEMPRE visible (heartbeat)
- logger.info/warning/error(...)     -> logs por nivel; silent se gestiona en logger.py
- logger.debug_ctx("DLNA", "...")    -> debug contextual (DEBUG_MODE gated)

Métricas OMDb
-------------
En SILENT+DEBUG se emiten snapshots/deltas de métricas de OMDb, y rankings por contenedor
para diagnosticar “coste” (requests, rate-limit, failures, etc.).
"""

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Protocol, TypeVar
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from backend import logger as logger
from backend.collection_analysis import analyze_movie, flush_external_caches
from backend.config import (
    DEBUG_MODE,
    METADATA_FIX_PATH,
    OMDB_HTTP_MAX_CONCURRENCY,
    OMDB_HTTP_MIN_INTERVAL_SECONDS,
    PLEX_ANALYZE_WORKERS,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
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
    Item de vídeo DLNA (título + URL + size opcional).

    - resource_url suele ser HTTP(s) apuntando al recurso.
    - size_bytes puede no existir o venir como string no numérica.
    """

    title: str
    resource_url: str
    size_bytes: int | None


# ============================================================================
# PARSEO DE TÍTULO / AÑO
# ============================================================================

_TITLE_YEAR_SUFFIX_RE: re.Pattern[str] = re.compile(
    r"(?P<base>.*?)"
    r"(?P<sep>\s*\.?\s*)"
    r"\(\s*(?P<year>\d{4})\s*\)\s*$"
)

_PROGRESS_EVERY_N_ITEMS: int = 100
_BROWSE_PAGE_SIZE: int = 200
_MAX_WORKERS_CAP: int = 64

# Bounded inflight
_DEFAULT_MAX_INFLIGHT_FACTOR: int = 4

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
    """
    Log ultra-compacto de métricas.

    Solo se emite en SILENT+DEBUG para no contaminar el modo normal.
    """
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


def _rank_top(deltas_by_group: dict[str, dict[str, int]], key: str, top_n: int = 5) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = [(group, int(d.get(key, 0))) for group, d in deltas_by_group.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _compute_cost_score(delta: Mapping[str, int]) -> int:
    """
    Heurística de coste:
    - requests cuestan poco
    - throttle medio
    - rate-limit y fallos mucho
    - sleeps por rate-limit muchísimo (bloquean el run)
    """
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
    if (line := _fmt(_rank_top(deltas_by_group, "http_requests", top_n=5))):
        logger.progress(f"[DLNA][DEBUG] Top containers by http_requests: {line}")
    if (line := _fmt(_rank_top(deltas_by_group, "rate_limit_sleeps", top_n=5))):
        logger.progress(f"[DLNA][DEBUG] Top containers by rate_limit_sleeps: {line}")
    if (line := _fmt(_rank_top(deltas_by_group, "http_failures", top_n=5))):
        logger.progress(f"[DLNA][DEBUG] Top containers by http_failures: {line}")
    if (line := _fmt(_rank_top(deltas_by_group, "rate_limit_hits", top_n=5))):
        logger.progress(f"[DLNA][DEBUG] Top containers by rate_limit_hits: {line}")


# ============================================================================
# WORKERS (cap por OMDb limiter)
# ============================================================================


def _compute_max_workers(requested: int) -> int:
    """
    Decide el número de workers del ThreadPool.

    Consideraciones DLNA:
    - IO: SOAP browse (scan), analyze_movie (caché/omdb/wiki)
    - CPU: parsing XML + scoring

    Capas:
    1) requested (PLEX_ANALYZE_WORKERS) con clamp [1.._MAX_WORKERS_CAP]
    2) cap adicional por OMDb para evitar miles de hilos inútiles:
       OMDB_HTTP_MAX_CONCURRENCY * 8 (mínimo 4) -> suficientemente holgado para
       que haya trabajo mientras el throttle/semáforo bloquea.
    """
    max_workers = int(requested)
    if max_workers < 1:
        max_workers = 1
    if max_workers > _MAX_WORKERS_CAP:
        max_workers = _MAX_WORKERS_CAP

    omdb_cap = max(4, int(OMDB_HTTP_MAX_CONCURRENCY) * 8)
    max_workers = min(max_workers, omdb_cap)

    return max(1, max_workers)


def _compute_max_inflight(max_workers: int) -> int:
    """
    Nº de futures máximos en vuelo por contenedor.

    - Debe ser >= max_workers para no starve.
    - El factor 4 suele dar buen equilibrio (latencia vs memoria).
    """
    inflight = max_workers * _DEFAULT_MAX_INFLIGHT_FACTOR
    return max(max_workers, inflight)


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
# UTILIDADES GENERALES (DLNA SOAP)
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


def _fetch_xml_root(url: str, timeout_s: float = 5.0) -> ET.Element | None:
    """
    Descarga y parsea un XML (p.ej. device description).

    Degradación suave:
    - cualquier error -> warning always + None
    """
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            data = resp.read()
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[DLNA] No se pudo descargar XML {url}: {exc!r}", always=True)
        return None

    try:
        return ET.fromstring(data)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[DLNA] No se pudo parsear XML {url}: {exc!r}", always=True)
        return None


def _find_content_directory_endpoints(device_location: str) -> tuple[str, str] | None:
    """
    Localiza controlURL + serviceType para ContentDirectory.

    Returns:
        (control_url, service_type) o None si no se encuentra.
    """
    root = _fetch_xml_root(device_location)
    if root is None:
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

        return urljoin(device_location, control_url), service_type

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
    - El DIDL-Lite puede venir escapado dentro de <Result>.
    - TotalMatches puede faltar -> fallback len(children).
    """
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

    req = Request(control_url, data=body.encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            raw = resp.read()
    except Exception as exc:  # pragma: no cover
        logger.error(f"[DLNA] Error SOAP Browse contra {control_url}: {exc!r}")
        return None

    try:
        envelope = ET.fromstring(raw)
    except Exception as exc:  # pragma: no cover
        logger.error(f"[DLNA] Respuesta SOAP inválida: {exc!r}")
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
        return None

    # 1) intenta parsear tal cual
    try:
        didl = ET.fromstring(result_text)
    except Exception:
        # 2) fallback: entidades escapadas
        unescaped = (
            result_text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
            .replace("&amp;", "&")
        )
        try:
            didl = ET.fromstring(unescaped)
        except Exception as exc:  # pragma: no cover
            logger.error(f"[DLNA] No se pudo parsear DIDL-Lite: {exc!r}")
            return None

    children = list(didl)
    return children, (total_matches or len(children))


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
    """
    Heurística simple para detectar raíces de vídeo.

    - Filtra “music/photos”
    - Busca tokens “video”
    """
    t = title.strip().lower()
    if not t:
        return False

    negative = (
        "music",
        "música",
        "audio",
        "photo",
        "photos",
        "foto",
        "fotos",
        "picture",
        "pictures",
        "imagen",
        "imágenes",
    )
    if any(n in t for n in negative):
        return False

    positive = ("video", "vídeo", "videos", "vídeos")
    return any(p in t for p in positive)


def _folder_browse_container_score(title: str) -> int:
    """
    Score para detectar el típico “Browse by folder / Folders / Carpetas”.
    """
    t = title.strip().lower()
    if not t:
        return 0

    strong = ("by folder", "browse folders", "examinar carpetas", "carpetas", "por carpeta", "folders")
    for s in strong:
        if t == s or s in t:
            return 100
    return 0


def _is_plex_virtual_container_title(title: str) -> bool:
    """
    Plex expone “vistas virtuales” por DLNA. No queremos analizar esas “carpetas”.
    """
    t = title.strip().lower()
    if not t:
        return True

    plex_virtual_tokens = (
        "video channels",
        "channels",
        "shared video",
        "remote video",
        "watch later",
        "recommended",
        "preferences",
        "continue watching",
        "recently viewed",
        "recently added",
        "recently released",
        "by collection",
        "by edition",
        "by genre",
        "by year",
        "by decade",
        "by director",
        "by starring actor",
        "by country",
        "by content rating",
        "by rating",
        "by resolution",
        "by first letter",
    )
    return any(tok in t for tok in plex_virtual_tokens)


def _list_root_containers(device: DLNADevice) -> tuple[list[_DlnaContainer], tuple[str, str] | None]:
    """
    Lista contenedores bajo el ObjectID raíz "0".
    """
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        logger.error(f"[DLNA] El dispositivo '{device.friendly_name}' no expone ContentDirectory.")
        return [], None

    control_url, service_type = endpoints
    root_children = _soap_browse_direct_children(control_url, service_type, "0", 0, 500)
    if root_children is None:
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
    Algunos servers exponen:
      Video -> (Browse by folder) -> (Folders) -> ...
    Bajamos automáticamente hasta 3 niveles si encontramos una carpeta “fuerte”.
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
# EXTRACCIÓN DE ITEMS DE VÍDEO
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


def _extract_video_item(elem: ET.Element) -> _DlnaVideoItem | None:
    """Extrae título, resource_url y size de un <item>."""
    title: str | None = None
    resource_url: str | None = None
    size_bytes: int | None = None

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

    return _DlnaVideoItem(title=title, resource_url=resource_url, size_bytes=size_bytes)


def _iter_video_items_recursive(device: DLNADevice, root_object_id: str) -> list[_DlnaVideoItem]:
    """
    Recorre recursivamente un árbol de contenedores DLNA y devuelve items de vídeo.

    Nota:
    - Se materializa la lista.
      * NO SILENT necesita total real para (i/total).
      * SILENT usa “por contenedor” y el escaneo completo simplifica.
    """
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        return []

    control_url, service_type = endpoints
    results: list[_DlnaVideoItem] = []
    stack: list[str] = [root_object_id]

    while stack:
        current_id = stack.pop()
        start = 0
        total = 1

        while start < total:
            browse = _soap_browse_direct_children(control_url, service_type, current_id, start, _BROWSE_PAGE_SIZE)
            if browse is None:
                break

            children, total_matches = browse
            total = total_matches

            for elem in children:
                if not isinstance(elem.tag, str):
                    continue

                if elem.tag.endswith("container"):
                    cid = elem.attrib.get("id")
                    if cid:
                        stack.append(cid)
                    continue

                if not elem.tag.endswith("item"):
                    continue

                if not _is_video_item(elem):
                    continue

                item = _extract_video_item(elem)
                if item is not None:
                    results.append(item)

            start += _BROWSE_PAGE_SIZE

    return results


# ============================================================================
# NORMALIZACIÓN PARA PIPELINE (título / file)
# ============================================================================


def _extract_year_from_title(title: str) -> tuple[str, int | None]:
    """
    Intenta parsear sufijo “(YYYY)” al final:
      "Alien (1979)" -> ("Alien", 1979)
    """
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
    """
    Intenta extraer extensión del último path segment del URL.
    Devuelve "" si no es fiable.
    """
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
    """
    Genera un `file` friendly y devuelve también la URL del recurso.

    - library se usa como “carpeta lógica” en el CSV.
    - Se intenta añadir extensión detectada desde URL.
    """
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
    - Selecciona contenedores (carpetas).
    - Escanea items por contenedor.
    - Analiza items con analyze_movie() en paralelo (bounded inflight).
    - Escribe CSVs en streaming y genera filtered al final.

    Importante:
    - Este orquestador llama a flush_external_caches() UNA sola vez al final.
      Esto reduce I/O y mantiene throughput alto.
    """
    t0 = time.monotonic()
    reset_omdb_metrics()

    # Aseguramos flush de caches aunque haya early-return o excepción inesperada.
    # (No es “por item”; es “por run”.)
    try:
        if device is None:
            device = _ask_dlna_device()
            if device is None:
                return

        server_label = f"{device.friendly_name} ({device.host}:{device.port})"
        logger.progress(f"[DLNA] Servidor: {server_label}")
        logger.debug_ctx("DLNA", f"location={device.location!r}")

        roots = _list_video_root_containers(device)
        if not roots:
            logger.error("[DLNA] No se han encontrado contenedores raíz de vídeo.", always=True)
            return

        # Selección de raíz de vídeo
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

        # Auto-descenso si detectamos “Browse by folder”
        base = _auto_descend_folder_browse(device, chosen_root)
        if base.object_id != chosen_root.object_id:
            logger.debug_ctx("DLNA", f"auto_descend: {chosen_root.title!r} -> {base.title!r}")

        # Selección de carpetas
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

        max_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS)
        max_inflight = _compute_max_inflight(max_workers)

        # Mensaje de config de workers
        if SILENT_MODE:
            logger.progress(
                f"[DLNA] ThreadPool workers={max_workers} inflight_cap={max_inflight} (por contenedor) "
                f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, "
                f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
                f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
            )
        else:
            logger.debug_ctx(
                "DLNA",
                f"ThreadPool workers={max_workers} inflight_cap={max_inflight} (por contenedor, cap por OMDb limiter) "
                f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS}",
            )

        filtered_rows: list[_Row] = []
        decisions_count: dict[str, int] = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}
        total_items_processed = 0
        total_items_errors = 0
        scan_errors = 0
        total_rows_written = 0
        total_suggestions_written = 0

        container_omdb_delta_scan: dict[str, dict[str, int]] = {}
        container_omdb_delta_analyze: dict[str, dict[str, int]] = {}

        analyze_snapshot_start: dict[str, object] | None = None
        if SILENT_MODE and DEBUG_MODE:
            analyze_snapshot_start = dict(get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: start:")

        def _maybe_print_item_logs(logs: list[str]) -> None:
            """
            `logs` vienen acotados desde collection_analysis.analyze_movie().

            Política:
            - NO SILENT: se imprimen como info normal.
            - SILENT+DEBUG: se imprimen always=True (útil para inspección).
            - SILENT sin debug: no se imprime nada (evitamos ruido/costes).
            """
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
            """
            Normaliza un item DLNA -> MovieInput y ejecuta analyze_movie().

            Importante:
            - source_movie=None (DLNA no tiene objeto Plex).
            - extra incluye source_url para trazabilidad.
            """
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

            # Ajustes “cosméticos” para DLNA
            if row:
                row["file"] = display_file
                row["file_url"] = file_url  # puede no estar en fieldnames del CSV, no pasa nada

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

        # =========================================================================
        # Writers globales (streaming)
        # =========================================================================
        with open_all_csv_writer(REPORT_ALL_PATH) as all_writer, open_suggestions_csv_writer(METADATA_FIX_PATH) as sugg_writer:
            # ---------------------------------------------------------------------
            # NO SILENT:
            # - materializa candidatos para tener total real y mostrar (i/total)
            # - luego analiza por contenedor con bounded inflight y orden estable
            # ---------------------------------------------------------------------
            if not SILENT_MODE:
                candidates_by_container: dict[str, list[tuple[str, str, int | None, str]]] = {}
                total_candidates = 0

                total_containers = len(selected_containers)
                for idx, container in enumerate(selected_containers, start=1):
                    logger.progress(f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {container.title}")

                    try:
                        items = _iter_video_items_recursive(device, container.object_id)
                    except Exception as exc:  # pragma: no cover
                        scan_errors += 1
                        logger.error(f"[DLNA] Error escaneando '{container.title}': {exc!r}", always=True)
                        continue

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

                    # Orden estable por contenedor
                    future_to_index: dict[Future[tuple[_Row | None, _Row | None, list[str]]], int] = {}
                    pending_by_index: dict[int, tuple[_Row | None, _Row | None, list[str]]] = {}
                    next_to_write = 1

                    inflight: set[Future[tuple[_Row | None, _Row | None, list[str]]]] = set()

                    def _drain_completed(*, drain_all: bool) -> None:
                        """
                        Drena futures completados.

                        - Guarda resultados por índice.
                        - Vuelca en orden cuando se puede (next_to_write).
                        """
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
                                    logger.error(f"[DLNA] Error analizando item (future): {exc!r}", always=True)
                                    total_items_processed += 1
                                    continue

                                if idx_local >= 1:
                                    pending_by_index[idx_local] = res

                                # Vuelca todo lo consecutivo posible
                                while next_to_write in pending_by_index:
                                    ready = pending_by_index.pop(next_to_write)
                                    _handle_result(ready, all_writer=all_writer, sugg_writer=sugg_writer)
                                    total_items_processed += 1
                                    next_to_write += 1

                            if not drain_all:
                                return

                    workers_here = min(max_workers, max(1, len(items)))
                    inflight_cap_here = min(max_inflight, max(1, len(items)))

                    with ThreadPoolExecutor(max_workers=workers_here) as executor:
                        for item_index, (raw_title, resource_url, file_size, library) in enumerate(items, start=1):
                            analyzed_so_far += 1

                            # Progreso humano (solo NO SILENT)
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

                    # Fallback: si algo quedó en pending (no debería), vaciamos ordenado
                    while next_to_write in pending_by_index:
                        ready = pending_by_index.pop(next_to_write)
                        _handle_result(ready, all_writer=all_writer, sugg_writer=sugg_writer)
                        total_items_processed += 1
                        next_to_write += 1

            # ---------------------------------------------------------------------
            # SILENT:
            # - streaming por contenedor
            # - bounded inflight por contenedor
            # - métricas por fase (scan/analyze) si DEBUG_MODE
            # ---------------------------------------------------------------------
            else:
                total_containers = len(selected_containers)
                total_candidates = 0
                analyzed_so_far = 0

                for idx, container in enumerate(selected_containers, start=1):
                    container_title = container.title
                    container_key = container_title or f"<container_{idx}>"

                    logger.progress(f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {container_title}")

                    scan_snap_start: dict[str, object] | None = None
                    if DEBUG_MODE:
                        scan_snap_start = dict(get_omdb_metrics_snapshot())
                        _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: scan:start:")

                    try:
                        items = _iter_video_items_recursive(device, container.object_id)
                    except Exception as exc:  # pragma: no cover
                        scan_errors += 1
                        logger.error(f"[DLNA] Error escaneando '{container_title}': {exc!r}", always=True)
                        continue

                    total_candidates += len(items)
                    logger.debug_ctx("DLNA", f"scan {container_title!r} items={len(items)}")

                    if DEBUG_MODE and scan_snap_start is not None:
                        scan_delta = _metrics_diff(scan_snap_start, get_omdb_metrics_snapshot())
                        container_omdb_delta_scan[container_key] = dict(scan_delta)
                        _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: scan:delta:", metrics=scan_delta)

                    if not items:
                        continue

                    logger.progress(
                        f"[DLNA] ({idx}/{total_containers}) Analizando contenedor: {container_title} (items={len(items)})"
                    )

                    analyze_snap_start_container: dict[str, object] | None = None
                    if DEBUG_MODE:
                        analyze_snap_start_container = dict(get_omdb_metrics_snapshot())
                        _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: analyze:start:")

                    workers_here = min(max_workers, max(1, len(items)))
                    inflight_cap_here = min(max_inflight, max(1, len(items)))

                    with ThreadPoolExecutor(max_workers=workers_here) as executor:
                        inflight: set[Future[tuple[_Row | None, _Row | None, list[str]]]] = set()

                        def _drain_completed(*, drain_all: bool) -> None:
                            nonlocal total_items_processed, total_items_errors

                            while inflight:
                                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                                if not done:
                                    return

                                for fut in done:
                                    inflight.discard(fut)
                                    try:
                                        res = fut.result()
                                    except Exception as exc:  # pragma: no cover
                                        total_items_errors += 1
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

                            fut = executor.submit(
                                _analyze_one,
                                raw_title=it.title,
                                resource_url=it.resource_url,
                                file_size=it.size_bytes,
                                library=container_title,
                            )
                            inflight.add(fut)

                            if len(inflight) >= inflight_cap_here:
                                _drain_completed(drain_all=False)

                        _drain_completed(drain_all=True)

                    if DEBUG_MODE and analyze_snap_start_container is not None:
                        analyze_delta_container = _metrics_diff(analyze_snap_start_container, get_omdb_metrics_snapshot())
                        container_omdb_delta_analyze[container_key] = dict(analyze_delta_container)
                        _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: analyze:delta:", metrics=analyze_delta_container)

                if total_candidates == 0 and scan_errors == 0:
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

        # Métricas globales delta
        if SILENT_MODE and DEBUG_MODE and analyze_snapshot_start is not None:
            analyze_delta = _metrics_diff(analyze_snapshot_start, get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: delta:", metrics=analyze_delta)

        # Resumen final (principalmente en SILENT)
        if SILENT_MODE:
            logger.progress(
                "[DLNA] Resumen final: "
                f"server={server_label} containers={len(selected_containers)} "
                f"workers={max_workers} inflight_cap={max_inflight} time={elapsed:.1f}s | "
                f"scan_errors={scan_errors} analysis_errors={total_items_errors} | "
                f"items={total_items_processed} rows={total_rows_written} "
                f"(KEEP={decisions_count['KEEP']} MAYBE={decisions_count['MAYBE']} "
                f"DELETE={decisions_count['DELETE']} UNKNOWN={decisions_count['UNKNOWN']}) | "
                f"filtered_rows={filtered_len} filtered_csv={filtered_csv_status} "
                f"suggestions={total_suggestions_written}"
            )

            logger.progress(
                "[DLNA] CSVs: "
                f"all={REPORT_ALL_PATH} | suggestions={METADATA_FIX_PATH} | filtered={REPORT_FILTERED_PATH}"
            )

            _log_omdb_metrics(prefix="[DLNA][DEBUG] Global:")

            if DEBUG_MODE:
                if container_omdb_delta_scan:
                    logger.progress("[DLNA][DEBUG] Rankings (scan deltas):")
                    _log_omdb_rankings(container_omdb_delta_scan, min_groups=2)
                if container_omdb_delta_analyze:
                    logger.progress("[DLNA][DEBUG] Rankings (analyze deltas):")
                    _log_omdb_rankings(container_omdb_delta_analyze, min_groups=2)

        logger.info("[DLNA] Análisis completado.", always=True)

    finally:
        # ✅ Patch mínimo solicitado:
        # - Mantiene el comportamiento “flush una vez al final del run”.
        # - Se ejecuta aunque haya returns tempranos o excepción inesperada.
        try:
            flush_external_caches()
        except Exception as exc:  # pragma: no cover
            logger.debug_ctx("DLNA", f"flush_external_caches failed: {exc!r}")