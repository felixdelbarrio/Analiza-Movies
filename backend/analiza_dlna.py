from __future__ import annotations

"""
analiza_dlna.py

Orquestador principal de análisis DLNA/UPnP.

Workflow (alto nivel):
1) Descubre servidores DLNA/UPnP en la red (o usa el device inyectado).
2) Encuentra contenedores raíz de vídeo y permite elegir uno.
3) Intenta descender automáticamente a un “browse by folder” si se detecta.
4) Permite seleccionar contenedores (carpetas) a analizar.
5) Escanea recursivamente el árbol DLNA desde esos contenedores y construye la lista
   de candidatos (items de vídeo).
6) Analiza los candidatos con analyze_movie(MovieInput):
   - OMDb + Wiki (cacheados)
   - decisión KEEP/MAYBE/DELETE, etc.
7) Genera CSVs:
   - report_all.csv
   - report_filtered.csv
   - metadata_fix.csv

Objetivos de salida por consola:
- SILENT_MODE=True:
    - Evitar logs detallados
    - Mantener señales mínimas de progreso (servidor / contenedor / resúmenes)
- SILENT_MODE=False:
    - Mostrar progreso por item (i/total) en consola
    - En modo normal, mostrar año si existe
- DEBUG_MODE=True:
    - Permitir más visibilidad (heartbeat cada N elementos, más contexto)
    - En modo normal, añadir extra útil por item (p.ej. tamaño si se conoce)

Performance:
- La fase de “análisis” (OMDb/Wiki) es I/O bound.
- Se usa paralelismo controlado con ThreadPool para acelerar.
- El nº de workers se lee de config.py (PLEX_ANALYZE_WORKERS) para mantener
  una configuración única y coherente en el proyecto.

Métricas OMDb por contenedor:
- Para que los deltas por contenedor sean REALMENTE atribuibles, se crea un ThreadPool
  por contenedor (barrera de concurrencia por grupo).
- Así evitamos que el trabajo del contenedor N se solape con el N+1.

Alineación con OMDb limiter (config.py):
- OMDB_HTTP_MAX_CONCURRENCY (semaforo global en omdb_client)
- OMDB_HTTP_MIN_INTERVAL_SECONDS (throttle global)
Aunque pongas muchos workers, OMDb se “suaviza”, pero aun así capamos workers
a un valor razonable para no crear hilos que van a estar casi siempre esperando.
"""

import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from backend import logger as _logger
from backend.collection_analysis import analyze_movie
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
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv


# ============================================================================
# MODELOS INTERNOS (DLNA)
# ============================================================================


@dataclass(frozen=True, slots=True)
class _DlnaContainer:
    """
    Representa un contenedor DLNA (carpeta / categoría) accesible por ObjectID.
    """

    object_id: str
    title: str


@dataclass(frozen=True, slots=True)
class _DlnaVideoItem:
    """
    Representa un item de vídeo DLNA (título + URL del recurso).
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

# Heartbeat para ejecución silenciosa + debug: evita “pantalla muerta”
_PROGRESS_EVERY_N_ITEMS: int = 100

# Tamaño de paginación DIDL al navegar contenedores (trade-off entre latencia y payload)
_BROWSE_PAGE_SIZE: int = 200


# ============================================================================
# LOGGING CONTROLADO POR MODOS (en línea con el resto del proyecto)
# ============================================================================


def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False → no hace nada.
    - DEBUG_MODE=True:
        * SILENT_MODE=True: progress para señales de vida sin ruido excesivo.
        * SILENT_MODE=False: info normal.
    """
    if not DEBUG_MODE:
        return

    text = str(msg)
    try:
        if SILENT_MODE:
            _logger.progress(f"[DLNA][DEBUG] {text}")
        else:
            _logger.info(f"[DLNA][DEBUG] {text}")
    except Exception:
        if not SILENT_MODE:
            print(text)


# --------------------------
# OMDb metrics helpers (igual que Plex)
# --------------------------


def _metrics_get_int(m: dict[str, object], key: str) -> int:
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


def _metrics_diff(before: dict[str, object], after: dict[str, object]) -> dict[str, int]:
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


def _log_omdb_metrics(prefix: str, metrics: dict[str, object] | None = None) -> None:
    """
    Imprime (solo en silent+debug) un resumen rápido del comportamiento del cliente OMDb:
    cache hits/misses, requests, sleeps por throttle, rate-limit, etc.

    Si `metrics` es None, usa snapshot global actual.
    """
    if not (SILENT_MODE and DEBUG_MODE):
        return

    m = metrics or get_omdb_metrics_snapshot()
    _logger.progress(
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


def _rank_top(
    deltas_by_group: dict[str, dict[str, int]],
    key: str,
    top_n: int = 5,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for group, d in deltas_by_group.items():
        rows.append((group, int(d.get(key, 0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _compute_cost_score(delta: dict[str, int]) -> int:
    """
    Score “coste total” para detectar contenedores “caros”.
    Pesos elegidos para reflejar dolor real:
      - rate_limit_sleeps: muy caro (bloquea y alarga ejecución) -> *50
      - http_failures: caro (reintentos/ruido/decisiones peor)  -> *20
      - rate_limit_hits: señal de presión                             -> *10
      - throttle_sleeps: coste moderado                                -> *3
      - http_requests: coste base                                       -> *1
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


def _rank_top_by_total_cost(
    deltas_by_group: dict[str, dict[str, int]],
    top_n: int = 5,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for group, d in deltas_by_group.items():
        rows.append((group, _compute_cost_score(d)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _log_omdb_rankings(deltas_by_group: dict[str, dict[str, int]], *, min_groups: int = 2) -> None:
    """
    Ranking automático (solo silent+debug) por deltas reales.
    Imprime solo si hay al menos `min_groups` grupos (por defecto: 2).
    """
    if not (SILENT_MODE and DEBUG_MODE):
        return
    if not deltas_by_group or len(deltas_by_group) < min_groups:
        return

    def _fmt(items: list[tuple[str, int]]) -> str:
        usable = [(name, val) for (name, val) in items if val > 0]
        return " | ".join([f"{i+1}) {name}: {val}" for i, (name, val) in enumerate(usable)])

    top_cost = _rank_top_by_total_cost(deltas_by_group, top_n=5)
    top_http = _rank_top(deltas_by_group, "http_requests", top_n=5)
    top_rls = _rank_top(deltas_by_group, "rate_limit_sleeps", top_n=5)
    top_fail = _rank_top(deltas_by_group, "http_failures", top_n=5)
    top_rlh = _rank_top(deltas_by_group, "rate_limit_hits", top_n=5)

    cost_line = _fmt(top_cost)
    http_line = _fmt(top_http)
    rls_line = _fmt(top_rls)
    fail_line = _fmt(top_fail)
    rlh_line = _fmt(top_rlh)

    if cost_line:
        _logger.progress(f"[DLNA][DEBUG] Top containers by TOTAL_COST: {cost_line}")
    if http_line:
        _logger.progress(f"[DLNA][DEBUG] Top containers by http_requests: {http_line}")
    if rls_line:
        _logger.progress(f"[DLNA][DEBUG] Top containers by rate_limit_sleeps: {rls_line}")
    if fail_line:
        _logger.progress(f"[DLNA][DEBUG] Top containers by http_failures: {fail_line}")
    if rlh_line:
        _logger.progress(f"[DLNA][DEBUG] Top containers by rate_limit_hits: {rlh_line}")


def _count_decisions(rows: list[dict[str, object]]) -> dict[str, int]:
    """
    Cuenta decisiones en all_rows para el resumen final.
    """
    out = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}
    for r in rows:
        d = r.get("decision")
        if d in ("KEEP", "MAYBE", "DELETE"):
            out[str(d)] += 1
        else:
            out["UNKNOWN"] += 1
    return out


# ============================================================================
# FORMATEO (progreso por item)
# ============================================================================


def _format_human_size(num_bytes: int) -> str:
    """
    Convierte bytes a unidades humanas (B, KiB, MiB, GiB, TiB).
    """
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 0

    while value >= 1024.0 and unit_index < (len(units) - 1):
        value /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_item_progress_line(
    *,
    index: int,
    total: int,
    title: str,
    year: int | None,
    file_size_bytes: int | None,
) -> str:
    """
    Construye la línea que se imprime en modo no-silent por item DLNA.

    Formato:
      - "(i/total) Título (Año) [Tamaño]"

    Reglas:
      - Año solo si existe
      - Tamaño solo si DEBUG_MODE=True y size está disponible
    """
    base = title.strip() or "UNKNOWN"
    if year is not None:
        base = f"{base} ({year})"

    if DEBUG_MODE and file_size_bytes is not None and file_size_bytes >= 0:
        base = f"{base} [{_format_human_size(file_size_bytes)}]"

    return f"({index}/{total}) {base}"


# ============================================================================
# UTILIDADES GENERALES
# ============================================================================


def _is_plex_server(device: DLNADevice) -> bool:
    """
    Detecta heurísticamente si el servidor DLNA es Plex Media Server.
    """
    return "plex media server" in device.friendly_name.lower()


def _xml_text(elem: ET.Element | None) -> str | None:
    """
    Extrae texto de un elemento XML (normalizado y sin espacios).
    """
    if elem is None or elem.text is None:
        return None
    val = elem.text.strip()
    return val or None


def _fetch_xml_root(url: str, timeout_s: float = 5.0) -> ET.Element | None:
    """
    Descarga y parsea XML desde una URL.
    """
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            data = resp.read()
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] No se pudo descargar XML {url}: {exc}")
        return None

    try:
        return ET.fromstring(data)
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] No se pudo parsear XML {url}: {exc}")
        return None


def _find_content_directory_endpoints(device_location: str) -> tuple[str, str] | None:
    """
    Dado el LOCATION del device, encuentra:
      - controlURL del servicio ContentDirectory
      - serviceType necesario para SOAPAction
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
    Ejecuta un SOAP BrowseDirectChildren sobre ContentDirectory.

    Devuelve:
      - lista de elementos DIDL (containers + items)
      - totalMatches (para paginación)
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
        _logger.error(f"[DLNA] Error SOAP Browse contra {control_url}: {exc}")
        return None

    try:
        envelope = ET.fromstring(raw)
    except Exception as exc:  # pragma: no cover
        _logger.error(f"[DLNA] Respuesta SOAP inválida: {exc}")
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

    try:
        didl = ET.fromstring(result_text)
    except Exception:
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
            _logger.error(f"[DLNA] No se pudo parsear DIDL-Lite: {exc}")
            return None

    children = list(didl)
    return children, (total_matches or len(children))


def _extract_container_title_and_id(container: ET.Element) -> _DlnaContainer | None:
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
    t = title.strip().lower()
    if not t:
        return 0

    strong = (
        "by folder",
        "browse folders",
        "examinar carpetas",
        "carpetas",
        "por carpeta",
        "folders",
    )
    for s in strong:
        if t == s or s in t:
            return 100
    return 0


def _is_plex_virtual_container_title(title: str) -> bool:
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


def _list_root_containers(
    device: DLNADevice,
) -> tuple[list[_DlnaContainer], tuple[str, str] | None]:
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        _logger.error(
            f"[DLNA] El dispositivo '{device.friendly_name}' no expone ContentDirectory."
        )
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
    containers, _ = _list_root_containers(device)
    return [c for c in containers if _is_likely_video_root_title(c.title)]


def _list_child_containers(device: DLNADevice, parent_object_id: str) -> list[_DlnaContainer]:
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
# INTERACCIÓN (SELECCIÓN DE SERVER / CARPETAS)
# ============================================================================


def _ask_dlna_device() -> DLNADevice | None:
    _logger.info("\nBuscando servidores DLNA/UPnP en la red...\n", always=True)
    devices = discover_dlna_devices()

    if not devices:
        _logger.error("[DLNA] No se han encontrado servidores DLNA/UPnP.", always=True)
        return None

    _logger.info("Se han encontrado los siguientes servidores DLNA/UPnP:\n", always=True)
    for idx, dev in enumerate(devices, start=1):
        _logger.info(f"  {idx}) {dev.friendly_name} ({dev.host}:{dev.port})", always=True)
        _logger.info(f"      LOCATION: {dev.location}", always=True)

    while True:
        raw = input(
            f"\nSelecciona un servidor (1-{len(devices)}) o pulsa Enter para cancelar: "
        ).strip()
        if raw == "":
            _logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if not raw.isdigit():
            _logger.warning(
                "Opción no válida. Debe ser un número (o Enter para cancelar).",
                always=True,
            )
            continue
        num = int(raw)
        if not (1 <= num <= len(devices)):
            _logger.warning("Opción fuera de rango.", always=True)
            continue
        chosen = devices[num - 1]
        _logger.info(
            f"\nHas seleccionado: {chosen.friendly_name} ({chosen.host}:{chosen.port})\n",
            always=True,
        )
        return chosen


def _parse_multi_selection(raw: str, max_value: int) -> list[int] | None:
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
    _logger.info("\nMenú (Enter cancela):", always=True)
    _logger.info("  0) Todas las carpetas de vídeo de DLNA", always=True)
    _logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    while True:
        raw = input("Selecciona una opción (0/1) o pulsa Enter para cancelar: ").strip()
        if raw == "":
            _logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if raw not in ("0", "1"):
            _logger.warning(
                "Opción no válida. Introduce 0 o 1 (o Enter para cancelar).",
                always=True,
            )
            continue

        if raw == "0":
            return [base]

        folders = _list_child_containers(device, base.object_id)

        if not folders:
            _logger.error(
                "[DLNA] No se han encontrado carpetas dentro del contenedor seleccionado.",
                always=True,
            )
            return None

        _logger.info("\nCarpetas detectadas (Enter cancela):", always=True)
        for idx, c in enumerate(folders, start=1):
            _logger.info(f"  {idx}) {c.title}", always=True)

        raw_sel = input(
            "Selecciona carpetas separadas por comas (ej: 1,2) o pulsa Enter para cancelar: "
        ).strip()
        if raw_sel == "":
            _logger.info("[DLNA] Operación cancelada.", always=True)
            return None

        selected = _parse_multi_selection(raw_sel, len(folders))
        if selected is None:
            _logger.warning(
                f"Selección no válida. Usa números 1-{len(folders)} separados por comas (ej: 1,2).",
                always=True,
            )
            continue

        return [folders[i - 1] for i in selected]


def _select_folders_plex(base: _DlnaContainer, device: DLNADevice) -> list[_DlnaContainer] | None:
    _logger.info("\nOpciones Plex (Enter cancela):", always=True)
    _logger.info("  0) Todas las carpetas de vídeo de Plex Media Server", always=True)
    _logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    while True:
        raw = input("Selecciona una opción (0/1) o pulsa Enter para cancelar: ").strip()
        if raw == "":
            _logger.info("[DLNA] Operación cancelada.", always=True)
            return None
        if raw not in ("0", "1"):
            _logger.warning(
                "Opción no válida. Introduce 0 o 1 (o Enter para cancelar).",
                always=True,
            )
            continue

        if raw == "0":
            return [base]

        folders = _list_child_containers(device, base.object_id)
        folders = [c for c in folders if not _is_plex_virtual_container_title(c.title)]

        if not folders:
            _logger.error(
                "[DLNA] No se han encontrado carpetas Plex seleccionables (tras filtrar vistas/servicios).",
                always=True,
            )
            return None

        _logger.info("\nCarpetas detectadas en Plex (Enter cancela):", always=True)
        for idx, c in enumerate(folders, start=1):
            _logger.info(f"  {idx}) {c.title}", always=True)

        raw_sel = input(
            "Selecciona carpetas separadas por comas (ej: 1,2) o pulsa Enter para cancelar: "
        ).strip()
        if raw_sel == "":
            _logger.info("[DLNA] Operación cancelada.", always=True)
            return None

        selected = _parse_multi_selection(raw_sel, len(folders))
        if selected is None:
            _logger.warning(
                f"Selección no válida. Usa números 1-{len(folders)} separados por comas (ej: 1,2).",
                always=True,
            )
            continue

        return [folders[i - 1] for i in selected]


# ============================================================================
# EXTRACCIÓN DE ITEMS DE VÍDEO
# ============================================================================


def _is_video_item(elem: ET.Element) -> bool:
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


def _extract_video_item(elem: ET.Element) -> _DlnaVideoItem | None:
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
            size_attr = ch.attrib.get("size")
            if size_attr and size_attr.isdigit():
                size_bytes = int(size_attr)

    if not title or not resource_url:
        return None

    return _DlnaVideoItem(title=title, resource_url=resource_url, size_bytes=size_bytes)


def _iter_video_items_recursive(device: DLNADevice, root_object_id: str) -> list[_DlnaVideoItem]:
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
            browse = _soap_browse_direct_children(
                control_url,
                service_type,
                current_id,
                start,
                _BROWSE_PAGE_SIZE,
            )
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
# NORMALIZACIÓN PARA PIPELINE (título / archivo)
# ============================================================================


def _extract_year_from_title(title: str) -> tuple[str, int | None]:
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
    Requisito DLNA para reporting:
      - file (friendly) = "{library}/{Nombre original del registro sin normalizar}{ext}"
      - file_url        = URL real del servidor
    """
    ext = _extract_ext_from_resource_url(resource_url)

    base = raw_title.strip()
    if not base:
        base = "UNKNOWN"

    if ext:
        base_l = base.lower()
        ext_l = ext.lower()
        if not base_l.endswith(ext_l):
            base = f"{base}{ext}"

    return f"{library}/{base}", resource_url


# ============================================================================
# ORQUESTACIÓN PRINCIPAL
# ============================================================================


def analyze_dlna_server(device: DLNADevice | None = None) -> None:
    """
    Ejecuta el pipeline de análisis usando un servidor DLNA/UPnP.
    """
    t0 = time.monotonic()

    # Métricas OMDb: resumen “limpio” por ejecución.
    reset_omdb_metrics()

    if device is None:
        device = _ask_dlna_device()
        if device is None:
            return

    server_label = f"{device.friendly_name} ({device.host}:{device.port})"

    _logger.progress(f"[DLNA] Servidor: {server_label}")
    _log_debug(f"location={device.location!r}")

    roots = _list_video_root_containers(device)
    if not roots:
        _logger.error("[DLNA] No se han encontrado contenedores raíz de vídeo.")
        return

    chosen_root: _DlnaContainer
    if len(roots) == 1:
        chosen_root = roots[0]
    else:
        _logger.info("\nDirectorios raíz de vídeo (Enter cancela):", always=True)
        for idx, c in enumerate(roots, start=1):
            _logger.info(f"  {idx}) {c.title}", always=True)

        while True:
            raw = input(
                f"Selecciona un directorio de vídeo (1-{len(roots)}) o pulsa Enter para cancelar: "
            ).strip()
            if raw == "":
                _logger.info("[DLNA] Operación cancelada.", always=True)
                return
            if not raw.isdigit():
                _logger.warning("Opción no válida. Debe ser un número.", always=True)
                continue
            n = int(raw)
            if not (1 <= n <= len(roots)):
                _logger.warning("Opción fuera de rango.", always=True)
                continue
            chosen_root = roots[n - 1]
            break

    _logger.progress(f"[DLNA] Raíz de vídeo: {chosen_root.title}")

    base = _auto_descend_folder_browse(device, chosen_root)
    if base.object_id != chosen_root.object_id:
        _log_debug(f"auto_descend: '{chosen_root.title}' -> '{base.title}'")

    selected_containers: list[_DlnaContainer] | None
    if _is_plex_server(device):
        selected_containers = _select_folders_plex(base, device)
    else:
        selected_containers = _select_folders_non_plex(base, device)

    if selected_containers is None:
        return

    if SILENT_MODE:
        titles = [c.title for c in selected_containers]
        _logger.progress(f"[DLNA] Contenedores seleccionados: {len(titles)}")
        if DEBUG_MODE:
            _logger.progress("[DLNA][DEBUG] " + " | ".join(titles))

    # -------------------------------------------------------------------------
    # Fase 1: descubrir candidatos (scan recursivo)
    # -------------------------------------------------------------------------
    candidates: list[tuple[str, str, int | None, str]] = []
    scan_errors = 0

    # Snapshots/deltas OMDb por contenedor (solo se usan en silent+debug)
    container_omdb_snapshot_start: dict[str, dict[str, object]] = {}
    container_omdb_delta_scan: dict[str, dict[str, int]] = {}

    total_containers = len(selected_containers)
    for idx, container in enumerate(selected_containers, start=1):
        container_key = container.title or f"<container_{idx}>"

        _logger.progress(
            f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {container.title}"
        )

        if SILENT_MODE and DEBUG_MODE:
            snap = get_omdb_metrics_snapshot()
            container_omdb_snapshot_start[container_key] = dict(snap)
            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: scan:start:")

        try:
            items = _iter_video_items_recursive(device, container.object_id)
        except Exception as exc:  # pragma: no cover
            scan_errors += 1
            _logger.error(f"[DLNA] Error escaneando '{container.title}': {exc!r}")
            continue

        _log_debug(f"scan '{container.title}' items={len(items)}")

        for it in items:
            candidates.append((it.title, it.resource_url, it.size_bytes, container.title))

        if SILENT_MODE and DEBUG_MODE:
            snap_before = container_omdb_snapshot_start.get(container_key, {})
            snap_after = get_omdb_metrics_snapshot()
            delta = _metrics_diff(snap_before, snap_after)
            container_omdb_delta_scan[container_key] = dict(delta)
            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: scan:delta:", metrics=delta)

    if not candidates:
        if scan_errors > 0:
            _logger.progress(f"[DLNA] No se han encontrado items de vídeo. errors={scan_errors}")
        else:
            _logger.progress("[DLNA] No se han encontrado items de vídeo.")
        return

    total_candidates = len(candidates)
    _logger.progress(f"[DLNA] Candidatos a analizar: {total_candidates}")

    # -------------------------------------------------------------------------
    # Fase 2: análisis (ThreadPool controlado) - BARRERA POR CONTENEDOR
    # -------------------------------------------------------------------------
    max_workers = int(PLEX_ANALYZE_WORKERS)
    if max_workers < 1:
        max_workers = 1
    if max_workers > 64:
        max_workers = 64

    omdb_cap = max(4, int(OMDB_HTTP_MAX_CONCURRENCY) * 8)
    max_workers = min(max_workers, omdb_cap)

    if SILENT_MODE:
        _logger.progress(
            f"[DLNA] Analizando con ThreadPool workers={max_workers} (por contenedor) "
            f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, "
            f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
            f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
        )
    else:
        _log_debug(
            f"ThreadPool workers={max_workers} (por contenedor, cap={omdb_cap} por OMDb limiter) "
            f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS}"
        )

    all_rows: list[dict[str, object]] = []
    suggestions_rows: list[dict[str, object]] = []
    analysis_errors = 0

    container_omdb_delta_analyze: dict[str, dict[str, int]] = {}

    def _analyze_one(
        *,
        raw_title: str,
        resource_url: str,
        file_size: int | None,
        library: str,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None, list[str], str, str]:
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
        return row, meta_sugg, logs, display_file, file_url

    # Snapshot/delta del bloque de análisis global
    analyze_snapshot_start: dict[str, object] | None = None
    if SILENT_MODE and DEBUG_MODE:
        analyze_snapshot_start = dict(get_omdb_metrics_snapshot())
        _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: start:")

    # Agrupar candidatos por contenedor (library) para poder hacer barrera por grupo
    candidates_by_container: dict[str, list[tuple[str, str, int | None, str]]] = {}
    for raw_title, resource_url, file_size, library in candidates:
        candidates_by_container.setdefault(library, []).append((raw_title, resource_url, file_size, library))

    analyzed_so_far = 0
    for cont_index, (container_title, items) in enumerate(candidates_by_container.items(), start=1):
        if SILENT_MODE:
            _logger.progress(
                f"[DLNA] ({cont_index}/{len(candidates_by_container)}) Analizando contenedor: {container_title} "
                f"(items={len(items)})"
            )

        # start snapshot por contenedor (analysis)
        cont_snap_start: dict[str, object] | None = None
        if SILENT_MODE and DEBUG_MODE:
            cont_snap_start = dict(get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_title}: analyze:start:")

        # En modo no-silent mantenemos el progreso por item global (como antes),
        # pero ahora el procesamiento ocurre por lotes (contenedor).
        futures: list[
            Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str], str, str]]
        ] = []

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(items)))) as executor:
            for raw_title, resource_url, file_size, library in items:
                analyzed_so_far += 1

                if not SILENT_MODE:
                    clean_title_preview, extracted_year_preview = _extract_year_from_title(raw_title)
                    display_title = raw_title if DEBUG_MODE else clean_title_preview
                    _logger.info(
                        _format_item_progress_line(
                            index=analyzed_so_far,
                            total=total_candidates,
                            title=display_title,
                            year=extracted_year_preview,
                            file_size_bytes=file_size,
                        )
                    )

                if SILENT_MODE and DEBUG_MODE and (analyzed_so_far % _PROGRESS_EVERY_N_ITEMS == 0):
                    _logger.progress(f"[DLNA][DEBUG] Encolados {analyzed_so_far}/{total_candidates} items...")

                futures.append(
                    executor.submit(
                        _analyze_one,
                        raw_title=raw_title,
                        resource_url=resource_url,
                        file_size=file_size,
                        library=library,
                    )
                )

            completed = 0
            for fut in as_completed(futures):
                completed += 1

                if SILENT_MODE and DEBUG_MODE and (completed % _PROGRESS_EVERY_N_ITEMS == 0):
                    _logger.progress(
                        f"[DLNA][DEBUG] {container_title}: completados {completed}/{len(items)} items..."
                    )

                try:
                    row, meta_sugg, logs, display_file, file_url = fut.result()
                except Exception as exc:  # pragma: no cover
                    analysis_errors += 1
                    _logger.error(f"[DLNA] Error analizando item (future): {exc!r}")
                    continue

                for log in logs:
                    _logger.info(log)

                if row:
                    row["file_url"] = file_url
                    row["file"] = display_file
                    all_rows.append(row)

                if meta_sugg:
                    suggestions_rows.append(meta_sugg)

        # delta por contenedor (analysis)
        if SILENT_MODE and DEBUG_MODE and cont_snap_start is not None:
            cont_delta = _metrics_diff(cont_snap_start, get_omdb_metrics_snapshot())
            container_omdb_delta_analyze[container_title] = dict(cont_delta)
            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_title}: analyze:delta:", metrics=cont_delta)

    t_analyze_elapsed = time.monotonic() - t0

    if SILENT_MODE and DEBUG_MODE and analyze_snapshot_start is not None:
        analyze_delta = _metrics_diff(analyze_snapshot_start, get_omdb_metrics_snapshot())
        _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: delta:", metrics=analyze_delta)

    if not all_rows:
        _logger.progress("[DLNA] No se han generado filas de análisis.")
        return

    # -------------------------------------------------------------------------
    # Fase 3: filtrado + ordenación + CSVs
    # -------------------------------------------------------------------------
    filtered_rows = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered_rows = sort_filtered_rows(filtered_rows) if filtered_rows else []

    write_all_csv(REPORT_ALL_PATH, all_rows)
    write_filtered_csv(REPORT_FILTERED_PATH, filtered_rows)
    write_suggestions_csv(METADATA_FIX_PATH, suggestions_rows)

    elapsed = time.monotonic() - t0

    # -------------------------------------------------------------------------
    # Resumen final
    # -------------------------------------------------------------------------
    if SILENT_MODE:
        decisions = _count_decisions(all_rows)

        _logger.progress(
            "[DLNA] Resumen final: "
            f"server={server_label} containers={len(selected_containers)} "
            f"candidates={total_candidates} workers={max_workers} time={elapsed:.1f}s | "
            f"scan_errors={scan_errors} analysis_errors={analysis_errors} | "
            f"rows={len(all_rows)} (KEEP={decisions['KEEP']} MAYBE={decisions['MAYBE']} "
            f"DELETE={decisions['DELETE']} UNKNOWN={decisions['UNKNOWN']}) | "
            f"filtered_rows={len(filtered_rows)} suggestions={len(suggestions_rows)}"
        )

        _logger.progress(
            "[DLNA] CSVs: "
            f"all={REPORT_ALL_PATH} | filtered={REPORT_FILTERED_PATH} | suggestions={METADATA_FIX_PATH}"
        )

        # Resumen global OMDb (solo silent+debug)
        _log_omdb_metrics(prefix="[DLNA][DEBUG] Global:")

        # Ranking (scan) y ranking (analysis) separados para ver dónde se “quema” más
        _logger.progress("[DLNA][DEBUG] Rankings (scan deltas):")
        _log_omdb_rankings(container_omdb_delta_scan, min_groups=2)

        _logger.progress("[DLNA][DEBUG] Rankings (analyze deltas):")
        _log_omdb_rankings(container_omdb_delta_analyze, min_groups=2)

    _logger.info(
        (
            f"[DLNA] Análisis completado. CSV completo: {REPORT_ALL_PATH} | "
            f"CSV filtrado: {REPORT_FILTERED_PATH}"
        ),
        always=True,
    )