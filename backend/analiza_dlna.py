from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger
from backend.collection_analysis import analyze_movie
from backend.config import (
    EXCLUDE_DLNA_LIBRARIES,
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from backend.decision_logic import sort_filtered_rows
from backend.dlna_discovery import DLNADevice, discover_dlna_devices
from backend.movie_input import MovieInput
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv


@dataclass(frozen=True, slots=True)
class _DlnaContainer:
    object_id: str
    title: str


@dataclass(frozen=True, slots=True)
class _DlnaVideoItem:
    title: str
    resource_url: str
    size_bytes: int | None
    year: int | None


def _is_plex_server(device: DLNADevice) -> bool:
    return "plex media server" in device.friendly_name.lower()


def _xml_text(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    val = elem.text.strip()
    return val or None


def _fetch_xml_root(url: str, timeout_s: float = 5.0) -> ET.Element | None:
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
        # Compatibilidad: algunos servidores esperan SOAPAction, otros SOAPACTION.
        "SOAPAction": soap_action,
        "SOAPACTION": soap_action,
    }

    req = Request(control_url, data=body.encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            raw = resp.read()
    except Exception as exc:  # pragma: no cover
        _logger.error(f"[DLNA] Error SOAP Browse contra {control_url}: {exc}", always=True)
        return None

    try:
        envelope = ET.fromstring(raw)
    except Exception as exc:  # pragma: no cover
        _logger.error(f"[DLNA] Respuesta SOAP inválida: {exc}", always=True)
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
            _logger.error(f"[DLNA] No se pudo parsear DIDL-Lite: {exc}", always=True)
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

    # Heurística genérica Plex: “vistas/servicios” típicos, no carpetas físicas.
    # No depende del usuario ni de la instalación.
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
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        _logger.error(
            f"[DLNA] El dispositivo '{device.friendly_name}' no expone ContentDirectory.",
            always=True,
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
            _logger.warning("Opción no válida. Debe ser un número (o Enter para cancelar).", always=True)
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
            _logger.warning("Opción no válida. Introduce 0 o 1 (o Enter para cancelar).", always=True)
            continue

        if raw == "0":
            return [base]

        folders = _list_child_containers(device, base.object_id)
        folders = [c for c in folders if c.title not in EXCLUDE_DLNA_LIBRARIES]

        if not folders:
            _logger.error("[DLNA] No se han encontrado carpetas dentro del contenedor seleccionado.", always=True)
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
            _logger.warning("Opción no válida. Introduce 0 o 1 (o Enter para cancelar).", always=True)
            continue

        if raw == "0":
            return [base]

        folders = _list_child_containers(device, base.object_id)
        folders = [c for c in folders if c.title not in EXCLUDE_DLNA_LIBRARIES]
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


def _extract_year_from_date(date_str: str) -> int | None:
    if len(date_str) >= 4 and date_str[:4].isdigit():
        y = int(date_str[:4])
        if 1900 <= y <= 2100:
            return y
    return None


def _extract_video_item(elem: ET.Element) -> _DlnaVideoItem | None:
    title: str | None = None
    resource_url: str | None = None
    size_bytes: int | None = None
    year: int | None = None

    for ch in list(elem):
        if not isinstance(ch.tag, str):
            continue

        if ch.tag.endswith("title") and title is None:
            title = _xml_text(ch)
        elif ch.tag.endswith("date") and year is None:
            dt = _xml_text(ch)
            if dt:
                year = _extract_year_from_date(dt)
        elif ch.tag.endswith("res") and resource_url is None:
            resource_url = _xml_text(ch)
            size_attr = ch.attrib.get("size")
            if size_attr and size_attr.isdigit():
                size_bytes = int(size_attr)

    if not title or not resource_url:
        return None

    return _DlnaVideoItem(
        title=title,
        resource_url=resource_url,
        size_bytes=size_bytes,
        year=year,
    )


def _iter_video_items_recursive(device: DLNADevice, root_object_id: str) -> list[_DlnaVideoItem]:
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        return []

    control_url, service_type = endpoints
    results: list[_DlnaVideoItem] = []
    stack: list[str] = [root_object_id]
    page_size = 200

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
                page_size,
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

            start += page_size

    return results


def analyze_dlna_server(device: DLNADevice | None = None) -> None:
    if device is None:
        device = _ask_dlna_device()
        if device is None:
            return

    roots = _list_video_root_containers(device)
    if not roots:
        _logger.error("[DLNA] No se han encontrado contenedores raíz de vídeo.", always=True)
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

    base = _auto_descend_folder_browse(device, chosen_root)

    selected_containers: list[_DlnaContainer] | None
    if _is_plex_server(device):
        selected_containers = _select_folders_plex(base, device)
    else:
        selected_containers = _select_folders_non_plex(base, device)

    if selected_containers is None:
        return

    candidates: list[tuple[str, int | None, str, int | None, str]] = []
    for container in selected_containers:
        if container.title in EXCLUDE_DLNA_LIBRARIES:
            _logger.info(f"[DLNA] Omitiendo '{container.title}' por EXCLUDE_DLNA_LIBRARIES.", always=True)
            continue

        items = _iter_video_items_recursive(device, container.object_id)
        for it in items:
            candidates.append((it.title, it.year, it.resource_url, it.size_bytes, container.title))

    if not candidates:
        _logger.info("[DLNA] No se han encontrado items de vídeo para analizar.", always=True)
        return

    _logger.info(f"[DLNA] Analizando {len(candidates)} item(s) de vídeo...", always=True)

    all_rows: list[dict[str, object]] = []
    suggestions_rows: list[dict[str, object]] = []

    for title, year, file_path_str, file_size, library in candidates:
        movie_input = MovieInput(
            source="dlna",
            library=library,
            title=title,
            year=year,
            file_path=file_path_str,
            file_size_bytes=file_size,
            imdb_id_hint=None,
            plex_guid=None,
            rating_key=None,
            thumb_url=None,
            extra={},
        )

        try:
            row, meta_sugg, logs = analyze_movie(movie_input, source_movie=None)
        except Exception as exc:  # pragma: no cover
            _logger.error(f"[DLNA] Error analizando {file_path_str}: {exc}", always=True)
            continue

        for log in logs:
            _logger.info(log)

        if row:
            all_rows.append(row)

        if meta_sugg:
            suggestions_rows.append(meta_sugg)

    if not all_rows:
        _logger.info("[DLNA] No se han generado filas de análisis.", always=True)
        return

    filtered_rows = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered_rows = sort_filtered_rows(filtered_rows) if filtered_rows else []

    # ---------------------------------------------------
    # Salidas CSV unificadas (siempre mismos nombres y en /reports)
    # ---------------------------------------------------
    write_all_csv(REPORT_ALL_PATH, all_rows)
    write_filtered_csv(REPORT_FILTERED_PATH, filtered_rows)
    write_suggestions_csv(METADATA_FIX_PATH, suggestions_rows)

    _logger.info(
        (
            f"[DLNA] Análisis completado. CSV completo: {REPORT_ALL_PATH} | "
            f"CSV filtrado: {REPORT_FILTERED_PATH}"
        ),
        always=True,
    )