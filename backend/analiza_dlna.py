from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger
from backend.analyze_input_core import analyze_input_movie
from backend.config import (
    EXCLUDE_DLNA_LIBRARIES,
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from backend.decision_logic import sort_filtered_rows
from backend.dlna_discovery import DLNADevice
from backend.movie_input import MovieInput
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv
from backend.wiki_client import get_movie_record

VIDEO_EXTENSIONS: set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg",
}


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
    ).encode("utf-8")

    headers = {
        "Content-Type": "text/xml; charset=\"utf-8\"",
        "SOAPACTION": f"\"{service_type}#Browse\"",
    }

    req = Request(control_url, data=body, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=10.0) as resp:
            xml_bytes = resp.read()
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] Error SOAP Browse en {control_url}: {exc}")
        return None

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] Error parseando respuesta SOAP: {exc}")
        return None

    result_xml: str | None = None
    total_matches: int | None = None

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if elem.tag.endswith("Result"):
            result_xml = _xml_text(elem)
        elif elem.tag.endswith("TotalMatches"):
            raw = _xml_text(elem)
            if raw and raw.isdigit():
                total_matches = int(raw)

    if not result_xml:
        return None

    try:
        didl_root = ET.fromstring(result_xml)
    except Exception:  # pragma: no cover
        return None

    children = list(didl_root)
    return children, (total_matches or 0)


def _extract_container_title_and_id(elem: ET.Element) -> _DlnaContainer | None:
    object_id = elem.attrib.get("id")
    if not object_id:
        return None

    title: str | None = None
    for ch in list(elem):
        if not isinstance(ch.tag, str):
            continue
        if ch.tag.endswith("title"):
            title = _xml_text(ch)
            break

    if not title:
        return None

    return _DlnaContainer(object_id=object_id, title=title)


def _is_likely_video_root_title(title: str) -> bool:
    t = title.strip().lower()
    if not t:
        return False
    keywords = ("video", "vídeo", "movies", "films", "películas", "cine")
    return any(k in t for k in keywords)


def _is_non_video_container_title(title: str) -> bool:
    t = title.strip().lower()
    if not t:
        return True
    bad = ("music", "música", "photo", "foto", "images", "imagenes", "pictures")
    return any(k in t for k in bad)


def _folder_browse_container_score(title: str) -> int:
    t = title.strip().lower()
    if not t:
        return 0

    score = 0
    if "movies" in t or "pel" in t or "cine" in t or "films" in t:
        score += 3
    if "video" in t or "vídeo" in t:
        score += 2
    if "all" in t or "todo" in t:
        score += 1
    if "library" in t or "biblioteca" in t:
        score += 1

    return score


def _list_root_containers(
    device: DLNADevice,
) -> tuple[list[_DlnaContainer], tuple[str, str] | None]:
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
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
    if not containers:
        return []
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
    candidates: list[_DlnaContainer] = []
    for elem in children:
        if not (isinstance(elem.tag, str) and elem.tag.endswith("container")):
            continue
        c = _extract_container_title_and_id(elem)
        if c is not None:
            candidates.append(c)

    return candidates


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
                best = c
                best_score = score

        if best is None or best_score <= 0:
            return current

        current = best

    return current


def _navigate_subfolders(device: DLNADevice, container: _DlnaContainer) -> _DlnaContainer | None:
    current = container
    for _ in range(10):
        children = _list_child_containers(device, current.object_id)
        filtered = [c for c in children if not _is_non_video_container_title(c.title)]
        filtered = [c for c in filtered if c.title not in EXCLUDE_DLNA_LIBRARIES]

        if not filtered:
            return current

        _logger.info("\nSubdirectorios disponibles:", always=True)
        _logger.info("  0) Usar este directorio", always=True)
        for idx, c in enumerate(filtered, start=1):
            _logger.info(f"  {idx}) {c.title}", always=True)

        raw = input(
            f"Selecciona un subdirectorio (0-{len(filtered)}) o pulsa Enter para cancelar: "
        ).strip()

        if not raw:
            return None
        if not raw.isdigit():
            _logger.warning("Selecciona un número válido.", always=True)
            continue

        val = int(raw)
        if val == 0:
            return current
        if not (1 <= val <= len(filtered)):
            _logger.warning("Opción fuera de rango.", always=True)
            continue

        current = filtered[val - 1]

    return current


def _parse_comma_selection(raw: str, max_value: int) -> list[int] | None:
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


def _select_folders_menu_non_plex(
    device: DLNADevice,
    base: _DlnaContainer,
) -> list[_DlnaContainer] | None:
    """Menú:
    0) Todas las carpetas de vídeo de DLNA
    1) Seleccionar qué carpetas analizar (múltiple con comas)
    Enter cancela.
    """
    _logger.info(
        "\nOpciones de análisis DLNA (Enter cancela):",
        always=True,
    )
    _logger.info("  0) Todas las carpetas de vídeo de DLNA", always=True)
    _logger.info("  1) Seleccionar qué carpetas analizar", always=True)

    raw = input("Selecciona una opción (0/1) o pulsa Enter para cancelar: ").strip()
    if not raw:
        return None
    if raw not in ("0", "1"):
        _logger.warning(
            "Opción no válida. Introduce 0 o 1 (o Enter para cancelar).",
            always=True,
        )
        return _select_folders_menu_non_plex(device, base)

    if raw == "0":
        return [base]

    folders = _list_child_containers(device, base.object_id)
    folders = [c for c in folders if not _is_non_video_container_title(c.title)]
    folders = [c for c in folders if c.title not in EXCLUDE_DLNA_LIBRARIES]

    if not folders:
        _logger.warning(
            "No se han encontrado carpetas dentro de este contenedor.",
            always=True,
        )
        return None

    _logger.info("\nCarpetas disponibles (Enter cancela):", always=True)
    for idx, c in enumerate(folders, start=1):
        _logger.info(f"  {idx}) {c.title}", always=True)

    raw_sel = input(
        "Selecciona carpetas separadas por comas (ej: 1,2) o pulsa Enter para cancelar: "
    ).strip()
    if not raw_sel:
        return None

    selected_nums = _parse_comma_selection(raw_sel, len(folders))
    if selected_nums is None:
        _logger.warning(
            (
                f"Selección no válida. Usa números 1-{len(folders)} "
                "separados por comas (ej: 1,2)."
            ),
            always=True,
        )
        return _select_folders_menu_non_plex(device, base)

    return [folders[n - 1] for n in selected_nums]


def _ask_video_root_containers_to_analyze(
    device: DLNADevice,
) -> list[_DlnaContainer] | None:
    roots = _list_video_root_containers(device)
    if not roots:
        _logger.error(
            "[DLNA] No se han encontrado directorios raíz de vídeo navegables en el servidor.",
            always=True,
        )
        return None

    if len(roots) == 1:
        chosen = _auto_descend_folder_browse(device, roots[0])
        if _is_plex_server(device):
            single = _navigate_subfolders(device, chosen)
            if single is None:
                return None
            return [single]
        return _select_folders_menu_non_plex(device, chosen)

    _logger.info("\nSe han encontrado los siguientes directorios raíz de vídeo:", always=True)
    for idx, c in enumerate(roots, start=1):
        _logger.info(f"  {idx}) {c.title}", always=True)

    raw = input(
        f"Selecciona un directorio de vídeo (1-{len(roots)}) o pulsa Enter para cancelar: "
    ).strip()
    if not raw:
        return None
    if not raw.isdigit():
        _logger.warning(
            "Selecciona un número válido (o Enter para cancelar).",
            always=True,
        )
        return _ask_video_root_containers_to_analyze(device)

    val = int(raw)
    if not (1 <= val <= len(roots)):
        _logger.warning("Opción fuera de rango.", always=True)
        return _ask_video_root_containers_to_analyze(device)

    chosen = _auto_descend_folder_browse(device, roots[val - 1])
    if _is_plex_server(device):
        single = _navigate_subfolders(device, chosen)
        if single is None:
            return None
        return [single]

    return _select_folders_menu_non_plex(device, chosen)


def _extract_year_from_date(date_str: str) -> int | None:
    if len(date_str) >= 4 and date_str[:4].isdigit():
        year = int(date_str[:4])
        if 1900 <= year <= 2100:
            return year
    return None


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

    if protocol_info and "video" in protocol_info.lower():
        return True

    return False


def _extract_video_item(elem: ET.Element) -> _DlnaVideoItem | None:
    title: str | None = None
    res_url: str | None = None
    size_bytes: int | None = None
    year: int | None = None

    for ch in list(elem):
        if not isinstance(ch.tag, str):
            continue
        if ch.tag.endswith("title"):
            title = _xml_text(ch)
        elif ch.tag.endswith("res"):
            res_url = _xml_text(ch)
            size_raw = ch.attrib.get("size")
            if size_raw and size_raw.isdigit():
                size_bytes = int(size_raw)
        elif ch.tag.endswith("date"):
            date_str = _xml_text(ch)
            if date_str:
                year = _extract_year_from_date(date_str)

    if not title or not res_url:
        return None

    return _DlnaVideoItem(
        title=title,
        resource_url=res_url,
        size_bytes=size_bytes,
        year=year,
    )


def _iter_video_items_recursive(device: DLNADevice, container_object_id: str) -> list[_DlnaVideoItem]:
    endpoints = _find_content_directory_endpoints(device.location)
    if endpoints is None:
        return []

    control_url, service_type = endpoints

    start = 0
    step = 200
    all_items: list[_DlnaVideoItem] = []

    while True:
        resp = _soap_browse_direct_children(control_url, service_type, container_object_id, start, step)
        if resp is None:
            break

        children, total_matches = resp
        if not children:
            break

        for elem in children:
            if not isinstance(elem.tag, str):
                continue

            if elem.tag.endswith("container"):
                sub = _extract_container_title_and_id(elem)
                if sub is None:
                    continue
                if _is_non_video_container_title(sub.title):
                    continue
                all_items.extend(_iter_video_items_recursive(device, sub.object_id))

            elif elem.tag.endswith("item"):
                if not _is_video_item(elem):
                    continue
                item = _extract_video_item(elem)
                if item is not None:
                    all_items.append(item)

        start += len(children)
        if total_matches and start >= total_matches:
            break

        if len(children) < step:
            break

    return all_items


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _guess_title_year(path: Path) -> tuple[str, int | None]:
    stem = path.stem
    title = stem
    year: int | None = None

    if "(" in stem and ")" in stem:
        before = stem.split("(", 1)[0]
        inside = stem.split("(", 1)[1].split(")", 1)[0]
        if inside.strip().isdigit():
            year_int = int(inside.strip())
            if 1900 <= year_int <= 2100:
                return before.strip(), year_int

    parts = stem.split(".")
    for part in parts:
        if len(part) == 4 and part.isdigit():
            year_int = int(part)
            if 1900 <= year_int <= 2100:
                year = year_int
                break

    return title.strip(), year


def _ask_root_directory() -> Path:
    while True:
        raw = input("Ruta del directorio raíz a analizar (DLNA/local): ").strip()
        if not raw:
            _logger.warning("Debes introducir una ruta no vacía.", always=True)
            continue

        path = Path(raw).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            _logger.error(f"La ruta {path} no existe o no es un directorio.", always=True)
            continue

        return path


def _iter_video_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        dirp = Path(dirpath)
        for name in filenames:
            candidate = dirp / name
            if _is_video_file(candidate):
                files.append(candidate)
    return files


def analyze_dlna_server(device: DLNADevice | None = None) -> None:
    candidates: list[tuple[str, int | None, str, int | None, str]] = []

    if device is None:
        local_root = _ask_root_directory()
        library = local_root.name

        if library in EXCLUDE_DLNA_LIBRARIES:
            _logger.info(
                (
                    f"[DLNA] La biblioteca '{library}' está en EXCLUDE_DLNA_LIBRARIES; "
                    "se omite el análisis."
                ),
                always=True,
            )
            return

        files = _iter_video_files(local_root)
        if not files:
            _logger.info(
                f"No se han encontrado ficheros de vídeo en {local_root}",
                always=True,
            )
            return

        _logger.info(
            f"Analizando {len(files)} ficheros de vídeo bajo {local_root}",
            always=True,
        )

        for file_path in files:
            title, year = _guess_title_year(file_path)
            try:
                file_size = file_path.stat().st_size
            except OSError:
                file_size = None
            candidates.append((title, year, str(file_path), file_size, library))

    else:
        selected_containers = _ask_video_root_containers_to_analyze(device)
        if selected_containers is None:
            _logger.info("[DLNA] Operación cancelada.", always=True)
            return

        total_items = 0
        for container in selected_containers:
            if container.title in EXCLUDE_DLNA_LIBRARIES:
                _logger.info(
                    (
                        f"[DLNA] La carpeta '{container.title}' está en "
                        "EXCLUDE_DLNA_LIBRARIES; se omite."
                    ),
                    always=True,
                )
                continue

            items = _iter_video_items_recursive(device, container.object_id)
            total_items += len(items)
            for item in items:
                candidates.append(
                    (
                        item.title,
                        item.year,
                        item.resource_url,
                        item.size_bytes,
                        container.title,
                    )
                )

        if not candidates:
            _logger.info(
                "[DLNA] No se han encontrado items de vídeo para analizar.",
                always=True,
            )
            return

        _logger.info(
            (
                f"[DLNA] Analizando {total_items} item(s) de vídeo en las "
                "carpetas seleccionadas."
            ),
            always=True,
        )

    all_rows: list[dict[str, object]] = []
    suggestions_rows: list[dict[str, object]] = []

    for title, year, file_path_str, file_size, library in candidates:

        def fetch_omdb(
            title_for_fetch: str,
            year_for_fetch: int | None,
        ) -> dict[str, object]:
            record = get_movie_record(
                title=title_for_fetch,
                year=year_for_fetch,
                imdb_id_hint=None,
            )
            if record is None:
                return {}
            if isinstance(record, dict):
                return record
            return dict(record)

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
            base_row = analyze_input_movie(movie_input, fetch_omdb)
        except Exception as exc:  # pragma: no cover
            _logger.error(f"[DLNA] Error analizando {file_path_str}: {exc}", always=True)
            continue

        if not base_row:
            _logger.warning(
                f"[DLNA] analyze_input_movie devolvió fila vacía para {file_path_str}",
                always=True,
            )
            continue

        row: dict[str, object] = dict(base_row)
        row["file"] = file_path_str

        file_size_bytes = row.get("file_size_bytes")
        if isinstance(file_size_bytes, int):
            row["file_size"] = file_size_bytes
        else:
            row["file_size"] = file_size

        omdb_data = fetch_omdb(title, year)

        poster_url: str | None = None
        trailer_url: str | None = None
        imdb_id: str | None = None
        omdb_json_str: str | None = None
        wikidata_id: str | None = None
        wikipedia_title: str | None = None

        if omdb_data:
            poster_raw = omdb_data.get("Poster")
            trailer_raw = omdb_data.get("Website")
            imdb_id_raw = omdb_data.get("imdbID")

            if isinstance(poster_raw, str):
                poster_url = poster_raw
            if isinstance(trailer_raw, str):
                trailer_url = trailer_raw
            if isinstance(imdb_id_raw, str):
                imdb_id = imdb_id_raw

            try:
                omdb_json_str = json.dumps(omdb_data, ensure_ascii=False)
            except Exception:
                omdb_json_str = str(omdb_data)

            wiki_raw = omdb_data.get("__wiki")
            if isinstance(wiki_raw, dict):
                wikidata_val = wiki_raw.get("wikidata_id")
                wiki_title_val = wiki_raw.get("wikipedia_title")
                if isinstance(wikidata_val, str):
                    wikidata_id = wikidata_val
                if isinstance(wiki_title_val, str):
                    wikipedia_title = wiki_title_val

        row["poster_url"] = poster_url
        row["trailer_url"] = trailer_url
        row["imdb_id"] = imdb_id
        row["thumb"] = None
        row["omdb_json"] = omdb_json_str
        row["wikidata_id"] = wikidata_id
        row["wikipedia_title"] = wikipedia_title
        row["guid"] = None
        row["rating_key"] = None

        all_rows.append(row)

    if not all_rows:
        _logger.info("No se han generado filas de análisis para DLNA.", always=True)
        return

    filtered_rows = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered_rows = sort_filtered_rows(filtered_rows) if filtered_rows else []

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


if __name__ == "__main__":
    analyze_dlna_server()