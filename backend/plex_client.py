# backend/plex_client.py
from __future__ import annotations

"""
backend/plex_client.py

Cliente ligero para Plex (plexapi) + helpers de extracción defensiva.

Filosofía de logs (alineada con el proyecto):
- SILENT_MODE=True:
    * no spamear info normal
    * solo avisos realmente relevantes con always=True
    * si DEBUG_MODE=True, permitir señales pequeñas (progress)
- SILENT_MODE=False:
    * info normal para pasos visibles (conectar, saltar librerías, etc.)
- Este módulo NO debe imprimir dumps grandes ni stacktraces; se deja al caller.

Notas:
- PlexServer y objetos de plexapi no tienen tipado estricto aquí; usamos object y
  accesos via getattr defensivos.
"""

from collections.abc import Sequence
from typing import cast

from plexapi.server import PlexServer  # type: ignore[import-not-found]

from backend import logger as _logger
from backend.config import BASEURL, DEBUG_MODE, EXCLUDE_PLEX_LIBRARIES, PLEX_PORT, PLEX_TOKEN, SILENT_MODE


# ============================================================
#                  LOGGING CONTROLADO POR MODOS
# ============================================================


def _log(msg: object) -> None:
    """
    Log normal:
    - En SILENT_MODE, el logger central normalmente silencia info.
    - Fallback a print solo si no estamos en silent.
    """
    text = str(msg)
    try:
        _logger.info(text)
    except Exception:
        if not SILENT_MODE:
            print(text)


def _log_always(msg: object) -> None:
    """Log siempre visible incluso en SILENT_MODE (always=True)."""
    text = str(msg)
    try:
        _logger.warning(text, always=True)
    except Exception:
        print(text)


def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False: no hace nada
    - DEBUG_MODE=True:
        * SILENT_MODE=True: progress (señales mínimas)
        * SILENT_MODE=False: info normal
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
#                     CONEXIÓN A PLEX
# ============================================================


def _build_plex_base_url() -> str:
    """
    Construye la URL base para Plex a partir de BASEURL (host sin puerto) y PLEX_PORT.

    Ejemplo:
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
    Crea y devuelve una instancia de PlexServer usando la configuración.

    Reglas:
    - Si faltan BASEURL/PLEX_TOKEN -> RuntimeError (determinístico).
    - Si falla la conexión -> se re-lanza la excepción para que el caller decida.
    """
    if not BASEURL or not PLEX_TOKEN:
        raise RuntimeError("Faltan BASEURL o PLEX_TOKEN en el .env")

    base_url = _build_plex_base_url()

    # En SILENT no queremos ruido, pero esta señal suele ser útil si DEBUG.
    if DEBUG_MODE:
        _log_debug(f"Connecting to Plex at {base_url}")

    try:
        plex = PlexServer(base_url, PLEX_TOKEN)
    except Exception as exc:
        _log_always(f"[PLEX] ERROR conectando a Plex ({base_url}): {exc!r}")
        raise

    if not SILENT_MODE:
        _log(f"[PLEX] Conectado a Plex: {base_url}")
    else:
        if DEBUG_MODE:
            _log_debug("Connected")

    return plex


def get_libraries_to_analyze(plex: PlexServer) -> list[object]:
    """
    Devuelve la lista de bibliotecas de Plex a analizar, excluyendo las de
    EXCLUDE_PLEX_LIBRARIES.

    Nota:
    - Devolvemos list[object] por tipado laxo de plexapi.
    """
    libraries: list[object] = []

    try:
        sections = plex.library.sections()
    except Exception as exc:  # pragma: no cover
        _log_always(f"[PLEX] ERROR obteniendo secciones de Plex: {exc!r}")
        return libraries

    for section in sections:
        name_obj = getattr(section, "title", "")
        name = name_obj if isinstance(name_obj, str) else ""
        if name in EXCLUDE_PLEX_LIBRARIES:
            if not SILENT_MODE:
                _log(f"[PLEX] Saltando biblioteca excluida: {name}")
            else:
                _log_debug(f"Skipping excluded library: {name}")
            continue
        libraries.append(section)

    if DEBUG_MODE:
        _log_debug(f"Libraries selected: {len(libraries)} (excluded={len(EXCLUDE_PLEX_LIBRARIES)})")

    return libraries


# ============================================================
#                  INFO DE ARCHIVOS DE PELÍCULAS
# ============================================================


def get_movie_file_info(movie: object) -> tuple[str | None, int | None]:
    """
    Devuelve (ruta_principal, tamaño_total_en_bytes) para una película de Plex.

    Reglas:
      - Si no hay media o parts válidos -> (None, None).
      - La ruta devuelta es el `file` del primer part válido encontrado.
      - El tamaño es la suma de los tamaños (`size`) de todos los parts válidos.
      - Defensiva: nunca lanza; retorna (None, None) en caso de estructura rara.
    """
    try:
        media_seq = getattr(movie, "media", None)
        if not isinstance(media_seq, Sequence) or not media_seq:
            return None, None

        best_path: str | None = None
        total_size: int = 0
        size_seen: bool = False

        for media in media_seq:
            parts = getattr(media, "parts", None) or []
            if not isinstance(parts, Sequence):
                continue

            for part in parts:
                file_path_obj = getattr(part, "file", None)
                size_obj = getattr(part, "size", None)

                file_path = file_path_obj if isinstance(file_path_obj, str) and file_path_obj.strip() else None
                size_val = size_obj if isinstance(size_obj, int) and size_obj > 0 else None

                if file_path and best_path is None:
                    best_path = file_path

                if size_val is not None:
                    total_size += size_val
                    size_seen = True

        if best_path is None:
            return None, None

        return best_path, (total_size if (size_seen and total_size > 0) else None)

    except Exception as exc:
        _log_debug(f"get_movie_file_info failed: {exc!r}")
        return None, None


# ============================================================
#        UTILIDADES PARA EXTRAER IMDB ID DESDE GUIDS PLEX
# ============================================================


def get_imdb_id_from_plex_guid(guid: str) -> str | None:
    """
    Intenta extraer un imdb_id (tt1234567) desde un guid de Plex.

    Ejemplos:
        'com.plexapp.agents.imdb://tt0111161?lang=en'
        'com.plexapp.agents.themoviedb://12345?lang=en'
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
    Intenta obtener un imdb_id (tt...) usando información de Plex:

    1) Recorre movie.guids (si existe).
    2) Fallback: guid principal (movie.guid).
    """
    try:
        guids = getattr(movie, "guids", None) or []
        if isinstance(guids, Sequence):
            for g in guids:
                gid = getattr(g, "id", None)
                if isinstance(gid, str):
                    imdb_id = get_imdb_id_from_plex_guid(gid)
                    if imdb_id:
                        return imdb_id
    except Exception:
        pass

    guid_main = getattr(movie, "guid", None)
    if isinstance(guid_main, str):
        return get_imdb_id_from_plex_guid(guid_main)

    return None


# ============================================================
#              OBTENER TÍTULO MÁS FIABLE PARA OMDb
# ============================================================


def get_best_search_title(movie: object) -> str:
    """
    Devuelve el mejor título estimado para buscar en OMDb.

    Preferimos `originalTitle` si existe y no está vacío, luego `title`.
    Siempre devolvemos una cadena (posiblemente vacía).
    """
    title_obj = getattr(movie, "originalTitle", None)
    if isinstance(title_obj, str) and title_obj.strip():
        return title_obj.strip()

    fallback_obj = getattr(movie, "title", None)
    if isinstance(fallback_obj, str):
        return fallback_obj.strip()

    return ""