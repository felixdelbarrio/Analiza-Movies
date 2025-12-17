# backend/plex_client.py
from __future__ import annotations

from typing import Any, Optional, Tuple

from plexapi.server import PlexServer

from backend.config import BASEURL, PLEX_TOKEN, PLEX_PORT, EXCLUDE_PLEX_LIBRARIES
from backend import logger as _logger


# ============================================================
#                  LOGGING CONTROLADO
# ============================================================


def _log(msg: str) -> None:
    """Logea vía logger central con fallback ligero.

    Delegamos en el logger central; si por alguna razón fallara,
    hacemos un print directo para no perder información.
    """
    try:
        _logger.info(msg)
    except Exception:
        print(msg)


# ============================================================
#                     CONEXIÓN A PLEX
# ============================================================


def _build_plex_base_url() -> str:
    """
    Construye la URL base para Plex a partir de BASEURL (host sin puerto)
    y PLEX_PORT.

    Ejemplo:
        BASEURL = "http://192.168.1.10"
        PLEX_PORT = 32400
        → "http://192.168.1.10:32400"
    """
    if not BASEURL:
        raise RuntimeError("BASEURL no está definido en el entorno (.env)")

    base = BASEURL.rstrip("/")  # por si viene con barra final
    return f"{base}:{PLEX_PORT}"


def connect_plex() -> PlexServer:
    """Crea y devuelve una instancia de `PlexServer` usando la configuración.

    Lanza RuntimeError si faltan variables de entorno o si la conexión falla.
    """
    if not BASEURL or not PLEX_TOKEN:
        raise RuntimeError("Faltan BASEURL o PLEX_TOKEN en el .env")

    base_url = _build_plex_base_url()
    _log(f"Conectando a Plex en {base_url} ...")
    try:
        plex = PlexServer(base_url, PLEX_TOKEN)
    except Exception as e:
        _log(f"ERROR conectando a Plex: {e}")
        raise
    _log("Conectado a Plex.")
    return plex


def get_libraries_to_analyze(plex: PlexServer) -> list[object]:
    """Devuelve la lista de bibliotecas de Plex a analizar.

    Filtra aquellas cuyo título esté incluido en
    `EXCLUDE_PLEX_LIBRARIES` definido en `backend.config`.
    """
    libraries: list[object] = []
    try:
        sections = plex.library.sections()
    except Exception as exc:  # pragma: no cover
        _log(f"ERROR obteniendo secciones de Plex: {exc}")
        return libraries

    for section in sections:
        name = getattr(section, "title", "") or ""
        if name in EXCLUDE_PLEX_LIBRARIES:
            _log(f"Saltando biblioteca Plex excluida: {name}")
            continue
        libraries.append(section)

    return libraries


# ============================================================
#                  INFO DE ARCHIVOS DE PELÍCULAS
# ============================================================


def get_movie_file_info(movie: Any) -> Tuple[Optional[str], Optional[int]]:
    """Devuelve (ruta_principal, tamaño_total_en_bytes) para una película de Plex.

    Reglas:
      - Si no hay media o parts válidos → (None, None).
      - La ruta devuelta es el `file` del primer part válido encontrado.
      - El tamaño es la suma de los tamaños (`size`) de todos los parts válidos.

    Es defensiva: si la estructura del objeto Plex cambia o faltan atributos,
    devuelve (None, None) en lugar de lanzar excepción.
    """
    try:
        media_seq = getattr(movie, "media", None)
        if not media_seq:
            return None, None

        best_path: Optional[str] = None
        total_size: int = 0

        for media in media_seq:
            parts = getattr(media, "parts", None) or []
            for part in parts:
                file_path = getattr(part, "file", None)
                size_attr = getattr(part, "size", None)

                if isinstance(file_path, str) and file_path and isinstance(size_attr, int):
                    if best_path is None:
                        best_path = file_path
                    total_size += size_attr

        if best_path is None:
            return None, None

        # Si por alguna razón total_size quedó en 0 (partes raras),
        # mejor devolver None en size.
        return best_path, total_size if total_size > 0 else None

    except Exception as e:
        _log(f"ERROR obteniendo info de archivo para película Plex: {e}")
        return None, None


# ============================================================
#        UTILIDADES PARA EXTRAER IMDB ID DESDE GUIDS PLEX
# ============================================================


def get_imdb_id_from_plex_guid(guid: str) -> Optional[str]:
    """Intenta extraer un imdb_id (tt1234567) desde un guid de Plex.

    Ejemplos de guid típico:
        'com.plexapp.agents.imdb://tt0111161?lang=en'
        'com.plexapp.agents.themoviedb://12345?lang=en'
    """
    if "imdb://" not in guid:
        return None

    try:
        after = guid.split("imdb://", 1)[1]
        imdb_id = after.split("?", 1)[0]
        return imdb_id or None
    except Exception:
        return None


def get_imdb_id_from_movie(movie: Any) -> Optional[str]:
    """
    Intenta obtener un imdb_id (tt...) usando toda la información de Plex:

    1) Recorre movie.guids para buscar tt...
    2) Si no encuentra nada, cae al guid principal (movie.guid).
    """
    # 1) Revisar todos los GUIDs individuales
    try:
        guids = getattr(movie, "guids", None) or []
        for g in guids:
            gid = getattr(g, "id", None)
            if isinstance(gid, str):
                imdb_id = get_imdb_id_from_plex_guid(gid)
                if imdb_id:
                    return imdb_id
    except Exception:
        pass

    # 2) Fallback: guid principal
    guid = getattr(movie, "guid", None)
    if isinstance(guid, str):
        return get_imdb_id_from_plex_guid(guid or "")
    return None


# ============================================================
#              OBTENER TÍTULO MÁS FIABLE PARA OMDb
# ============================================================


def get_best_search_title(movie: Any) -> str:
    """Devuelve el mejor título estimado para buscar en OMDb.

    Preferimos `originalTitle` si existe y no está vacío, luego `title`.
    Siempre devolvemos una cadena (posiblemente vacía) para evitar
    excepciones en código que llama a esta función.
    """
    title = getattr(movie, "originalTitle", None)
    if isinstance(title, str) and title.strip():
        return title.strip()

    fallback = getattr(movie, "title", None)
    if isinstance(fallback, str):
        return fallback.strip()

    return ""