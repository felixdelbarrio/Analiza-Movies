from __future__ import annotations

"""analiza_dlna.py

Flujo de análisis para contenidos obtenidos desde una fuente tipo DLNA
(o, de forma simplificada, desde un árbol de directorios local).

Este script:
  1. Pregunta por un directorio raíz.
  2. Busca ficheros de vídeo de forma recursiva.
  3. Para cada fichero construye un MovieInput.
  4. Usa el core genérico `analyze_input_movie` para obtener una fila base.
  5. Enriquece la fila con algunos campos adicionales.
  6. Escribe:
       - un CSV completo (todas las filas)
       - un CSV filtrado (DELETE / MAYBE)
       - un CSV de sugerencias de metadata vacío (para compatibilidad con dashboard)
"""

import json
import os
from pathlib import Path

from backend import logger as _logger
from backend.config import (
    OUTPUT_PREFIX,
    METADATA_OUTPUT_PREFIX,
    EXCLUDE_DLNA_LIBRARIES,
)
from backend.decision_logic import sort_filtered_rows
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv
from backend.movie_input import MovieInput
from backend.analyze_input_core import analyze_input_movie
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


def _is_video_file(path: Path) -> bool:
    """Devuelve True si el Path apunta a un fichero de vídeo soportado."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _guess_title_year(file_path: Path) -> tuple[str, int | None]:
    """Intenta inferir título y año a partir del nombre del fichero.

    Estrategia muy simple y defensiva:
      - Si el nombre contiene ' (YYYY)' usamos esa parte.
      - Si contiene '.YYYY.' donde YYYY parece un año, lo usamos.
      - En caso contrario, devuelve el stem completo como título y year=None.
    """
    stem = file_path.stem
    title = stem
    year: int | None = None

    # Patrón 1: Título (YYYY)
    if "(" in stem and ")" in stem:
        before, _, after = stem.partition("(")
        maybe_year, _, _ = after.partition(")")
        maybe_year = maybe_year.strip()
        if len(maybe_year) == 4 and maybe_year.isdigit():
            year_int = int(maybe_year)
            if 1900 <= year_int <= 2100:
                title = before.strip()
                year = year_int
                return title, year

    # Patrón 2: Título.YYYY.algo
    parts = stem.split(".")
    for part in parts:
        if len(part) == 4 and part.isdigit():
            year_int = int(part)
            if 1900 <= year_int <= 2100:
                year = year_int
                break

    return title.strip(), year


def _ask_root_directory() -> Path:
    """Pregunta al usuario por el directorio raíz a analizar."""
    while True:
        raw = input("Ruta del directorio raíz a analizar (DLNA/local): ").strip()
        if not raw:
            _logger.warning("Debes introducir una ruta no vacía.", always=True)
            continue

        path = Path(raw).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            _logger.error(
                f"La ruta {path} no existe o no es un directorio.",
                always=True,
            )
            continue

        return path


def _iter_video_files(root: Path) -> list[Path]:
    """Devuelve una lista de ficheros de vídeo bajo el directorio raíz."""
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        dirp = Path(dirpath)
        for name in filenames:
            candidate = dirp / name
            if _is_video_file(candidate):
                files.append(candidate)
    return files


def analyze_dlna_server() -> None:
    """Flujo principal de análisis para una fuente DLNA/local."""
    root = _ask_root_directory()
    library = root.name  # etiqueta simple de biblioteca (nombre del directorio raíz)

    # -------------------------------------------------
    # Respetar EXCLUDE_DLNA_LIBRARIES
    # -------------------------------------------------
    if library in EXCLUDE_DLNA_LIBRARIES:
        _logger.info(
            f"[DLNA] La biblioteca '{library}' está en EXCLUDE_DLNA_LIBRARIES; "
            "se omite el análisis.",
            always=True,
        )
        return

    files = _iter_video_files(root)
    if not files:
        _logger.info(
            f"No se han encontrado ficheros de vídeo en {root}",
            always=True,
        )
        return

    _logger.info(
        f"Analizando {len(files)} ficheros de vídeo bajo {root}",
        always=True,
    )

    all_rows: list[dict[str, object]] = []
    suggestions_rows: list[dict[str, object]] = []

    # -------------------------------------------------
    # Bucle principal de análisis por fichero
    # -------------------------------------------------
    for file_path in files:
        title, year = _guess_title_year(file_path)

        # `library` es el mismo para todos los ficheros (nombre del root)
        try:
            stat = file_path.stat()
            file_size: int | None = stat.st_size
        except OSError:
            file_size = None

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
            file_path=str(file_path),
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
            _logger.error(
                f"[DLNA] Error analizando {file_path}: {exc}",
                always=True,
            )
            continue

        if not base_row:
            _logger.warning(
                f"[DLNA] analyze_input_movie devolvió fila vacía para {file_path}",
                always=True,
            )
            continue

        # Enriquecemos la fila con campos adicionales usados por reporting.
        row: dict[str, object] = dict(base_row)
        row["file"] = str(file_path)

        # file_size_bytes -> file_size (por compatibilidad con dashboard)
        file_size_bytes = row.get("file_size_bytes")
        if isinstance(file_size_bytes, int):
            row["file_size"] = file_size_bytes
        else:
            row["file_size"] = file_size

        # Campos adicionales desde OMDb/Wiki
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
        _logger.info(
            "No se han generado filas de análisis para DLNA.",
            always=True,
        )
        return

    # Filtrado DELETE/MAYBE + ordenación
    filtered_rows = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered_rows = sort_filtered_rows(filtered_rows) if filtered_rows else []

    # Salidas
    all_path = f"{OUTPUT_PREFIX}_dlna_all.csv"
    filtered_path = f"{OUTPUT_PREFIX}_dlna_filtered.csv"
    suggestions_path = f"{METADATA_OUTPUT_PREFIX}_dlna.csv"

    write_all_csv(all_path, all_rows)
    write_filtered_csv(filtered_path, filtered_rows)
    # Por ahora no generamos sugerencias para DLNA, pero escribimos CSV vacío compatible
    write_suggestions_csv(suggestions_path, suggestions_rows)

    _logger.info(
        f"[DLNA] Análisis completado. CSV completo: {all_path} | "
        f"CSV filtrado: {filtered_path}",
        always=True,
    )


if __name__ == "__main__":
    analyze_dlna_server()