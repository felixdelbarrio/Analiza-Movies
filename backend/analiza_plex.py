from __future__ import annotations

"""
analiza_plex.py

Orquestador principal de análisis Plex.
"""

from backend import logger as _logger
from backend.collection_analysis import analyze_movie
from backend.config import (
    EXCLUDE_PLEX_LIBRARIES,
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from backend.decision_logic import sort_filtered_rows
from backend.plex_client import (
    connect_plex,
    get_best_search_title,
    get_imdb_id_from_movie,
    get_libraries_to_analyze,
    get_movie_file_info,
)
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv
from backend.movie_input import MovieInput


def analyze_all_libraries() -> None:
    """Analiza todas las bibliotecas Plex aplicando EXCLUDE_PLEX_LIBRARIES."""
    plex = connect_plex()
    libraries = get_libraries_to_analyze(plex)

    all_rows: list[dict[str, object]] = []
    suggestion_rows: list[dict[str, object]] = []

    for library in libraries:
        lib_name = getattr(library, "title", "") or ""

        # ---------------------------------------------------
        # Respetar EXCLUDE_PLEX_LIBRARIES
        # ---------------------------------------------------
        if lib_name in EXCLUDE_PLEX_LIBRARIES:
            _logger.info(
                f"[PLEX] Biblioteca excluida por configuración: {lib_name}",
                always=True,
            )
            continue

        _logger.info(f"Analizando biblioteca Plex: {lib_name}")

        for movie in library.search():
            title = getattr(movie, "title", "") or ""
            year_value = getattr(movie, "year", None)
            year: int | None = year_value if isinstance(year_value, int) else None

            rating_key_raw = getattr(movie, "ratingKey", None)
            rating_key: str | None = (
                str(rating_key_raw) if rating_key_raw is not None else None
            )

            guid = getattr(movie, "guid", None)
            thumb = getattr(movie, "thumb", None)

            file_path, file_size = get_movie_file_info(movie)

            imdb_id_hint = get_imdb_id_from_movie(movie)
            search_title = get_best_search_title(movie) or title

            movie_input = MovieInput(
                source="plex",
                library=lib_name,
                title=search_title,
                year=year,
                file_path=file_path or "",
                file_size_bytes=file_size,
                imdb_id_hint=imdb_id_hint,
                plex_guid=guid,
                rating_key=rating_key,
                thumb_url=thumb,
                extra={
                    # Precedencia: Plex manda para reportar
                    "display_title": title,
                    "display_year": year,
                },
            )

            row, meta_sugg, logs = analyze_movie(
                movie_input,
                source_movie=movie,
            )

            for log in logs:
                _logger.info(log)

            if row:
                all_rows.append(row)

            if meta_sugg:
                suggestion_rows.append(meta_sugg)

    # ---------------------------------------------------
    # Filtrado y ordenación final
    # ---------------------------------------------------
    filtered = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered = sort_filtered_rows(filtered)

    # ---------------------------------------------------
    # Salida CSV (rutas estándar en /reports)
    # ---------------------------------------------------
    write_all_csv(REPORT_ALL_PATH, all_rows)
    write_filtered_csv(REPORT_FILTERED_PATH, filtered)
    write_suggestions_csv(METADATA_FIX_PATH, suggestion_rows)

    _logger.info("[PLEX] Análisis completado.")