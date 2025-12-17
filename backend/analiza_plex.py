from __future__ import annotations

"""
analiza_plex.py

Orquestador principal de análisis Plex.
"""

from backend.movie_analyzer import analyze_single_movie
from backend.plex_client import (
    connect_plex,
    get_libraries_to_analyze,
)
from backend import logger as _logger
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv
from backend.decision_logic import sort_filtered_rows
from backend.config import (
    OUTPUT_PREFIX,
    METADATA_OUTPUT_PREFIX,
    EXCLUDE_PLEX_LIBRARIES,
)


def analyze_all_libraries() -> None:
    """Analiza todas las bibliotecas Plex aplicando EXCLUDE_PLEX_LIBRARIES."""
    plex = connect_plex()
    libraries = get_libraries_to_analyze(plex)

    all_rows: list[dict[str, object]] = []
    suggestion_rows: list[dict[str, object]] = []

    for library in libraries:
        lib_name = getattr(library, "title", "")

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
            row, meta_sugg, logs = analyze_single_movie(movie)

            # Mostrar logs generados por el analizador de forma controlada
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
    # Salida CSV
    # ---------------------------------------------------
    write_all_csv(f"{OUTPUT_PREFIX}_plex_all.csv", all_rows)
    write_filtered_csv(f"{OUTPUT_PREFIX}_plex_filtered.csv", filtered)
    write_suggestions_csv(f"{METADATA_OUTPUT_PREFIX}_plex.csv", suggestion_rows)

    _logger.info("[PLEX] Análisis completado.")