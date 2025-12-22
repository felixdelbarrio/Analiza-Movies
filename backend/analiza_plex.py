from __future__ import annotations

"""
analiza_plex.py

Orquestador principal de análisis Plex.

Objetivos de salida por consola:
- SILENT_MODE=True:
    - Evitar logs detallados
    - Mantener señales mínimas de progreso (biblioteca actual + resúmenes)
- SILENT_MODE=False:
    - Mostrar progreso por película (i/total) en consola
    - En modo normal, mostrar también año si existe
- DEBUG_MODE=True:
    - Permitir más visibilidad (heartbeat cada N elementos, más contexto)
    - En modo normal, añadir extra útil por película (p.ej. tamaño si se conoce)
"""

import time
from typing import Any, Iterable

from backend import logger as _logger
from backend.collection_analysis import analyze_movie
from backend.config import (
    DEBUG_MODE,
    EXCLUDE_PLEX_LIBRARIES,
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
    SILENT_MODE,
)
from backend.decision_logic import sort_filtered_rows
from backend.movie_input import MovieInput
from backend.plex_client import (
    connect_plex,
    get_best_search_title,
    get_imdb_id_from_movie,
    get_libraries_to_analyze,
    get_movie_file_info,
)
from backend.reporting import write_all_csv, write_filtered_csv, write_suggestions_csv


# ============================================================================
# CONFIG: Idioma por librería
# ============================================================================
_PLEX_LIBRARY_LANGUAGE_DEFAULT: str = "es"

_PLEX_LIBRARY_LANGUAGE_BY_NAME: dict[str, str] = {
    # "Animación 2D": "es",
    # "Animación 3D": "es",
    # "Movies": "es",
}


# Heartbeat: solo para dar señales en ejecuciones largas.
# Se usa únicamente cuando (SILENT_MODE=True y DEBUG_MODE=True)
_PROGRESS_EVERY_N_MOVIES: int = 100


def _get_plex_library_language(lib_name: str) -> str:
    """
    Resuelve el idioma objetivo para una biblioteca Plex.

    Se inyecta en MovieInput.extra["library_language"] para que el pipeline
    de sugerencias/metadata pueda usarlo.
    """
    lang = _PLEX_LIBRARY_LANGUAGE_BY_NAME.get(lib_name)
    return lang or _PLEX_LIBRARY_LANGUAGE_DEFAULT


def _library_title(library: Any) -> str:
    """
    Obtiene un título seguro de biblioteca Plex.

    PlexAPI no siempre expone tipos estáticos (depende del entorno),
    por eso usamos getattr defensivo.
    """
    return (getattr(library, "title", "") or "").strip()


def _library_total_items(library: Any) -> int | None:
    """
    Intenta obtener el total de items de la biblioteca sin materializar el listado.

    Muchos objetos de sección en PlexAPI exponen `totalSize`, pero no es universal.
    Si no está disponible o no es un int válido, devuelve None.
    """
    raw = getattr(library, "totalSize", None)
    return raw if isinstance(raw, int) and raw >= 0 else None


def _iter_movies_with_total(
    library: Any,
) -> tuple[Iterable[Any], int | None]:
    """
    Devuelve:
      - iterable de películas
      - total (si se puede determinar) o None

    Estrategia:
      - Si SILENT_MODE=True: no necesitamos total; iteramos directo (eficiente).
      - Si SILENT_MODE=False:
          * intentamos totalSize (sin coste)
          * si no existe y DEBUG_MODE=True: materializamos list(...) para total exacto
          * si no existe y DEBUG_MODE=False: no penalizamos → total None
    """
    if SILENT_MODE:
        return library.search(), None

    total = _library_total_items(library)
    if total is not None:
        return library.search(), total

    if DEBUG_MODE:
        movies = list(library.search())
        return movies, len(movies)

    return library.search(), None


def _format_progress_prefix(index: int, total: int | None) -> str:
    """
    Formatea el prefijo de progreso por película.

    Ejemplos:
      - total conocido: "(15/100)"
      - total desconocido: "(15/?)"
    """
    if total is None:
        return f"({index}/?)"
    return f"({index}/{total})"


def _format_human_size(num_bytes: int) -> str:
    """
    Formatea bytes a unidades humanas (KiB, MiB, GiB, TiB).

    Nota:
    - Se usa solo para output informativo (no afecta lógica).
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


def _format_movie_progress_line(
    *,
    index: int,
    total: int | None,
    title: str,
    year: int | None,
    file_size_bytes: int | None,
) -> str:
    """
    Construye la línea que se imprime en modo no-silent por película.

    Reglas:
      - Siempre incluye prefijo (i/total) + título
      - Si hay year, lo añade: "Título (2009)"
      - Si DEBUG_MODE y hay tamaño, añade: "[1.4 GiB]"
    """
    prefix = _format_progress_prefix(index, total)

    base = title.strip() or "UNKNOWN"
    if year is not None:
        base = f"{base} ({year})"

    if DEBUG_MODE and file_size_bytes is not None and file_size_bytes >= 0:
        base = f"{base} [{_format_human_size(file_size_bytes)}]"

    return f"{prefix} {base}"


def analyze_all_libraries() -> None:
    """
    Analiza todas las bibliotecas Plex aplicando EXCLUDE_PLEX_LIBRARIES.

    Salidas:
    - report_all.csv
    - report_filtered.csv
    - metadata_fix.csv
    """
    t0 = time.monotonic()

    plex = connect_plex()
    raw_libraries = get_libraries_to_analyze(plex)

    # Pre-filtrado para:
    # - calcular (i/n) real
    # - evitar “saltos” confusos en el progreso
    libraries: list[Any] = []
    excluded: list[str] = []
    for lib in raw_libraries:
        name = _library_title(lib)
        if name and name in EXCLUDE_PLEX_LIBRARIES:
            excluded.append(name)
            continue
        libraries.append(lib)

    total_libs = len(libraries)

    # En silent, informar de exclusiones ayuda a entender por qué (i/n) es menor
    if SILENT_MODE and excluded:
        _logger.progress(
            "[PLEX] Bibliotecas excluidas por configuración: "
            + ", ".join(sorted(excluded))
        )

    if total_libs == 0:
        _logger.progress("[PLEX] No hay bibliotecas para analizar (0).")
        return

    # Acumuladores globales
    all_rows: list[dict[str, object]] = []
    suggestion_rows: list[dict[str, object]] = []

    total_movies_processed = 0
    total_movies_errors = 0

    # -------------------------------------------------------------------------
    # Iteración por bibliotecas
    # -------------------------------------------------------------------------
    for lib_index, library in enumerate(libraries, start=1):
        lib_name = _library_title(library)

        # Señal de progreso SIEMPRE visible (aunque SILENT_MODE=True)
        if lib_name:
            _logger.progress(f"[PLEX] ({lib_index}/{total_libs}) {lib_name}")
        else:
            _logger.progress(f"[PLEX] ({lib_index}/{total_libs}) <sin nombre>")

        # En modo no-silent mantenemos el log tradicional
        _logger.info(f"Analizando biblioteca Plex: {lib_name}")

        # Idioma decidido a nivel de librería Plex
        library_language = _get_plex_library_language(lib_name)

        if SILENT_MODE and DEBUG_MODE:
            _logger.progress(
                f"[PLEX][DEBUG] library_language={library_language!r} "
                f"excluded={len(excluded)}"
            )

        # Contadores por biblioteca
        lib_movies_processed = 0
        lib_movies_errors = 0
        lib_rows_added = 0
        lib_suggestions_added = 0

        t_lib = time.monotonic()

        movies_iter, total_movies_in_library = _iter_movies_with_total(library)

        # ---------------------------------------------------------------------
        # Iteración por items de la biblioteca
        # ---------------------------------------------------------------------
        for movie_index, movie in enumerate(movies_iter, start=1):
            title = getattr(movie, "title", "") or ""
            year_value = getattr(movie, "year", None)
            year: int | None = year_value if isinstance(year_value, int) else None

            guid = getattr(movie, "guid", None)
            thumb = getattr(movie, "thumb", None)

            file_path, file_size = get_movie_file_info(movie)

            # En modo no-silent mostramos progreso por película:
            #  - (i/total) Título (Año) [Tamaño]  (tamaño solo en DEBUG_MODE)
            if not SILENT_MODE:
                _logger.info(
                    _format_movie_progress_line(
                        index=movie_index,
                        total=total_movies_in_library,
                        title=title,
                        year=year,
                        file_size_bytes=file_size,
                    )
                )

            rating_key_raw = getattr(movie, "ratingKey", None)
            rating_key: str | None = (
                str(rating_key_raw) if rating_key_raw is not None else None
            )

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
                    # Clave: permite que sugerencias/metadata apliquen idioma por librería
                    "library_language": library_language,
                },
            )

            # Heartbeat en silencioso + debug: evita la sensación de “está colgado”
            if SILENT_MODE and DEBUG_MODE and (movie_index % _PROGRESS_EVERY_N_MOVIES == 0):
                _logger.progress(
                    f"[PLEX][DEBUG] {lib_name}: procesadas {movie_index} películas..."
                )

            try:
                row, meta_sugg, logs = analyze_movie(
                    movie_input,
                    source_movie=movie,
                )
            except Exception as exc:
                lib_movies_errors += 1
                total_movies_errors += 1
                _logger.error(
                    f"[PLEX] Error analizando '{title}' ({year or 'n/a'}) "
                    f"en '{lib_name}': {exc!r}"
                )
                continue

            for log in logs:
                _logger.info(log)

            if row:
                all_rows.append(row)
                lib_rows_added += 1

            if meta_sugg:
                suggestion_rows.append(meta_sugg)
                lib_suggestions_added += 1

            lib_movies_processed += 1
            total_movies_processed += 1

        # ---------------------------------------------------------------------
        # Resumen por biblioteca
        # ---------------------------------------------------------------------
        t_lib_elapsed = time.monotonic() - t_lib

        if SILENT_MODE:
            _logger.progress(
                "[PLEX] Biblioteca finalizada: "
                f"{lib_name} | movies={lib_movies_processed} | "
                f"errors={lib_movies_errors} | rows={lib_rows_added} | "
                f"suggestions={lib_suggestions_added} | "
                f"time={t_lib_elapsed:.1f}s"
            )
        elif DEBUG_MODE:
            _logger.info(
                "[PLEX][DEBUG] Biblioteca finalizada: "
                f"{lib_name} movies={lib_movies_processed} "
                f"errors={lib_movies_errors} rows={lib_rows_added} "
                f"suggestions={lib_suggestions_added} time={t_lib_elapsed:.1f}s"
            )

    # -------------------------------------------------------------------------
    # Filtrado y ordenación final
    # -------------------------------------------------------------------------
    filtered = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered = sort_filtered_rows(filtered)

    # -------------------------------------------------------------------------
    # Salida CSV
    # -------------------------------------------------------------------------
    write_all_csv(REPORT_ALL_PATH, all_rows)
    write_filtered_csv(REPORT_FILTERED_PATH, filtered)
    write_suggestions_csv(METADATA_FIX_PATH, suggestion_rows)

    elapsed = time.monotonic() - t0

    if SILENT_MODE:
        _logger.progress(
            "[PLEX] Análisis completado. "
            f"libraries={total_libs} movies={total_movies_processed} "
            f"errors={total_movies_errors} all_rows={len(all_rows)} "
            f"filtered_rows={len(filtered)} suggestions={len(suggestion_rows)} "
            f"time={elapsed:.1f}s"
        )

    _logger.info("[PLEX] Análisis completado.")