from __future__ import annotations

"""
analiza_plex.py

Orquestador principal de análisis Plex (modo streaming).

Cambios clave (streaming):
- report_all.csv se escribe incrementalmente (open_all_csv_writer) => no se acumula all_rows.
- metadata_fix.csv se escribe incrementalmente (open_suggestions_csv_writer) => no se acumula suggestion_rows.
- Solo se mantiene en memoria:
    * filtered_rows (DELETE/MAYBE) para ordenarlo al final (si hay)
    * contadores de decisiones para el resumen final
- En modo NO SILENT:
    * se conserva orden estable por biblioteca (buffer solo de esa biblioteca).

Notas de consola (alineado con backend/logger.py):
- SILENT_MODE=True:
    * progreso mínimo con _logger.progress(...)
    * resumen final compacto + explícito sobre si se generó report_filtered.csv
- DEBUG_MODE=True:
    * métricas/diagnóstico extra sin “spamear”
"""

import time
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from backend import logger as _logger
from backend.collection_analysis import analyze_movie
from backend.config import (
    DEBUG_MODE,
    EXCLUDE_PLEX_LIBRARIES,
    METADATA_FIX_PATH,
    OMDB_HTTP_MAX_CONCURRENCY,
    OMDB_HTTP_MIN_INTERVAL_SECONDS,
    PLEX_ANALYZE_WORKERS,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
    SILENT_MODE,
)
from backend.decision_logic import sort_filtered_rows
from backend.movie_input import MovieInput
from backend.omdb_client import get_omdb_metrics_snapshot, reset_omdb_metrics
from backend.plex_client import (
    connect_plex,
    get_best_search_title,
    get_imdb_id_from_movie,
    get_libraries_to_analyze,
    get_movie_file_info,
)
from backend.reporting import (
    open_all_csv_writer,
    open_filtered_csv_writer_only_if_rows,
    open_suggestions_csv_writer,
)

# ============================================================================
# CONFIG: Idioma por librería
# ============================================================================
_PLEX_LIBRARY_LANGUAGE_DEFAULT: str = "es"
_PLEX_LIBRARY_LANGUAGE_BY_NAME: dict[str, str] = {
    # "Animación 2D": "es",
    # "Animación 3D": "es",
    # "Movies": "es",
}

_PROGRESS_EVERY_N_MOVIES: int = 100
_MAX_WORKERS_CAP: int = 64


# --------------------------
# OMDb metrics helpers
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


def _rank_top(deltas_by_lib: dict[str, dict[str, int]], key: str, top_n: int = 5) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = [(lib, int(d.get(key, 0))) for lib, d in deltas_by_lib.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _compute_cost_score(delta: dict[str, int]) -> int:
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


def _rank_top_by_total_cost(deltas_by_lib: dict[str, dict[str, int]], top_n: int = 5) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = [(lib, _compute_cost_score(d)) for lib, d in deltas_by_lib.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _log_omdb_rankings(deltas_by_lib: dict[str, dict[str, int]], *, min_groups: int = 2) -> None:
    if not (SILENT_MODE and DEBUG_MODE):
        return
    if not deltas_by_lib or len(deltas_by_lib) < min_groups:
        return

    top_cost = _rank_top_by_total_cost(deltas_by_lib, top_n=5)
    top_http = _rank_top(deltas_by_lib, "http_requests", top_n=5)
    top_rls = _rank_top(deltas_by_lib, "rate_limit_sleeps", top_n=5)
    top_fail = _rank_top(deltas_by_lib, "http_failures", top_n=5)
    top_rlh = _rank_top(deltas_by_lib, "rate_limit_hits", top_n=5)

    def _fmt(items: list[tuple[str, int]]) -> str:
        usable = [(name, val) for (name, val) in items if val > 0]
        return " | ".join([f"{i+1}) {name}: {val}" for i, (name, val) in enumerate(usable)])

    if (line := _fmt(top_cost)):
        _logger.progress(f"[PLEX][DEBUG] Top libraries by TOTAL_COST: {line}")
    if (line := _fmt(top_http)):
        _logger.progress(f"[PLEX][DEBUG] Top libraries by http_requests: {line}")
    if (line := _fmt(top_rls)):
        _logger.progress(f"[PLEX][DEBUG] Top libraries by rate_limit_sleeps: {line}")
    if (line := _fmt(top_fail)):
        _logger.progress(f"[PLEX][DEBUG] Top libraries by http_failures: {line}")
    if (line := _fmt(top_rlh)):
        _logger.progress(f"[PLEX][DEBUG] Top libraries by rate_limit_hits: {line}")


# ============================================================================
# WORKERS (conectado a OMDb limiter)
# ============================================================================
def _compute_max_workers(requested: int, total_work_items: int | None) -> int:
    max_workers = int(requested)

    if max_workers < 1:
        max_workers = 1
    if max_workers > _MAX_WORKERS_CAP:
        max_workers = _MAX_WORKERS_CAP

    omdb_cap = max(4, int(OMDB_HTTP_MAX_CONCURRENCY) * 8)
    max_workers = min(max_workers, omdb_cap)

    if total_work_items is not None and total_work_items > 0:
        max_workers = min(max_workers, total_work_items)

    if max_workers < 1:
        max_workers = 1

    return max_workers


# ============================================================================
# UTILIDADES PLEX (idioma / safe access)
# ============================================================================
def _get_plex_library_language(lib_name: str) -> str:
    return _PLEX_LIBRARY_LANGUAGE_BY_NAME.get(lib_name) or _PLEX_LIBRARY_LANGUAGE_DEFAULT


def _library_title(library: Any) -> str:
    return (getattr(library, "title", "") or "").strip()


def _library_total_items(library: Any) -> int | None:
    raw = getattr(library, "totalSize", None)
    return raw if isinstance(raw, int) and raw >= 0 else None


def _iter_movies_with_total(library: Any) -> tuple[Iterable[Any], int | None]:
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
    return f"({index}/{total})" if total is not None else f"({index}/?)"


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


def _format_movie_progress_line(
    *,
    index: int,
    total: int | None,
    title: str,
    year: int | None,
    file_size_bytes: int | None,
) -> str:
    prefix = _format_progress_prefix(index, total)
    base = title.strip() or "UNKNOWN"
    if year is not None:
        base = f"{base} ({year})"
    if DEBUG_MODE and file_size_bytes is not None and file_size_bytes >= 0:
        base = f"{base} [{_format_human_size(file_size_bytes)}]"
    return f"{prefix} {base}"


# ============================================================================
# ORQUESTACIÓN PRINCIPAL
# ============================================================================
def analyze_all_libraries() -> None:
    t0 = time.monotonic()

    reset_omdb_metrics()

    plex = connect_plex()
    raw_libraries = get_libraries_to_analyze(plex)

    libraries: list[Any] = []
    excluded: list[str] = []
    for lib in raw_libraries:
        name = _library_title(lib)
        if name and name in EXCLUDE_PLEX_LIBRARIES:
            excluded.append(name)
            continue
        libraries.append(lib)

    total_libs = len(libraries)

    if SILENT_MODE and excluded:
        _logger.progress("[PLEX] Bibliotecas excluidas por configuración: " + ", ".join(sorted(excluded)))

    if total_libs == 0:
        _logger.progress("[PLEX] No hay bibliotecas para analizar (0).")
        return

    # Streaming outputs
    filtered_rows: list[dict[str, object]] = []

    # Resumen final sin all_rows
    decisions_count: dict[str, int] = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}

    total_movies_processed = 0  # = futures completados (con o sin row)
    total_movies_errors = 0
    total_rows_written = 0
    total_suggestions_written = 0

    max_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS, total_work_items=None)

    if SILENT_MODE:
        _logger.progress(
            f"[PLEX] ThreadPool workers={max_workers} (por biblioteca) "
            f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, "
            f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
            f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
        )
    else:
        _logger.debug_ctx(
            "PLEX",
            f"ThreadPool workers={max_workers} (por biblioteca) "
            f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, cap por OMDb limiter)",
        )

    lib_omdb_delta_prepare: dict[str, dict[str, int]] = {}
    lib_omdb_delta_analyze: dict[str, dict[str, int]] = {}

    def _maybe_print_movie_logs(logs: list[str]) -> None:
        if not logs:
            return
        if not SILENT_MODE:
            for line in logs:
                _logger.info(line)
            return
        if DEBUG_MODE:
            for line in logs:
                _logger.info(line, always=True)

    def _tally_decision(row: Mapping[str, object]) -> None:
        d = row.get("decision")
        if d in ("KEEP", "MAYBE", "DELETE"):
            decisions_count[str(d)] += 1
        else:
            decisions_count["UNKNOWN"] += 1

    # Abrimos writers UNA VEZ (streaming global).
    with open_all_csv_writer(REPORT_ALL_PATH) as all_writer, open_suggestions_csv_writer(
        METADATA_FIX_PATH
    ) as sugg_writer:
        for lib_index, library in enumerate(libraries, start=1):
            lib_name = _library_title(library)
            lib_key = lib_name or f"<lib_{lib_index}>"

            _logger.progress(f"[PLEX] ({lib_index}/{total_libs}) {lib_name or '<sin nombre>'}")
            if not SILENT_MODE:
                _logger.info(f"Analizando biblioteca Plex: {lib_name}")

            library_language = _get_plex_library_language(lib_name)

            _logger.debug_ctx(
                "PLEX",
                f"library_language={library_language!r} excluded={len(excluded)} "
                f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY} "
                f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS}",
            )

            lib_movies_enqueued = 0
            lib_movies_completed = 0
            lib_movies_errors = 0
            lib_rows_written = 0
            lib_suggestions_written = 0

            t_lib = time.monotonic()
            movies_iter, total_movies_in_library = _iter_movies_with_total(library)

            prepare_snap_start: dict[str, object] | None = None
            analyze_snap_start: dict[str, object] | None = None

            if SILENT_MODE and DEBUG_MODE:
                prepare_snap_start = dict(get_omdb_metrics_snapshot())
                _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: prepare:start:")

            future_to_index: dict[
                Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str]]],
                int,
            ] = {}

            # Solo para NO SILENT: orden estable dentro de biblioteca
            results: list[tuple[dict[str, object] | None, dict[str, object] | None, list[str]] | None] = []
            titles_for_index: list[str] = []
            years_for_index: list[int | None] = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for movie_index, movie in enumerate(movies_iter, start=1):
                    title = getattr(movie, "title", "") or ""
                    year_value = getattr(movie, "year", None)
                    year: int | None = year_value if isinstance(year_value, int) else None

                    guid = getattr(movie, "guid", None)
                    thumb = getattr(movie, "thumb", None)

                    file_path, file_size = get_movie_file_info(movie)

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
                    rating_key: str | None = str(rating_key_raw) if rating_key_raw is not None else None

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
                            "display_title": title,
                            "display_year": year,
                            "library_language": library_language,
                        },
                    )

                    fut = executor.submit(analyze_movie, movie_input, source_movie=movie)
                    lib_movies_enqueued += 1

                    if SILENT_MODE:
                        future_to_index[fut] = -1
                    else:
                        results.append(None)
                        titles_for_index.append(title)
                        years_for_index.append(year)
                        future_to_index[fut] = movie_index - 1

                    if SILENT_MODE and DEBUG_MODE and (movie_index % _PROGRESS_EVERY_N_MOVIES == 0):
                        _logger.progress(f"[PLEX][DEBUG] {lib_key}: encoladas {movie_index} películas...")

                if SILENT_MODE and DEBUG_MODE and prepare_snap_start is not None:
                    prepare_delta = _metrics_diff(prepare_snap_start, get_omdb_metrics_snapshot())
                    lib_omdb_delta_prepare[lib_key] = dict(prepare_delta)
                    _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: prepare:delta:", metrics=prepare_delta)

                    analyze_snap_start = dict(get_omdb_metrics_snapshot())
                    _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: analyze:start:")

                for fut in as_completed(future_to_index):
                    idx = future_to_index[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        lib_movies_errors += 1
                        total_movies_errors += 1
                        total_movies_processed += 1  # completado con error

                        if not SILENT_MODE and 0 <= idx < len(titles_for_index):
                            t = titles_for_index[idx]
                            y = years_for_index[idx]
                            _logger.error(
                                f"[PLEX] Error analizando '{t}' ({y or 'n/a'}) en '{lib_name}': {exc!r}",
                                always=True,
                            )
                        else:
                            _logger.error(f"[PLEX] Error analizando película en '{lib_name}': {exc!r}", always=True)

                        lib_movies_completed += 1
                        continue

                    if SILENT_MODE:
                        # streaming: escribir inmediatamente a CSV
                        row, meta_sugg, logs = res
                        _maybe_print_movie_logs(logs)

                        if row:
                            all_writer.write_row(row)
                            total_rows_written += 1
                            lib_rows_written += 1

                            _tally_decision(row)

                            if row.get("decision") in {"DELETE", "MAYBE"}:
                                filtered_rows.append(dict(row))

                        if meta_sugg:
                            sugg_writer.write_row(meta_sugg)
                            total_suggestions_written += 1
                            lib_suggestions_written += 1

                        total_movies_processed += 1
                    else:
                        if 0 <= idx < len(results):
                            results[idx] = res

                    lib_movies_completed += 1

                    if SILENT_MODE and DEBUG_MODE and (lib_movies_completed % _PROGRESS_EVERY_N_MOVIES == 0):
                        _logger.progress(
                            f"[PLEX][DEBUG] {lib_key}: completadas {lib_movies_completed}/{lib_movies_enqueued}..."
                        )

            # NO SILENT: volcamos en orden estable (por biblioteca) escribiendo a CSV
            if not SILENT_MODE:
                for res in results:
                    if res is None:
                        continue

                    row, meta_sugg, logs = res
                    _maybe_print_movie_logs(logs)

                    if row:
                        all_writer.write_row(row)
                        total_rows_written += 1
                        lib_rows_written += 1

                        _tally_decision(row)

                        if row.get("decision") in {"DELETE", "MAYBE"}:
                            filtered_rows.append(dict(row))

                    if meta_sugg:
                        sugg_writer.write_row(meta_sugg)
                        total_suggestions_written += 1
                        lib_suggestions_written += 1

                    # “processed” = resultado completado, haya row o no
                    total_movies_processed += 1

            if SILENT_MODE and DEBUG_MODE and analyze_snap_start is not None:
                analyze_delta = _metrics_diff(analyze_snap_start, get_omdb_metrics_snapshot())
                lib_omdb_delta_analyze[lib_key] = dict(analyze_delta)
                _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: analyze:delta:", metrics=analyze_delta)

            t_lib_elapsed = time.monotonic() - t_lib

            if SILENT_MODE:
                _logger.progress(
                    "[PLEX] Biblioteca finalizada: "
                    f"{lib_name} | enqueued={lib_movies_enqueued} | "
                    f"completed={lib_movies_completed} | "
                    f"errors={lib_movies_errors} | rows={lib_rows_written} | "
                    f"suggestions={lib_suggestions_written} | "
                    f"time={t_lib_elapsed:.1f}s"
                )
            elif DEBUG_MODE:
                _logger.info(
                    "[PLEX][DEBUG] Biblioteca finalizada: "
                    f"{lib_name} enqueued={lib_movies_enqueued} "
                    f"completed={lib_movies_completed} errors={lib_movies_errors} "
                    f"rows={lib_rows_written} suggestions={lib_suggestions_written} "
                    f"time={t_lib_elapsed:.1f}s"
                )

    # --- filtered report (solo si hay DELETE/MAYBE) ---
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

    if SILENT_MODE:
        _logger.progress(
            "[PLEX] Resumen final: "
            f"libraries={total_libs} excluded_libs={len(excluded)} "
            f"movies={total_movies_processed} errors={total_movies_errors} "
            f"workers={max_workers} time={elapsed:.1f}s | "
            f"rows={total_rows_written} (KEEP={decisions_count['KEEP']} MAYBE={decisions_count['MAYBE']} "
            f"DELETE={decisions_count['DELETE']} UNKNOWN={decisions_count['UNKNOWN']}) | "
            f"filtered_rows={filtered_len} filtered_csv={filtered_csv_status} "
            f"suggestions={total_suggestions_written}"
        )

        _logger.progress(
            "[PLEX] CSVs: "
            f"all={REPORT_ALL_PATH} | suggestions={METADATA_FIX_PATH} | filtered={REPORT_FILTERED_PATH}"
        )

        _log_omdb_metrics(prefix="[PLEX][DEBUG] Global:")

        if DEBUG_MODE:
            if lib_omdb_delta_prepare:
                _logger.progress("[PLEX][DEBUG] Rankings (prepare deltas):")
                _log_omdb_rankings(lib_omdb_delta_prepare, min_groups=2)

            if lib_omdb_delta_analyze:
                _logger.progress("[PLEX][DEBUG] Rankings (analyze deltas):")
                _log_omdb_rankings(lib_omdb_delta_analyze, min_groups=2)

    _logger.info("[PLEX] Análisis completado.", always=True)