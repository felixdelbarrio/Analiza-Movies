from __future__ import annotations

"""
backend/analiza_plex.py

Orquestador principal de análisis Plex (modo streaming + bounded inflight).

✅ Objetivo
----------
Conectar Plex -> MovieInput -> pipeline por-item (analyze_movie) y escribir CSVs:

- report_all.csv         (streaming: se escribe a medida que completan items)
- metadata_fix.csv       (streaming: sugerencias de metadata)
- report_filtered.csv    (al final; solo si hay DELETE/MAYBE; se ordena)

✅ Optimización / robustez (alineado con tus últimos cambios)
------------------------------------------------------------
1) Bounded in-flight futures (por biblioteca)
   - Cap “inflight” = workers * factor
   - Menos memoria, menos latencia, más estable en bibliotecas grandes

2) Orden estable en NO SILENT sin lista gigante
   - pending_by_index + next_to_write (memoria ~ O(inflight), no O(N))

3) Variables centralizables en config.py
   - Este módulo NO define knobs hard-coded si ya están en config.py.
   - Para no romper compatibilidad: getattr(config, ..., default).

4) Métricas con run_metrics.py (best-effort)
   - Instrumenta: timing por biblioteca, contadores de encoladas/procesadas/errores/decisiones.
   - Gated por PLEX_RUN_METRICS_ENABLED.

5) Resilience (resilience.py)
   - Envuelve el arranque “sensible” (connect_plex / get_libraries_to_analyze) con wrapper best-effort.

⚠️ Logs (mantiene la política)
------------------------------
- logger.progress(...)               -> SIEMPRE visible (heartbeat)
- logger.info/warning/error(...)     -> respetan SILENT_MODE salvo error(always=True)
- logger.debug_ctx("PLEX", "...")    -> debug contextual (DEBUG_MODE gated)

Importante:
- El orquestador llama a flush_external_caches() UNA sola vez al final del run (finally).
  No por biblioteca, no por item.

Nota sobre Pylance / sintaxis
-----------------------------
- Python NO permite `nonlocal (a, b, c)`. Debe declararse uno a uno.
  Este fichero ya está corregido para evitar esos errores.
"""

import time
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, cast

from backend import config as _cfg
from backend import logger as logger
from backend.collection_analysis import analyze_movie, flush_external_caches
from backend.decision_logic import sort_filtered_rows
from backend.movie_input import MovieInput
from backend.plex_client import (
    connect_plex,
    get_best_search_title,
    get_original_title,
    get_imdb_id_from_movie,
    get_libraries_to_analyze,
    get_movie_file_info,
)
from backend.reporting import (
    open_all_csv_writer,
    open_filtered_csv_writer_only_if_rows,
    open_suggestions_csv_writer,
)

# -----------------------------------------------------------------------------
# Best-effort: integración con resilience.py y run_metrics.py
# -----------------------------------------------------------------------------
try:
    # Se asume un wrapper del estilo: resilient_call(fn, *, label="...", **kw) -> T
    from backend.resilience import resilient_call as _resilient_call  # type: ignore
except Exception:  # pragma: no cover
    _resilient_call = None  # type: ignore


try:
    # Se asume un API flexible; adaptamos con wrappers no-op si difiere.
    import backend.run_metrics as _rm  # type: ignore
except Exception:  # pragma: no cover
    _rm = None  # type: ignore


# ============================================================================
# Knobs (preferimos config.py; fallback a defaults si no existen)
# ============================================================================

_PLEX_RUN_METRICS_ENABLED: bool = bool(getattr(_cfg, "PLEX_RUN_METRICS_ENABLED", True))

_PLEX_LIBRARY_LANGUAGE_DEFAULT: str = cast(str, getattr(_cfg, "PLEX_LIBRARY_LANGUAGE_DEFAULT", "es"))

_PLEX_LIBRARY_LANGUAGE_BY_NAME: dict[str, str] = cast(
    dict[str, str],
    getattr(_cfg, "PLEX_LIBRARY_LANGUAGE_BY_NAME", {}),
)

_PROGRESS_EVERY_N_MOVIES: int = int(getattr(_cfg, "PLEX_PROGRESS_EVERY_N_MOVIES", 100))
_MAX_WORKERS_CAP: int = int(getattr(_cfg, "PLEX_MAX_WORKERS_CAP", 64))
_DEFAULT_MAX_INFLIGHT_FACTOR: int = int(getattr(_cfg, "PLEX_MAX_INFLIGHT_FACTOR", 4))

DEBUG_MODE: bool = bool(getattr(_cfg, "DEBUG_MODE", False))
SILENT_MODE: bool = bool(getattr(_cfg, "SILENT_MODE", False))

PLEX_ANALYZE_WORKERS: int = int(getattr(_cfg, "PLEX_ANALYZE_WORKERS", 8))
EXCLUDE_PLEX_LIBRARIES: list[str] = list(getattr(_cfg, "EXCLUDE_PLEX_LIBRARIES", []))

REPORT_ALL_PATH: str = str(getattr(_cfg, "REPORT_ALL_PATH"))
REPORT_FILTERED_PATH: str = str(getattr(_cfg, "REPORT_FILTERED_PATH"))
METADATA_FIX_PATH: str = str(getattr(_cfg, "METADATA_FIX_PATH"))

# OMDb knobs (solo para capear workers; no hacemos lógica OMDb aquí)
OMDB_HTTP_MAX_CONCURRENCY: int = int(getattr(_cfg, "OMDB_HTTP_MAX_CONCURRENCY", 2))
OMDB_HTTP_MIN_INTERVAL_SECONDS: float = float(getattr(_cfg, "OMDB_HTTP_MIN_INTERVAL_SECONDS", 0.10))


# -----------------------------------------------------------------------------
# Helpers métricas (best-effort) — gated por PLEX_RUN_METRICS_ENABLED
# -----------------------------------------------------------------------------
def _rm_inc(name: str, value: int = 1, **tags: object) -> None:
    if not _PLEX_RUN_METRICS_ENABLED or _rm is None:
        return
    try:
        fn = getattr(_rm, "inc", None) or getattr(_rm, "counter_inc", None)
        if callable(fn):
            try:
                fn(name, value=value, tags=tags)  # type: ignore[misc]
            except TypeError:
                fn(name, value=value, **tags)  # type: ignore[misc]
    except Exception:
        return


def _rm_set(name: str, value: object, **tags: object) -> None:
    if not _PLEX_RUN_METRICS_ENABLED or _rm is None:
        return
    try:
        fn = getattr(_rm, "set_gauge", None) or getattr(_rm, "gauge_set", None)
        if callable(fn):
            try:
                fn(name, value=value, tags=tags)  # type: ignore[misc]
            except TypeError:
                fn(name, value=value, **tags)  # type: ignore[misc]
    except Exception:
        return


class _RM_Timer:
    def __init__(self, name: str, **tags: object) -> None:
        self._name = name
        self._tags = tags
        self._t0 = 0.0

    def __enter__(self) -> "_RM_Timer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        elapsed = time.monotonic() - self._t0
        if not _PLEX_RUN_METRICS_ENABLED or _rm is None:
            return
        try:
            fn = getattr(_rm, "observe_seconds", None) or getattr(_rm, "timing", None)
            if callable(fn):
                try:
                    fn(self._name, seconds=elapsed, tags=self._tags)  # type: ignore[misc]
                except TypeError:
                    fn(self._name, seconds=elapsed, **self._tags)  # type: ignore[misc]
        except Exception:
            return


def _with_resilience(fn: Callable[[], Any], *, label: str) -> Any:
    if _resilient_call is None:
        return fn()
    return _resilient_call(fn, label=label)  # type: ignore[misc]


# ============================================================================
# WORKERS / inflight caps
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

    return max(1, max_workers)


def _compute_max_inflight(max_workers: int) -> int:
    inflight = int(max_workers) * int(_DEFAULT_MAX_INFLIGHT_FACTOR)
    return max(int(max_workers), int(inflight))


# ============================================================================
# UTILIDADES PLEX
# ============================================================================
def _get_plex_library_language(lib_name: str) -> str:
    if lib_name and lib_name in _PLEX_LIBRARY_LANGUAGE_BY_NAME:
        return str(_PLEX_LIBRARY_LANGUAGE_BY_NAME[lib_name])
    return str(_PLEX_LIBRARY_LANGUAGE_DEFAULT)


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
    """
    Punto de entrada principal para analizar Plex.

    Garantía:
    - flush_external_caches() se llama UNA vez al final del run (finally),
      incluso si retornamos temprano por errores/0 bibliotecas.
    """
    t0 = time.monotonic()
    _rm_inc("plex.run.start", 1)

    # IMPORTANT: todo el flujo dentro de try/finally para garantizar flush.
    try:
        # “Conectores” sensibles: se benefician de resilience si está disponible.
        try:
            plex = _with_resilience(connect_plex, label="plex.connect")
            raw_libraries = _with_resilience(lambda: get_libraries_to_analyze(plex), label="plex.list_libraries")
        except Exception as exc:
            logger.error(f"[PLEX] Error conectando/listando bibliotecas: {exc!r}", always=True)
            _rm_inc("plex.run.fatal_error", 1)
            return

        # Filtrado por EXCLUDE_PLEX_LIBRARIES
        libraries: list[Any] = []
        excluded: list[str] = []
        for lib in raw_libraries:
            name = _library_title(lib)
            if name and name in EXCLUDE_PLEX_LIBRARIES:
                excluded.append(name)
                continue
            libraries.append(lib)

        total_libs = len(libraries)
        _rm_set("plex.libraries.total", total_libs)
        _rm_set("plex.libraries.excluded", len(excluded))

        if SILENT_MODE and excluded:
            logger.progress("[PLEX] Bibliotecas excluidas por configuración: " + ", ".join(sorted(excluded)))

        if total_libs == 0:
            logger.progress("[PLEX] No hay bibliotecas para analizar (0).")
            _rm_inc("plex.run.no_libraries", 1)
            return

        filtered_rows: list[dict[str, object]] = []
        decisions_count: dict[str, int] = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}

        total_movies_processed = 0
        total_movies_errors = 0
        total_rows_written = 0
        total_suggestions_written = 0

        max_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS, total_work_items=None)
        max_inflight = _compute_max_inflight(max_workers)

        _rm_set("plex.pool.workers", max_workers)
        _rm_set("plex.pool.inflight_cap", max_inflight)

        if SILENT_MODE:
            logger.progress(
                f"[PLEX] ThreadPool workers={max_workers} inflight_cap={max_inflight} (por biblioteca) "
                f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, "
                f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
                f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
            )
        else:
            logger.debug_ctx(
                "PLEX",
                f"ThreadPool workers={max_workers} inflight_cap={max_inflight} (por biblioteca) "
                f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, cap por OMDb limiter)",
            )

        def _maybe_print_movie_logs(logs: list[str]) -> None:
            if not logs:
                return

            silent = bool(SILENT_MODE)
            debug = bool(DEBUG_MODE)

            if not silent:
                for line in logs:
                    logger.info(line)
                return

            if debug:
                for line in logs:
                    logger.info(line, always=True)

        def _tally_decision(row: Mapping[str, object]) -> None:
            d = row.get("decision")
            if d in ("KEEP", "MAYBE", "DELETE"):
                decisions_count[str(d)] += 1
                _rm_inc(f"plex.decision.{str(d).lower()}", 1)
            else:
                decisions_count["UNKNOWN"] += 1
                _rm_inc("plex.decision.unknown", 1)

        def _handle_result(
            res: tuple[dict[str, object] | None, dict[str, object] | None, list[str]],
            *,
            all_writer: Any,
            sugg_writer: Any,
            lib_rows_written_ref: dict[str, int],
            lib_suggestions_written_ref: dict[str, int],
        ) -> None:
            nonlocal total_rows_written
            nonlocal total_suggestions_written

            row, meta_sugg, logs = res
            _maybe_print_movie_logs(logs)

            if row:
                all_writer.write_row(row)
                total_rows_written += 1
                lib_rows_written_ref["v"] += 1

                _tally_decision(row)

                if row.get("decision") in {"DELETE", "MAYBE"}:
                    filtered_rows.append(dict(row))

            if meta_sugg:
                sugg_writer.write_row(meta_sugg)
                total_suggestions_written += 1
                lib_suggestions_written_ref["v"] += 1
                _rm_inc("plex.metadata_suggestion.written", 1)

        # =========================================================================
        # Writers (streaming global)
        # =========================================================================
        with open_all_csv_writer(REPORT_ALL_PATH) as all_writer, open_suggestions_csv_writer(METADATA_FIX_PATH) as sugg_writer:
            for lib_index, library in enumerate(libraries, start=1):
                lib_name = _library_title(library)
                lib_key = lib_name or f"<lib_{lib_index}>"

                logger.progress(f"[PLEX] ({lib_index}/{total_libs}) {lib_name or '<sin nombre>'}")
                if not SILENT_MODE:
                    logger.info(f"Analizando biblioteca Plex: {lib_name}")

                library_language = _get_plex_library_language(lib_name)
                logger.debug_ctx(
                    "PLEX",
                    f"library_language={library_language!r} excluded={len(excluded)} "
                    f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY} "
                    f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS}",
                )

                lib_movies_enqueued = 0
                lib_movies_completed = 0
                lib_movies_errors = 0
                lib_rows_written = {"v": 0}
                lib_suggestions_written = {"v": 0}

                movies_iter, total_movies_in_library = _iter_movies_with_total(library)

                future_to_index: dict[
                    Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str]]],
                    int,
                ] = {}

                next_to_write = 1
                pending_by_index: dict[
                    int,
                    tuple[dict[str, object] | None, dict[str, object] | None, list[str]],
                ] = {}

                index_to_title_year: dict[int, tuple[str, int | None]] = {}

                inflight: set[
                    Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str]]]
                ] = set()

                def _drain_completed(*, drain_all: bool) -> None:
                    nonlocal lib_movies_completed
                    nonlocal lib_movies_errors
                    nonlocal total_movies_errors
                    nonlocal total_movies_processed
                    nonlocal next_to_write

                    while inflight:
                        done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
                        if not done:
                            return

                        for fut in done:
                            inflight.discard(fut)
                            idx_local = future_to_index.get(fut, -1)

                            try:
                                res = fut.result()
                            except Exception as exc:
                                lib_movies_errors += 1
                                total_movies_errors += 1
                                total_movies_processed += 1
                                _rm_inc("plex.movie.error", 1, library=lib_key)

                                if (not SILENT_MODE) and (idx_local in index_to_title_year):
                                    t, y = index_to_title_year[idx_local]
                                    logger.error(
                                        f"[PLEX] Error analizando '{t}' ({y or 'n/a'}) en '{lib_name}': {exc!r}",
                                        always=True,
                                    )
                                else:
                                    logger.error(
                                        f"[PLEX] Error analizando película en '{lib_name}': {exc!r}",
                                        always=True,
                                    )

                                lib_movies_completed += 1
                                continue

                            if SILENT_MODE:
                                _handle_result(
                                    res,
                                    all_writer=all_writer,
                                    sugg_writer=sugg_writer,
                                    lib_rows_written_ref=lib_rows_written,
                                    lib_suggestions_written_ref=lib_suggestions_written,
                                )
                                total_movies_processed += 1
                                _rm_inc("plex.movie.processed", 1, library=lib_key)
                            else:
                                if idx_local >= 1:
                                    pending_by_index[idx_local] = res

                                while next_to_write in pending_by_index:
                                    ready = pending_by_index.pop(next_to_write)
                                    _handle_result(
                                        ready,
                                        all_writer=all_writer,
                                        sugg_writer=sugg_writer,
                                        lib_rows_written_ref=lib_rows_written,
                                        lib_suggestions_written_ref=lib_suggestions_written,
                                    )
                                    total_movies_processed += 1
                                    _rm_inc("plex.movie.processed", 1, library=lib_key)
                                    next_to_write += 1

                            lib_movies_completed += 1

                            if SILENT_MODE and DEBUG_MODE and (lib_movies_completed % _PROGRESS_EVERY_N_MOVIES == 0):
                                logger.progress(
                                    f"[PLEX][DEBUG] {lib_key}: completadas {lib_movies_completed}/{lib_movies_enqueued}..."
                                )

                        if not drain_all:
                            return

                with _RM_Timer("plex.library.seconds", library=lib_key), ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for movie_index, movie in enumerate(movies_iter, start=1):
                        title = getattr(movie, "title", "") or ""

                        year_value = getattr(movie, "year", None)
                        year: int | None = year_value if isinstance(year_value, int) else None

                        guid = getattr(movie, "guid", None)
                        thumb = getattr(movie, "thumb", None)

                        file_path, file_size = get_movie_file_info(movie)

                        if not SILENT_MODE:
                            logger.info(
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

                        plex_original_title = get_original_title(movie)

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
                                "plex_original_title": plex_original_title,
                                "library_language": library_language,
                            },
                        )

                        fut = executor.submit(analyze_movie, movie_input, source_movie=movie)

                        future_to_index[fut] = movie_index
                        inflight.add(fut)
                        lib_movies_enqueued += 1
                        _rm_inc("plex.movie.enqueued", 1, library=lib_key)

                        if not SILENT_MODE:
                            index_to_title_year[movie_index] = (title, year)

                        if SILENT_MODE and DEBUG_MODE and (movie_index % _PROGRESS_EVERY_N_MOVIES == 0):
                            logger.progress(f"[PLEX][DEBUG] {lib_key}: encoladas {movie_index} películas...")

                        if len(inflight) >= max_inflight:
                            _drain_completed(drain_all=False)

                    _drain_completed(drain_all=True)

                if not SILENT_MODE:
                    while next_to_write in pending_by_index:
                        ready = pending_by_index.pop(next_to_write)
                        _handle_result(
                            ready,
                            all_writer=all_writer,
                            sugg_writer=sugg_writer,
                            lib_rows_written_ref=lib_rows_written,
                            lib_suggestions_written_ref=lib_suggestions_written,
                        )
                        total_movies_processed += 1
                        _rm_inc("plex.movie.processed", 1, library=lib_key)
                        next_to_write += 1

                if SILENT_MODE:
                    logger.progress(
                        "[PLEX] Biblioteca finalizada: "
                        f"{lib_name} | enqueued={lib_movies_enqueued} | "
                        f"completed={lib_movies_completed} | "
                        f"errors={lib_movies_errors} | rows={lib_rows_written['v']} | "
                        f"suggestions={lib_suggestions_written['v']}"
                    )
                elif DEBUG_MODE:
                    logger.info(
                        "[PLEX][DEBUG] Biblioteca finalizada: "
                        f"{lib_name} enqueued={lib_movies_enqueued} "
                        f"completed={lib_movies_completed} errors={lib_movies_errors} "
                        f"rows={lib_rows_written['v']} suggestions={lib_suggestions_written['v']}"
                    )

                _rm_set("plex.library.movies.enqueued", lib_movies_enqueued, library=lib_key)
                _rm_set("plex.library.movies.errors", lib_movies_errors, library=lib_key)

        # =========================================================================
        # filtered report (solo si hay DELETE/MAYBE)
        # =========================================================================
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

        _rm_set("plex.run.seconds", elapsed)
        _rm_set("plex.movies.processed", total_movies_processed)
        _rm_set("plex.movies.errors", total_movies_errors)
        _rm_set("plex.rows.written", total_rows_written)
        _rm_set("plex.suggestions.written", total_suggestions_written)
        _rm_set("plex.filtered.rows", filtered_len)

        if SILENT_MODE:
            logger.progress(
                "[PLEX] Resumen final: "
                f"libraries={total_libs} excluded_libs={len(excluded)} "
                f"movies={total_movies_processed} errors={total_movies_errors} "
                f"workers={max_workers} inflight_cap={max_inflight} time={elapsed:.1f}s | "
                f"rows={total_rows_written} (KEEP={decisions_count['KEEP']} MAYBE={decisions_count['MAYBE']} "
                f"DELETE={decisions_count['DELETE']} UNKNOWN={decisions_count['UNKNOWN']}) | "
                f"filtered_rows={filtered_len} filtered_csv={filtered_csv_status} "
                f"suggestions={total_suggestions_written}"
            )
            logger.progress(
                "[PLEX] CSVs: "
                f"all={REPORT_ALL_PATH} | suggestions={METADATA_FIX_PATH} | filtered={REPORT_FILTERED_PATH}"
            )

        logger.info("[PLEX] Análisis completado.", always=True)
        _rm_inc("plex.run.ok", 1)

    except Exception as exc:
        _rm_inc("plex.run.exception", 1)
        logger.error(f"[PLEX] Error inesperado en el run: {exc!r}", always=True)
        raise

    finally:
        # ✅ Flush una vez al final del run, siempre (aunque retornemos antes).
        try:
            flush_external_caches()
        except Exception as exc:  # pragma: no cover
            logger.debug_ctx("PLEX", f"flush_external_caches failed: {exc!r}")

        if _PLEX_RUN_METRICS_ENABLED and _rm is not None:
            try:
                fn = getattr(_rm, "flush", None) or getattr(_rm, "log_summary", None)
                if callable(fn):
                    fn()  # type: ignore[misc]
            except Exception:
                pass