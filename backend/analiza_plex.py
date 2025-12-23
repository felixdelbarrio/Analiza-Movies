from __future__ import annotations

"""
backend/collection_analiza_plex.py  (antes: analiza_plex.py)

Orquestador principal de análisis Plex (modo streaming + bounded inflight).

✅ Cambio de esta iteración (patch mínimo)
-----------------------------------------
- Este orquestador sigue llamando a `flush_external_caches()` EXACTAMENTE igual que antes
  (una vez al final del run), pero ahora ese flush ya NO depende de ningún wrapper de Wiki
  (p.ej. `get_wiki_client()`), porque `collection_analysis.py` ya hace llamadas directas
  (get_wiki_for_input / get_wiki) y expone un flush agregado.

Objetivo del módulo
-------------------
Conecta Plex -> MovieInput -> pipeline por-item (analyze_movie) y escribe CSVs:

- report_all.csv         (streaming: se escribe a medida que completan items)
- metadata_fix.csv       (streaming: sugerencias de metadata; en Plex suele tener contenido)
- report_filtered.csv    (al final y solo si hay filas DELETE/MAYBE; se ordena)

Mejoras clave (performance: IO + APIs externas)
-----------------------------------------------
1) Bounded in-flight futures (por biblioteca)
   - Mantiene un máximo de futures “en vuelo” (inflight_cap).
   - Beneficios:
     * menos memoria (closures + buffers + resultados)
     * menor latencia: se empiezan a escribir filas antes
     * mejor comportamiento en bibliotecas grandes

2) Orden estable en NO SILENT sin lista gigante
   - pending_by_index + next_to_write (memoria ~ O(inflight) en vez de O(N)).

3) Compatible con “Lazy OMDb/Wiki”
   - analyze_movie / core pueden no llamar OMDb.
   - Aun así, limitamos concurrencia con un cap sensato para evitar “storm” de IO.

Filosofía de logging (alineado con backend/logger.py)
-----------------------------------------------------
- logger.progress(...)               -> SIEMPRE visible (heartbeat)
- logger.info/warning/error(...)     -> logs por nivel (respetan SILENT_MODE salvo error())
- logger.debug_ctx("PLEX", "...")    -> debug contextual (DEBUG_MODE gated)

En SILENT_MODE:
- progresos compactos con progress()
- resumen final explícito + estado de filtered.csv
En DEBUG_MODE:
- métricas OMDb y rankings sin spamear en modo normal

Notas sobre métricas “prepare/analyze”
--------------------------------------
Se mantienen por continuidad histórica. Con drenaje durante encolado:
- “prepare” puede incluir algo de ejecución real (porque drenamos para acotar inflight),
  pero sigue siendo útil como aproximación para comparar coste entre bibliotecas.
"""

import time
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from backend import logger as logger
from backend.collection_analysis import analyze_movie, flush_external_caches
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
    # "Movies": "es",
}

_PROGRESS_EVERY_N_MOVIES: int = 100
_MAX_WORKERS_CAP: int = 64

# Máximo de futures simultáneos por biblioteca = workers * factor
_DEFAULT_MAX_INFLIGHT_FACTOR: int = 4


# ============================================================================
# OMDb metrics helpers (solo SILENT+DEBUG)
# ============================================================================


def _metrics_get_int(m: Mapping[str, object], key: str) -> int:
    """Parse defensivo de métricas para evitar fallos por tipos inesperados."""
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


def _metrics_diff(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, int]:
    """Diff parcial de métricas “interesantes” (evita ruido)."""
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


def _log_omdb_metrics(prefix: str, metrics: Mapping[str, object] | None = None) -> None:
    """
    Log ultra-compacto de métricas OMDb.

    Solo se emite en SILENT+DEBUG:
    - SILENT para no contaminar UI normal
    - DEBUG para no pagar coste/ruido si no se está diagnosticando
    """
    if not (SILENT_MODE and DEBUG_MODE):
        return

    m = metrics or get_omdb_metrics_snapshot()
    logger.progress(
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


def _compute_cost_score(delta: Mapping[str, int]) -> int:
    """
    Heurística de “coste”:
    - requests cuestan poco
    - throttle medio
    - rate-limit y fallos mucho
    - sleeps por rate-limit muchísimo
    """
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
    """Rankings compactos para diagnosticar bibliotecas “caras”."""
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
        logger.progress(f"[PLEX][DEBUG] Top libraries by TOTAL_COST: {line}")
    if (line := _fmt(top_http)):
        logger.progress(f"[PLEX][DEBUG] Top libraries by http_requests: {line}")
    if (line := _fmt(top_rls)):
        logger.progress(f"[PLEX][DEBUG] Top libraries by rate_limit_sleeps: {line}")
    if (line := _fmt(top_fail)):
        logger.progress(f"[PLEX][DEBUG] Top libraries by http_failures: {line}")
    if (line := _fmt(top_rlh)):
        logger.progress(f"[PLEX][DEBUG] Top libraries by rate_limit_hits: {line}")


# ============================================================================
# WORKERS (cap por OMDb limiter)
# ============================================================================


def _compute_max_workers(requested: int, total_work_items: int | None) -> int:
    """
    Decide el número de workers reales del ThreadPool por biblioteca.

    Capas:
    1) requested (PLEX_ANALYZE_WORKERS) con clamp [1.._MAX_WORKERS_CAP]
    2) cap relativo a OMDb (evita saturar la capa HTTP si hay storm):
       OMDB_HTTP_MAX_CONCURRENCY * 8 (mínimo 4)
    3) cap por total_work_items si se conoce (no spawn de más hilos que items)
    """
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
    """
    Máximo de futures simultáneos “en vuelo” por biblioteca.

    Regla: inflight >= workers para no starve.
    """
    inflight = max_workers * _DEFAULT_MAX_INFLIGHT_FACTOR
    return max(max_workers, inflight)


# ============================================================================
# UTILIDADES PLEX (idioma / safe access)
# ============================================================================


def _get_plex_library_language(lib_name: str) -> str:
    """Idioma por librería, usado por el scoring/core via MovieInput.extra."""
    return _PLEX_LIBRARY_LANGUAGE_BY_NAME.get(lib_name) or _PLEX_LIBRARY_LANGUAGE_DEFAULT


def _library_title(library: Any) -> str:
    """Título de biblioteca Plex (seguro)."""
    return (getattr(library, "title", "") or "").strip()


def _library_total_items(library: Any) -> int | None:
    """totalSize si Plex lo aporta."""
    raw = getattr(library, "totalSize", None)
    return raw if isinstance(raw, int) and raw >= 0 else None


def _iter_movies_with_total(library: Any) -> tuple[Iterable[Any], int | None]:
    """
    Devuelve (iterable_de_movies, total_en_biblioteca | None).

    - SILENT_MODE: no materializamos para contar (evitamos RAM).
    - NO SILENT:
        * si Plex da totalSize, lo usamos para (i/total)
        * si no y DEBUG_MODE: materializamos para poder mostrar total real
        * si no: devolvemos total None
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
    """Línea humana (NO SILENT) por item."""
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

    Política de memoria:
    - Siempre: filtered_rows (DELETE/MAYBE) para ordenación final.
    - SILENT: filas se escriben al vuelo (no se guarda la biblioteca completa).
    - NO SILENT: orden estable por biblioteca con buffer acotado (pending_by_index).

    Robustez:
    - Errores por-item no paran el run; se contabilizan y se reportan.
    - Bibliotecas excluidas por config se saltan de forma explícita.

    Importante:
    - Este orquestador llama a flush_external_caches() UNA sola vez al final del run.
      (No por biblioteca, no por item.)
    """
    t0 = time.monotonic()
    reset_omdb_metrics()

    # ✅ Patch mínimo solicitado:
    # Aseguramos flush de caches aunque haya early-return o excepción inesperada.
    try:
        plex = connect_plex()
        raw_libraries = get_libraries_to_analyze(plex)

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

        if SILENT_MODE and excluded:
            logger.progress("[PLEX] Bibliotecas excluidas por configuración: " + ", ".join(sorted(excluded)))

        if total_libs == 0:
            logger.progress("[PLEX] No hay bibliotecas para analizar (0).")
            return

        filtered_rows: list[dict[str, object]] = []
        decisions_count: dict[str, int] = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}

        total_movies_processed = 0
        total_movies_errors = 0
        total_rows_written = 0
        total_suggestions_written = 0

        # Workers globales (cap adicional por OMDb)
        max_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS, total_work_items=None)
        max_inflight = _compute_max_inflight(max_workers)

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

        # Rankings de métricas (solo SILENT+DEBUG)
        lib_omdb_delta_prepare: dict[str, dict[str, int]] = {}
        lib_omdb_delta_analyze: dict[str, dict[str, int]] = {}

        def _maybe_print_movie_logs(logs: list[str]) -> None:
            """
            logs vienen acotados desde collection_analysis.

            - NO SILENT: se imprimen como info normal.
            - SILENT+DEBUG: se imprimen always=True (útil para inspección sin romper silent).
            - SILENT sin debug: no se imprime nada.
            """
            if not logs:
                return
            if not SILENT_MODE:
                for line in logs:
                    logger.info(line)
                return
            if DEBUG_MODE:
                for line in logs:
                    logger.info(line, always=True)

        def _tally_decision(row: Mapping[str, object]) -> None:
            """Acumula decisiones para resumen final."""
            d = row.get("decision")
            if d in ("KEEP", "MAYBE", "DELETE"):
                decisions_count[str(d)] += 1
            else:
                decisions_count["UNKNOWN"] += 1

        def _handle_result(
            res: tuple[dict[str, object] | None, dict[str, object] | None, list[str]],
            *,
            all_writer: Any,
            sugg_writer: Any,
            lib_rows_written_ref: dict[str, int],
            lib_suggestions_written_ref: dict[str, int],
        ) -> None:
            """
            Aplica un resultado de analyze_movie:
            - imprime logs (según modo)
            - escribe row a report_all.csv
            - acumula filtered_rows si DELETE/MAYBE
            - escribe meta_sugg a metadata_fix.csv
            - actualiza contadores globales y por biblioteca
            """
            nonlocal total_rows_written, total_suggestions_written

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

                t_lib = time.monotonic()
                movies_iter, total_movies_in_library = _iter_movies_with_total(library)

                # Métricas (SILENT+DEBUG)
                prepare_snap_start: dict[str, object] | None = None
                analyze_snap_start: dict[str, object] | None = None

                if SILENT_MODE and DEBUG_MODE:
                    prepare_snap_start = dict(get_omdb_metrics_snapshot())
                    _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: prepare:start:")

                # future -> índice (1..N)
                future_to_index: dict[
                    Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str]]],
                    int,
                ] = {}

                # NO SILENT: orden estable sin lista gigante
                next_to_write = 1
                pending_by_index: dict[
                    int,
                    tuple[dict[str, object] | None, dict[str, object] | None, list[str]],
                ] = {}

                # Para enriquecer errores (NO SILENT)
                index_to_title_year: dict[int, tuple[str, int | None]] = {}

                inflight: set[
                    Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str]]]
                ] = set()

                def _drain_completed(*, drain_all: bool) -> None:
                    """
                    Drena futures completados.

                    - drain_all=False:
                        procesa un batch completado para liberar sitio en inflight.
                    - drain_all=True:
                        drena hasta completar todo lo pendiente.

                    Importante:
                    - en SILENT se procesa/escribe inmediatamente.
                    - en NO SILENT se guarda por índice y se vuelca en orden.
                    """
                    nonlocal lib_movies_completed, lib_movies_errors, total_movies_errors, total_movies_processed, next_to_write

                    while inflight:
                        done, _ = wait(inflight, return_when=FIRST_COMPLETED)
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

                                if not SILENT_MODE and idx_local in index_to_title_year:
                                    t, y = index_to_title_year[idx_local]
                                    logger.error(
                                        f"[PLEX] Error analizando '{t}' ({y or 'n/a'}) en '{lib_name}': {exc!r}",
                                        always=True,
                                    )
                                else:
                                    logger.error(f"[PLEX] Error analizando película en '{lib_name}': {exc!r}", always=True)

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
                                    next_to_write += 1

                            lib_movies_completed += 1

                            if SILENT_MODE and DEBUG_MODE and (lib_movies_completed % _PROGRESS_EVERY_N_MOVIES == 0):
                                logger.progress(
                                    f"[PLEX][DEBUG] {lib_key}: completadas {lib_movies_completed}/{lib_movies_enqueued}..."
                                )

                        if not drain_all:
                            return

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

                        future_to_index[fut] = movie_index
                        inflight.add(fut)
                        lib_movies_enqueued += 1

                        if not SILENT_MODE:
                            index_to_title_year[movie_index] = (title, year)

                        if SILENT_MODE and DEBUG_MODE and (movie_index % _PROGRESS_EVERY_N_MOVIES == 0):
                            logger.progress(f"[PLEX][DEBUG] {lib_key}: encoladas {movie_index} películas...")

                        if len(inflight) >= max_inflight:
                            _drain_completed(drain_all=False)

                    # “prepare delta” al terminar encolado
                    if SILENT_MODE and DEBUG_MODE and prepare_snap_start is not None:
                        prepare_delta = _metrics_diff(prepare_snap_start, get_omdb_metrics_snapshot())
                        lib_omdb_delta_prepare[lib_key] = dict(prepare_delta)
                        _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: prepare:delta:", metrics=prepare_delta)

                        analyze_snap_start = dict(get_omdb_metrics_snapshot())
                        _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: analyze:start:")

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
                        next_to_write += 1

                if SILENT_MODE and DEBUG_MODE and analyze_snap_start is not None:
                    analyze_delta = _metrics_diff(analyze_snap_start, get_omdb_metrics_snapshot())
                    lib_omdb_delta_analyze[lib_key] = dict(analyze_delta)
                    _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: analyze:delta:", metrics=analyze_delta)

                t_lib_elapsed = time.monotonic() - t_lib

                if SILENT_MODE:
                    logger.progress(
                        "[PLEX] Biblioteca finalizada: "
                        f"{lib_name} | enqueued={lib_movies_enqueued} | "
                        f"completed={lib_movies_completed} | "
                        f"errors={lib_movies_errors} | rows={lib_rows_written['v']} | "
                        f"suggestions={lib_suggestions_written['v']} | "
                        f"time={t_lib_elapsed:.1f}s"
                    )
                elif DEBUG_MODE:
                    logger.info(
                        "[PLEX][DEBUG] Biblioteca finalizada: "
                        f"{lib_name} enqueued={lib_movies_enqueued} "
                        f"completed={lib_movies_completed} errors={lib_movies_errors} "
                        f"rows={lib_rows_written['v']} suggestions={lib_suggestions_written['v']} "
                        f"time={t_lib_elapsed:.1f}s"
                    )

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

            _log_omdb_metrics(prefix="[PLEX][DEBUG] Global:")

            if DEBUG_MODE:
                if lib_omdb_delta_prepare:
                    logger.progress("[PLEX][DEBUG] Rankings (prepare deltas):")
                    _log_omdb_rankings(lib_omdb_delta_prepare, min_groups=2)

                if lib_omdb_delta_analyze:
                    logger.progress("[PLEX][DEBUG] Rankings (analyze deltas):")
                    _log_omdb_rankings(lib_omdb_delta_analyze, min_groups=2)

        logger.info("[PLEX] Análisis completado.", always=True)

    finally:
        # ✅ Patch mínimo: flush una vez al final del run, siempre.
        # (No por biblioteca, no por item.)
        try:
            flush_external_caches()
        except Exception as exc:  # pragma: no cover
            logger.debug_ctx("PLEX", f"flush_external_caches failed: {exc!r}")