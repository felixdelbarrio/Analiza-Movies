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

Performance:
- Paralelismo controlado con ThreadPool (I/O bound: Plex/OMDb/Wiki).
- Se mantiene la salida y el orden estable del pipeline:
    * Progreso por película se imprime al encolar (modo no-silent).
    * Resultados se agregan en el orden original de la biblioteca.

Métricas OMDb por biblioteca:
- Para que los deltas por biblioteca sean REALMENTE atribuibles, se crea un ThreadPool
  por biblioteca (barrera de concurrencia por grupo).
- Así evitamos que el trabajo de la librería N se solape con encolado/ejecución de la N+1.

Alineación con OMDb limiter (config.py):
- OMDB_HTTP_MAX_CONCURRENCY: semáforo global para llamadas HTTP OMDb (en omdb_client)
- OMDB_HTTP_MIN_INTERVAL_SECONDS: throttle global
Aunque haya muchos workers, OMDb se “suaviza”. Aun así, capamos workers a un valor
razonable para evitar hilos ociosos y reducir presión global del sistema.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Iterable

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

_MAX_WORKERS_CAP: int = 64


# ============================================================================
# LOGGING CONTROLADO POR MODOS (en línea con el resto del proyecto)
# ============================================================================


def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False → no hace nada.
    - DEBUG_MODE=True:
        * SILENT_MODE=True: progress para señales de vida sin ruido excesivo.
        * SILENT_MODE=False: info normal.
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


# --------------------------
# OMDb metrics helpers (igual que DLNA)
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
    """
    Imprime (solo en silent+debug) un resumen rápido del comportamiento del cliente OMDb.
    Si `metrics` es None, usa snapshot global actual.
    """
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


def _rank_top(
    deltas_by_lib: dict[str, dict[str, int]],
    key: str,
    top_n: int = 5,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for lib, d in deltas_by_lib.items():
        rows.append((lib, int(d.get(key, 0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _compute_cost_score(delta: dict[str, int]) -> int:
    """
    Score “coste total” para detectar bibliotecas “caras”.

    Pesos (idénticos a DLNA para comparabilidad):
      - rate_limit_sleeps: muy caro (bloquea y alarga ejecución) -> *50
      - http_failures: caro (reintentos/ruido/decisiones peor)  -> *20
      - rate_limit_hits: señal de presión                             -> *10
      - throttle_sleeps: coste moderado                                -> *3
      - http_requests: coste base                                       -> *1
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


def _rank_top_by_total_cost(
    deltas_by_lib: dict[str, dict[str, int]],
    top_n: int = 5,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for lib, d in deltas_by_lib.items():
        rows.append((lib, _compute_cost_score(d)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _log_omdb_rankings(deltas_by_lib: dict[str, dict[str, int]], *, min_groups: int = 2) -> None:
    """
    Ranking automático (solo silent+debug) por deltas reales.
    Incluye “Top 5 por coste total”.
    """
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

    cost_line = _fmt(top_cost)
    http_line = _fmt(top_http)
    rls_line = _fmt(top_rls)
    fail_line = _fmt(top_fail)
    rlh_line = _fmt(top_rlh)

    if cost_line:
        _logger.progress(f"[PLEX][DEBUG] Top libraries by TOTAL_COST: {cost_line}")
    if http_line:
        _logger.progress(f"[PLEX][DEBUG] Top libraries by http_requests: {http_line}")
    if rls_line:
        _logger.progress(f"[PLEX][DEBUG] Top libraries by rate_limit_sleeps: {rls_line}")
    if fail_line:
        _logger.progress(f"[PLEX][DEBUG] Top libraries by http_failures: {fail_line}")
    if rlh_line:
        _logger.progress(f"[PLEX][DEBUG] Top libraries by rate_limit_hits: {rlh_line}")


def _count_decisions(rows: list[dict[str, object]]) -> dict[str, int]:
    """
    Cuenta decisiones en all_rows para el resumen final.
    """
    out = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}
    for r in rows:
        d = r.get("decision")
        if d in ("KEEP", "MAYBE", "DELETE"):
            out[str(d)] += 1
        else:
            out["UNKNOWN"] += 1
    return out


# ============================================================================
# WORKERS (conectado a OMDb limiter)
# ============================================================================


def _compute_max_workers(requested: int, total_work_items: int | None) -> int:
    """
    Resuelve max_workers final.

    - requested: viene de PLEX_ANALYZE_WORKERS (config.py)
    - total_work_items: si se conoce, evitamos crear más workers que trabajo

    Cap inteligente por OMDb limiter:
      workers <= max(4, OMDB_HTTP_MAX_CONCURRENCY * 8)
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

    if max_workers < 1:
        max_workers = 1

    return max_workers


# ============================================================================
# UTILIDADES PLEX (idioma / safe access)
# ============================================================================


def _get_plex_library_language(lib_name: str) -> str:
    lang = _PLEX_LIBRARY_LANGUAGE_BY_NAME.get(lib_name)
    return lang or _PLEX_LIBRARY_LANGUAGE_DEFAULT


def _library_title(library: Any) -> str:
    return (getattr(library, "title", "") or "").strip()


def _library_total_items(library: Any) -> int | None:
    raw = getattr(library, "totalSize", None)
    return raw if isinstance(raw, int) and raw >= 0 else None


def _iter_movies_with_total(
    library: Any,
) -> tuple[Iterable[Any], int | None]:
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
    if total is None:
        return f"({index}/?)"
    return f"({index}/{total})"


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
    Analiza todas las bibliotecas Plex aplicando EXCLUDE_PLEX_LIBRARIES.

    Salidas:
    - report_all.csv
    - report_filtered.csv
    - metadata_fix.csv
    """
    t0 = time.monotonic()

    # Métricas OMDb: resumen “limpio” por ejecución.
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
        _logger.progress(
            "[PLEX] Bibliotecas excluidas por configuración: " + ", ".join(sorted(excluded))
        )

    if total_libs == 0:
        _logger.progress("[PLEX] No hay bibliotecas para analizar (0).")
        return

    all_rows: list[dict[str, object]] = []
    suggestion_rows: list[dict[str, object]] = []

    total_movies_processed = 0
    total_movies_errors = 0

    max_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS, total_work_items=None)

    # Importante: ahora el pool es por biblioteca (métricas por biblioteca fiables).
    if SILENT_MODE:
        _logger.progress(
            f"[PLEX] ThreadPool workers={max_workers} (por biblioteca) "
            f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, "
            f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
            f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
        )
    else:
        _log_debug(
            f"ThreadPool workers={max_workers} (por biblioteca) "
            f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, cap por OMDb limiter)"
        )

    # Snapshots/deltas OMDb por biblioteca (solo se usan en silent+debug)
    lib_omdb_snapshot_start: dict[str, dict[str, object]] = {}
    lib_omdb_delta: dict[str, dict[str, int]] = {}

    for lib_index, library in enumerate(libraries, start=1):
        lib_name = _library_title(library)
        lib_key = lib_name or f"<lib_{lib_index}>"

        if lib_name:
            _logger.progress(f"[PLEX] ({lib_index}/{total_libs}) {lib_name}")
        else:
            _logger.progress(f"[PLEX] ({lib_index}/{total_libs}) <sin nombre>")

        # --- OMDb metrics al inicio de cada biblioteca (justo tras progress) ---
        if SILENT_MODE and DEBUG_MODE:
            snap = get_omdb_metrics_snapshot()
            lib_omdb_snapshot_start[lib_key] = dict(snap)
            _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: start:")

        _logger.info(f"Analizando biblioteca Plex: {lib_name}")

        library_language = _get_plex_library_language(lib_name)

        _log_debug(
            f"library_language={library_language!r} excluded={len(excluded)} "
            f"OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY} "
            f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS}"
        )

        lib_movies_enqueued = 0
        lib_movies_completed = 0
        lib_movies_errors = 0
        lib_rows_added = 0
        lib_suggestions_added = 0

        t_lib = time.monotonic()

        movies_iter, total_movies_in_library = _iter_movies_with_total(library)

        results: list[
            tuple[dict[str, object] | None, dict[str, object] | None, list[str]] | None
        ] = []
        titles_for_index: list[str] = []
        years_for_index: list[int | None] = []

        future_to_index: dict[
            Future[tuple[dict[str, object] | None, dict[str, object] | None, list[str]]], int
        ] = {}

        # ------------------------------------------------------------------
        # BARRERA POR BIBLIOTECA:
        # creamos un executor por biblioteca para que los deltas OMDb
        # no se contaminen con trabajo concurrente de otra biblioteca.
        # ------------------------------------------------------------------
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

                results.append(None)
                titles_for_index.append(title)
                years_for_index.append(year)

                fut = executor.submit(
                    analyze_movie,
                    movie_input,
                    source_movie=movie,
                )
                future_to_index[fut] = movie_index - 1
                lib_movies_enqueued += 1

                if SILENT_MODE and DEBUG_MODE and (movie_index % _PROGRESS_EVERY_N_MOVIES == 0):
                    _logger.progress(f"[PLEX][DEBUG] {lib_key}: encoladas {movie_index} películas...")

            for fut in as_completed(future_to_index):
                idx = future_to_index[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    title = titles_for_index[idx] if idx < len(titles_for_index) else ""
                    year = years_for_index[idx] if idx < len(years_for_index) else None
                    lib_movies_errors += 1
                    total_movies_errors += 1
                    _logger.error(
                        f"[PLEX] Error analizando '{title}' ({year or 'n/a'}) "
                        f"en '{lib_name}': {exc!r}"
                    )
                    results[idx] = None

                lib_movies_completed += 1

                if SILENT_MODE and DEBUG_MODE and (lib_movies_completed % _PROGRESS_EVERY_N_MOVIES == 0):
                    _logger.progress(
                        f"[PLEX][DEBUG] {lib_key}: completadas {lib_movies_completed}/{lib_movies_enqueued}..."
                    )

        for res in results:
            if res is None:
                continue

            row, meta_sugg, logs = res

            for log in logs:
                _logger.info(log)

            if row:
                all_rows.append(row)
                lib_rows_added += 1

            if meta_sugg:
                suggestion_rows.append(meta_sugg)
                lib_suggestions_added += 1

            total_movies_processed += 1

        t_lib_elapsed = time.monotonic() - t_lib

        if SILENT_MODE:
            _logger.progress(
                "[PLEX] Biblioteca finalizada: "
                f"{lib_name} | enqueued={lib_movies_enqueued} | "
                f"completed={lib_movies_completed} | "
                f"errors={lib_movies_errors} | rows={lib_rows_added} | "
                f"suggestions={lib_suggestions_added} | "
                f"time={t_lib_elapsed:.1f}s"
            )

            # --- Snapshot + delta por biblioteca (solo silent+debug) ---
            if DEBUG_MODE:
                snap_before = lib_omdb_snapshot_start.get(lib_key, {})
                snap_after = get_omdb_metrics_snapshot()
                delta = _metrics_diff(snap_before, snap_after)
                lib_omdb_delta[lib_key] = dict(delta)

                _log_omdb_metrics(prefix=f"[PLEX][DEBUG] {lib_key}: delta:", metrics=delta)

        elif DEBUG_MODE:
            _logger.info(
                "[PLEX][DEBUG] Biblioteca finalizada: "
                f"{lib_name} enqueued={lib_movies_enqueued} "
                f"completed={lib_movies_completed} errors={lib_movies_errors} "
                f"rows={lib_rows_added} suggestions={lib_suggestions_added} "
                f"time={t_lib_elapsed:.1f}s"
            )

    filtered = [r for r in all_rows if r.get("decision") in {"DELETE", "MAYBE"}]
    filtered = sort_filtered_rows(filtered)

    write_all_csv(REPORT_ALL_PATH, all_rows)
    write_filtered_csv(REPORT_FILTERED_PATH, filtered)
    write_suggestions_csv(METADATA_FIX_PATH, suggestion_rows)

    elapsed = time.monotonic() - t0

    if SILENT_MODE:
        decisions = _count_decisions(all_rows)

        _logger.progress(
            "[PLEX] Resumen final: "
            f"libraries={total_libs} excluded_libs={len(excluded)} "
            f"movies={total_movies_processed} errors={total_movies_errors} "
            f"workers={max_workers} time={elapsed:.1f}s | "
            f"rows={len(all_rows)} (KEEP={decisions['KEEP']} MAYBE={decisions['MAYBE']} "
            f"DELETE={decisions['DELETE']} UNKNOWN={decisions['UNKNOWN']}) | "
            f"filtered_rows={len(filtered)} suggestions={len(suggestion_rows)}"
        )

        _logger.progress(
            "[PLEX] CSVs: "
            f"all={REPORT_ALL_PATH} | filtered={REPORT_FILTERED_PATH} | suggestions={METADATA_FIX_PATH}"
        )

        # Métricas globales OMDb (solo silent+debug)
        _log_omdb_metrics(prefix="[PLEX][DEBUG] Global:")

        # Ranking automático (solo silent+debug) + Top 5 por coste total
        _log_omdb_rankings(lib_omdb_delta, min_groups=2)

    _logger.info("[PLEX] Análisis completado.", always=True)