from __future__ import annotations

"""
backend/analiza_dlna.py

Orquestador principal de análisis DLNA/UPnP (streaming).

Responsabilidades de este módulo (post-split)
---------------------------------------------
✅ Se queda con:
- Orquestación: scan -> analyze_movie -> writers (CSV) -> filtered final.
- Concurrencia: scan_workers / analyze_workers / inflight bounded.
- Dedupe GLOBAL por run (entre contenedores seleccionados).
- Métricas y resumen final.
- flush_external_caches() en finally.

🚫 Se delega en backend.dlna_client:
- discovery + selección de device (CLI)
- selección interactiva de contenedores de vídeo (root + folder-browse + multi-select)
- endpoints cache, SOAP Browse, DIDL parse
- traversal recursivo seguro (visited + fuses)
- contadores agregados (dlna.browse.*, dlna.xml_fetch.*) y circuit breaker (si aplica)

Política de logs (alineada con backend/logger.py)
------------------------------------------------
- Menús, prompts y validación de input: SIEMPRE visibles -> logger.info(..., always=True)
  (Este módulo no presenta menús DLNA; están en DLNAClient).
- Estado global del run sin “spam” -> logger.progress(...)
- Debug contextual -> logger.debug_ctx("DLNA", "...") (respeta DEBUG/SILENT en logger)
- Errores de pipeline siempre visibles -> logger.error(..., always=True)

Diseño / seguridad
------------------
Este módulo produce decisiones/reportes que pueden llevar a borrados aguas arriba.
Se prioriza:
- determinismo en outputs (NO-SILENT: orden estable por item)
- robustez (no romper pipeline ante DLNA “quirky”)
- logging controlado y sin duplicar UX
- límites duros (fuses) ya aplicados en DLNAClient traversal
"""

import re
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from typing import Protocol, TypeVar

from backend import logger as logger
from backend.collection_analysis import analyze_movie, flush_external_caches
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.config_omdb import OMDB_HTTP_MAX_CONCURRENCY, OMDB_HTTP_MIN_INTERVAL_SECONDS
from backend.config_plex import PLEX_ANALYZE_WORKERS
from backend.config_reports import METADATA_FIX_PATH, REPORT_ALL_PATH, REPORT_FILTERED_PATH
from backend.decision_logic import sort_filtered_rows
from backend.movie_input import MovieInput
from backend.omdb_client import get_omdb_metrics_snapshot, reset_omdb_metrics
from backend.reporting import (
    open_all_csv_writer,
    open_filtered_csv_writer_only_if_rows,
    open_suggestions_csv_writer,
)

from backend.dlna_client import (
    DLNAClient,
    DlnaContainer,
    DlnaVideoItem,
    TraversalLimits,
    build_traversal_limits,
)

# ---------------------------------------------------------------------------
# Opcional: métricas agregadas (si existe)
# ---------------------------------------------------------------------------
try:
    from backend.run_metrics import METRICS  # type: ignore
except Exception:  # pragma: no cover

    class _NoopMetrics:
        def incr(self, key: str, n: int = 1) -> None:
            return

        def observe_ms(self, key: str, ms: float) -> None:
            return

        def add_error(self, subsystem: str, action: str, *, endpoint: str | None, detail: str) -> None:
            return

        def snapshot(self) -> dict[str, object]:
            return {"counters": {}, "derived": {}, "timings_ms": {}, "errors": []}

    METRICS = _NoopMetrics()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Config best-effort para scan workers (compat)
# ---------------------------------------------------------------------------
try:
    from backend.config import DLNA_SCAN_WORKERS  # type: ignore
except Exception:  # pragma: no cover
    DLNA_SCAN_WORKERS = 2  # type: ignore

# ---------------------------------------------------------------------------
# Concurrency knobs locales
# ---------------------------------------------------------------------------
_PROGRESS_EVERY_N_ITEMS: int = 100
_MAX_WORKERS_CAP: int = 64
_DEFAULT_MAX_INFLIGHT_FACTOR: int = 4

# ---------------------------------------------------------------------------
# Parsing de título/año (sufijo "(YYYY)")
# ---------------------------------------------------------------------------
_TITLE_YEAR_SUFFIX_RE: re.Pattern[str] = re.compile(
    r"(?P<base>.*?)"
    r"(?P<sep>\s*\.?\s*)"
    r"\(\s*(?P<year>\d{4})\s*\)\s*$"
)

# ============================================================================
# TIPADO (writers)
# ============================================================================

_Row = dict[str, object]


class _RowWriter(Protocol):
    """Interfaz mínima que exponen nuestros CSV writers del módulo reporting."""
    def write_row(self, row: _Row) -> None: ...


_WAll = TypeVar("_WAll", bound=_RowWriter)
_WSugg = TypeVar("_WSugg", bound=_RowWriter)

# ============================================================================
# Dedupe global por run
# ============================================================================


@dataclass(slots=True)
class _GlobalDedupeState:
    """
    Dedupe global por run (entre contenedores seleccionados).

    Evita analizar el mismo fichero si aparece en varias carpetas/vistas.
    Claves:
    - resource_url (primaria)
    - item_id (refuerzo si existe)
    """
    seen_item_urls: set[str]
    seen_item_ids: set[str]
    skipped_global: int = 0


# ============================================================================
# OMDb metrics helpers (solo SILENT+DEBUG)
# ============================================================================


def _metrics_get_int(m: Mapping[str, object], key: str) -> int:
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
    """
    Delta (after - before) para un subconjunto estable de contadores OMDb.
    Útil para profiling en SILENT+DEBUG sin ensuciar consola normal.
    """
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
    Log compacto de métricas OMDb (solo cuando SILENT+DEBUG para no ensuciar consola).
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


def _compute_cost_score(delta: Mapping[str, int]) -> int:
    """
    Heurística simple: aproxima “coste” OMDb de un grupo de items.
    Útil para ranking (solo SILENT+DEBUG).
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


def _rank_top_by_total_cost(deltas_by_group: dict[str, dict[str, int]], top_n: int = 5) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = [(group, _compute_cost_score(d)) for group, d in deltas_by_group.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _log_omdb_rankings(deltas_by_group: dict[str, dict[str, int]], *, min_groups: int = 2) -> None:
    """
    Ranking por coste OMDb (solo SILENT+DEBUG) para detectar contenedores “caros”.
    """
    if not (SILENT_MODE and DEBUG_MODE):
        return
    if not deltas_by_group or len(deltas_by_group) < min_groups:
        return

    def _fmt(items: list[tuple[str, int]]) -> str:
        usable = [(name, val) for (name, val) in items if val > 0]
        return " | ".join([f"{i+1}) {name}: {val}" for i, (name, val) in enumerate(usable)])

    if (line := _fmt(_rank_top_by_total_cost(deltas_by_group, top_n=5))):
        logger.progress(f"[DLNA][DEBUG] Top containers by TOTAL_COST: {line}")


# ============================================================================
# Concurrency helpers
# ============================================================================


def _compute_max_workers(requested: int) -> int:
    """
    Decide nº de workers para ANALYZE con caps defensivos.

    Nota:
    - Cap global (MAX_WORKERS_CAP)
    - Cap por OMDb: más threads no ayudan si OMDb limita.
    """
    max_workers = int(requested)
    if max_workers < 1:
        max_workers = 1
    if max_workers > _MAX_WORKERS_CAP:
        max_workers = _MAX_WORKERS_CAP

    # Cap adicional por OMDb (evita hilos inútiles cuando el rate limiter manda)
    omdb_cap = max(4, int(OMDB_HTTP_MAX_CONCURRENCY) * 8)
    max_workers = min(max_workers, omdb_cap)
    return max(1, max_workers)


def _compute_max_inflight(max_workers: int) -> int:
    """
    Nº máximo de futures en vuelo (por contenedor) para backpressure.
    """
    inflight = max_workers * _DEFAULT_MAX_INFLIGHT_FACTOR
    return max(max_workers, inflight)


def _compute_scan_workers() -> int:
    """
    Workers para SCAN (Browse DLNA).
    Ligero: muchos servers DLNA se degradan con demasiadas peticiones.
    """
    try:
        w = int(DLNA_SCAN_WORKERS)
    except Exception:
        w = 2
    return max(1, min(8, w))


# ============================================================================
# Normalización título/año + file display
# ============================================================================


def _extract_year_from_title(title: str) -> tuple[str, int | None]:
    """
    Parsea sufijo “(YYYY)” al final si existe.
    Ej: 'Alien (1979)' -> ('Alien', 1979)
    """
    raw = title.strip()
    if not raw:
        return title, None

    match = _TITLE_YEAR_SUFFIX_RE.match(raw)
    if not match:
        return title, None

    year_str = match.group("year")
    if not year_str.isdigit():
        return title, None

    year = int(year_str)
    if not (1900 <= year <= 2100):
        return title, None

    base = match.group("base").strip()
    base = base.rstrip(".").strip()
    return (base or title), year


def _dlna_display_file(client: DLNAClient, library: str, raw_title: str, resource_url: str) -> tuple[str, str]:
    """
    Genera un `file_path` friendly y devuelve también la URL del recurso.

    Nota:
    - extensión best-effort (no rompe si no se puede).
    - Para DLNA usamos "library/title.ext" como convención humana.
    """
    ext = client.extract_ext_from_resource_url(resource_url)
    base = raw_title.strip() or "UNKNOWN"
    if ext and not base.lower().endswith(ext.lower()):
        base = f"{base}{ext}"
    return f"{library}/{base}", resource_url


def _format_item_progress_line(*, index: int, total: int, title: str, year: int | None, file_size_bytes: int | None) -> str:
    """
    Línea de progreso por item (modo NO-SILENT).
    En DEBUG, puede incluir size (si el server lo aporta).
    """
    base = title.strip() or "UNKNOWN"
    if year is not None:
        base = f"{base} ({year})"
    if DEBUG_MODE and file_size_bytes is not None and file_size_bytes >= 0:
        base = f"{base} [{file_size_bytes} bytes]"
    return f"({index}/{total}) {base}"


# ============================================================================
# Snapshot counters helper (robust)
# ============================================================================


def _safe_snapshot_counters() -> Mapping[str, object]:
    """
    Recupera `METRICS.snapshot()['counters']` sin romper el pipeline
    si el backend de métricas no existe o falla.
    """
    try:
        snap = METRICS.snapshot()
        if isinstance(snap, Mapping):
            c = snap.get("counters", {})
            return c if isinstance(c, Mapping) else {}
    except Exception:
        pass
    return {}


# ============================================================================
# Entry-point
# ============================================================================


def analyze_dlna_server(device=None) -> None:
    """
    Entry-point principal (API pública estable).

    Flujo:
    - Selecciona servidor DLNA (si no viene por parámetro) -> DLNAClient (UX).
    - Selección interactiva de contenedores (delegada a DLNAClient).
    - SCAN por contenedor:
        * NO-SILENT: secuencial (orden estable) y acumulación de candidatos por contenedor.
        * SILENT: concurrente (scan_workers) y streaming de análisis.
    - ANALYZE por item (cap por OMDb + bounded inflight).
    - CSVs:
        * all.csv + suggestions.csv: streaming
        * filtered.csv: al final (si hay filas) ordenado por decision_logic
    - Flush caches en finally.
    """
    t0 = time.monotonic()
    reset_omdb_metrics()

    client = DLNAClient()
    global_dedupe = _GlobalDedupeState(seen_item_urls=set(), seen_item_ids=set())
    limits: TraversalLimits = build_traversal_limits()

    try:
        # ------------------------------------------------------------
        # Selección de server (UX delegada a DLNAClient si no viene por parámetro)
        # ------------------------------------------------------------
        if device is None:
            device = client.ask_user_to_select_device()
            if device is None:
                return

        server_label = f"{device.friendly_name} ({device.host}:{device.port})"
        logger.progress(f"[DLNA] Servidor: {server_label}")

        if DEBUG_MODE:
            logger.debug_ctx("DLNA", f"location={device.location!r}")
            logger.debug_ctx(
                "DLNA",
                "traverse_limits: "
                f"max_depth={limits.max_depth} max_containers={limits.max_containers} "
                f"max_items_total={limits.max_items_total} max_empty_pages={limits.max_empty_pages} "
                f"max_pages_per_container={limits.max_pages_per_container}",
            )

        # ------------------------------------------------------------
        # Selección de contenedores (UX + heurística delegada en el cliente)
        # ------------------------------------------------------------
        picked = client.ask_user_to_select_video_containers(device)
        if picked is None:
            return

        chosen_root, selected_containers = picked

        # En NO-SILENT dejamos rastro humano adicional; en SILENT el cliente ya fue compacto.
        if not SILENT_MODE:
            logger.progress(f"[DLNA] Raíz de vídeo: {chosen_root.title}")

        # ------------------------------------------------------------
        # Concurrency
        # ------------------------------------------------------------
        scan_workers = _compute_scan_workers()
        analyze_workers = _compute_max_workers(PLEX_ANALYZE_WORKERS)
        max_inflight = _compute_max_inflight(analyze_workers)

        if SILENT_MODE:
            logger.progress(
                f"[DLNA] Concurrency: scan_workers={scan_workers} analyze_workers={analyze_workers} inflight_cap={max_inflight} | "
                f"(PLEX_ANALYZE_WORKERS={PLEX_ANALYZE_WORKERS}, OMDB_HTTP_MAX_CONCURRENCY={OMDB_HTTP_MAX_CONCURRENCY}, "
                f"OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})"
            )
        else:
            logger.debug_ctx(
                "DLNA",
                f"Concurrency: scan_workers={scan_workers} analyze_workers={analyze_workers} inflight_cap={max_inflight} "
                f"(cap por OMDb limiter, OMDB_HTTP_MIN_INTERVAL_SECONDS={OMDB_HTTP_MIN_INTERVAL_SECONDS})",
            )

        # ------------------------------------------------------------
        # Estado run
        # ------------------------------------------------------------
        filtered_rows: list[_Row] = []
        decisions_count: dict[str, int] = {"KEEP": 0, "MAYBE": 0, "DELETE": 0, "UNKNOWN": 0}
        total_items_processed = 0
        total_items_errors = 0
        total_rows_written = 0
        total_suggestions_written = 0
        container_omdb_delta_analyze: dict[str, dict[str, int]] = {}

        analyze_snapshot_start: dict[str, object] | None = None
        if SILENT_MODE and DEBUG_MODE:
            analyze_snapshot_start = dict(get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: start:")

        # ------------------------------------------------------------
        # Helpers internos
        # ------------------------------------------------------------
        def _maybe_print_item_logs(logs: list[str]) -> None:
            """
            Logs por item (de analyze_movie):
            - NO-SILENT: visibles (logger.info normal).
            - SILENT: solo visibles cuando DEBUG_MODE (always=True).
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
            d = row.get("decision")
            if d in ("KEEP", "MAYBE", "DELETE"):
                decisions_count[str(d)] += 1
            else:
                decisions_count["UNKNOWN"] += 1

        def _apply_global_dedupe(container_title: str, items: list[DlnaVideoItem]) -> list[DlnaVideoItem]:
            """
            Dedupe GLOBAL por run:
            - resource_url como clave primaria.
            - item_id como refuerzo (si existe).
            """
            if not items:
                return items

            out: list[DlnaVideoItem] = []
            for it in items:
                if it.item_id and it.item_id in global_dedupe.seen_item_ids:
                    global_dedupe.skipped_global += 1
                    continue
                if it.resource_url in global_dedupe.seen_item_urls:
                    global_dedupe.skipped_global += 1
                    continue

                if it.item_id:
                    global_dedupe.seen_item_ids.add(it.item_id)
                global_dedupe.seen_item_urls.add(it.resource_url)
                out.append(it)

            if DEBUG_MODE and (len(out) != len(items)):
                logger.debug_ctx(
                    "DLNA",
                    f"global_dedupe: container={container_title!r} in={len(items)} out={len(out)} skipped={len(items)-len(out)}",
                )
            return out

        def _analyze_one(
            *,
            raw_title: str,
            resource_url: str,
            file_size: int | None,
            library: str,
        ) -> tuple[_Row | None, _Row | None, list[str]]:
            """
            Analiza un item DLNA con el pipeline existente (analyze_movie).
            Importante: aquí NO hay UX; solo construcción de MovieInput + llamada.
            """
            clean_title, extracted_year = _extract_year_from_title(raw_title)
            display_file, file_url = _dlna_display_file(client, library, raw_title, resource_url)

            movie_input = MovieInput(
                source="dlna",
                library=library,
                title=clean_title,
                year=extracted_year,
                file_path=display_file,
                file_size_bytes=file_size,
                imdb_id_hint=None,
                plex_guid=None,
                rating_key=None,
                thumb_url=None,
                extra={"source_url": file_url},
            )

            row, meta_sugg, logs = analyze_movie(movie_input, source_movie=None)

            if row:
                row["file"] = display_file
                row["file_url"] = file_url

            return row, meta_sugg, logs

        def _handle_result(
            res: tuple[_Row | None, _Row | None, list[str]],
            *,
            all_writer: _WAll,
            sugg_writer: _WSugg,
        ) -> None:
            """
            Persiste resultados:
            - all.csv: streaming
            - suggestions.csv: streaming
            - filtered_rows: buffer para ordenación final
            """
            nonlocal total_rows_written, total_suggestions_written

            row, meta_sugg, logs = res
            _maybe_print_item_logs(logs)

            if row:
                all_writer.write_row(row)
                total_rows_written += 1
                _tally_decision(row)
                if row.get("decision") in {"DELETE", "MAYBE"}:
                    filtered_rows.append(dict(row))

            if meta_sugg:
                sugg_writer.write_row(meta_sugg)
                total_suggestions_written += 1

        # =========================================================================
        # Writers globales (streaming)
        # =========================================================================
        with open_all_csv_writer(REPORT_ALL_PATH) as all_writer, open_suggestions_csv_writer(METADATA_FIX_PATH) as sugg_writer:

            def _scan_container(c: DlnaContainer) -> tuple[str, list[DlnaVideoItem]]:
                """
                Escanea un contenedor y devuelve (título, items).

                Nota:
                - La robustez del browse/traversal y los fuses viven en DLNAClient.
                - Aquí solo medimos latencia agregada y devolvemos lista de items.
                """
                METRICS.incr("dlna.scan.containers")
                t_scan0 = time.monotonic()
                items, _stats = client.iter_video_items_recursive(device, c.object_id, limits=limits)
                METRICS.observe_ms("dlna.scan.container_latency_ms", (time.monotonic() - t_scan0) * 1000.0)
                return c.title, items

            if not SILENT_MODE:
                # ---------------- NO SILENT (determinismo: orden estable) ----------------
                candidates_by_container: dict[str, list[tuple[str, str, int | None, str]]] = {}
                total_candidates = 0

                total_containers = len(selected_containers)
                for idx, container in enumerate(selected_containers, start=1):
                    logger.progress(f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {container.title}")

                    items, _stats = client.iter_video_items_recursive(device, container.object_id, limits=limits)
                    items = _apply_global_dedupe(container.title, items)

                    if DEBUG_MODE:
                        logger.debug_ctx("DLNA", f"scan {container.title!r} items={len(items)}")

                    bucket = candidates_by_container.setdefault(container.title, [])
                    for it in items:
                        bucket.append((it.title, it.resource_url, it.size_bytes, container.title))
                    total_candidates += len(items)

                if total_candidates == 0:
                    logger.progress("[DLNA] No se han encontrado items de vídeo.")
                    return

                logger.progress(f"[DLNA] Candidatos a analizar: {total_candidates}")
                analyzed_so_far = 0

                # Importante: iteramos en el orden de inserción para mantener determinismo humano.
                for container_title, items in candidates_by_container.items():
                    if not items:
                        continue

                    logger.progress(f"[DLNA] Analizando contenedor: {container_title} (items={len(items)})")

                    future_to_index: dict[Future[tuple[_Row | None, _Row | None, list[str]]], int] = {}
                    pending_by_index: dict[int, tuple[_Row | None, _Row | None, list[str]]] = {}
                    next_to_write = 1
                    inflight: set[Future[tuple[_Row | None, _Row | None, list[str]]]] = set()

                    def _drain_completed(*, drain_all: bool) -> None:
                        """
                        Drena futures completados preservando orden por índice local.
                        Esto evita que el scheduling del ThreadPool cambie el orden de salida en consola/CSVs.
                        """
                        nonlocal next_to_write, total_items_processed, total_items_errors

                        while inflight:
                            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                            if not done:
                                return

                            for fut in done:
                                inflight.discard(fut)
                                idx_local = future_to_index.get(fut, -1)

                                try:
                                    res = fut.result()
                                except Exception as exc:  # pragma: no cover
                                    total_items_errors += 1
                                    METRICS.incr("dlna.analyze.future_errors")
                                    METRICS.add_error("dlna", "analyze_future", endpoint=None, detail=repr(exc))
                                    logger.error(f"[DLNA] Error analizando item (future): {exc!r}", always=True)
                                    total_items_processed += 1
                                    continue

                                if idx_local >= 1:
                                    pending_by_index[idx_local] = res

                                while next_to_write in pending_by_index:
                                    ready = pending_by_index.pop(next_to_write)
                                    _handle_result(ready, all_writer=all_writer, sugg_writer=sugg_writer)
                                    total_items_processed += 1
                                    next_to_write += 1

                            if not drain_all:
                                return

                    workers_here = min(analyze_workers, max(1, len(items)))
                    inflight_cap_here = min(max_inflight, max(1, len(items)))

                    with ThreadPoolExecutor(max_workers=workers_here) as executor:
                        for item_index, (raw_title, resource_url, file_size, library) in enumerate(items, start=1):
                            analyzed_so_far += 1
                            clean_title_preview, extracted_year_preview = _extract_year_from_title(raw_title)
                            display_title = raw_title if DEBUG_MODE else clean_title_preview

                            # NO-SILENT: progreso por item (humano).
                            logger.info(
                                _format_item_progress_line(
                                    index=analyzed_so_far,
                                    total=total_candidates,
                                    title=display_title,
                                    year=extracted_year_preview,
                                    file_size_bytes=file_size,
                                )
                            )

                            fut = executor.submit(
                                _analyze_one,
                                raw_title=raw_title,
                                resource_url=resource_url,
                                file_size=file_size,
                                library=library,
                            )

                            future_to_index[fut] = item_index
                            inflight.add(fut)

                            if len(inflight) >= inflight_cap_here:
                                _drain_completed(drain_all=False)

                        _drain_completed(drain_all=True)

                    # Extra safety: si quedara algo pendiente (no debería), lo drenamos en orden.
                    while next_to_write in pending_by_index:
                        ready = pending_by_index.pop(next_to_write)
                        _handle_result(ready, all_writer=all_writer, sugg_writer=sugg_writer)
                        total_items_processed += 1
                        next_to_write += 1

            else:
                # ---------------- SILENT (streaming + scan concurrente) ----------------
                total_containers = len(selected_containers)
                total_candidates = 0
                analyzed_so_far = 0

                with ThreadPoolExecutor(max_workers=scan_workers) as scan_pool, ThreadPoolExecutor(
                    max_workers=analyze_workers
                ) as analyze_pool:
                    scan_futs: list[Future[tuple[str, list[DlnaVideoItem]]]] = []
                    for idx, c in enumerate(selected_containers, start=1):
                        logger.progress(f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {c.title}")
                        scan_futs.append(scan_pool.submit(_scan_container, c))

                    for fut in as_completed(scan_futs):
                        try:
                            container_title, items = fut.result()
                        except Exception as exc:  # pragma: no cover
                            METRICS.incr("dlna.scan.errors")
                            METRICS.add_error("dlna", "scan_container", endpoint=None, detail=repr(exc))
                            logger.error(f"[DLNA] Error escaneando contenedor (future): {exc!r}", always=True)
                            continue

                        items = _apply_global_dedupe(container_title, items)
                        container_key = container_title or "<container>"
                        total_candidates += len(items)

                        if DEBUG_MODE:
                            logger.debug_ctx("DLNA", f"scan {container_title!r} items={len(items)}")

                        if not items:
                            continue

                        logger.progress(f"[DLNA] Analizando contenedor: {container_title} (items={len(items)})")

                        analyze_snap_start_container: dict[str, object] | None = None
                        if DEBUG_MODE:
                            analyze_snap_start_container = dict(get_omdb_metrics_snapshot())
                            _log_omdb_metrics(prefix=f"[DLNA][DEBUG] {container_key}: analyze:start:")

                        inflight: set[Future[tuple[_Row | None, _Row | None, list[str]]]] = set()
                        inflight_cap_here = min(max_inflight, max(1, len(items)))

                        def _drain_completed(*, drain_all: bool) -> None:
                            nonlocal total_items_processed, total_items_errors

                            while inflight:
                                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                                if not done:
                                    return

                                for af in done:
                                    inflight.discard(af)
                                    try:
                                        res = af.result()
                                    except Exception as exc:  # pragma: no cover
                                        total_items_errors += 1
                                        METRICS.incr("dlna.analyze.future_errors")
                                        METRICS.add_error("dlna", "analyze_future", endpoint=None, detail=repr(exc))
                                        logger.error(f"[DLNA] Error analizando item (future): {exc!r}", always=True)
                                        total_items_processed += 1
                                        continue

                                    _handle_result(res, all_writer=all_writer, sugg_writer=sugg_writer)
                                    total_items_processed += 1

                                if not drain_all:
                                    return

                        for it in items:
                            analyzed_so_far += 1
                            if DEBUG_MODE and (analyzed_so_far % _PROGRESS_EVERY_N_ITEMS == 0):
                                logger.progress(f"[DLNA][DEBUG] Progreso: analizados {analyzed_so_far} items...")

                            inflight.add(
                                analyze_pool.submit(
                                    _analyze_one,
                                    raw_title=it.title,
                                    resource_url=it.resource_url,
                                    file_size=it.size_bytes,
                                    library=container_title,
                                )
                            )

                            if len(inflight) >= inflight_cap_here:
                                _drain_completed(drain_all=False)

                        _drain_completed(drain_all=True)

                        if DEBUG_MODE and analyze_snap_start_container is not None:
                            analyze_delta_container = _metrics_diff(analyze_snap_start_container, get_omdb_metrics_snapshot())
                            container_omdb_delta_analyze[container_key] = dict(analyze_delta_container)
                            _log_omdb_metrics(
                                prefix=f"[DLNA][DEBUG] {container_key}: analyze:delta:",
                                metrics=analyze_delta_container,
                            )

                # Si no hay candidatos y tampoco hay errores, reportamos “no items” de forma limpia.
                if total_candidates == 0 and int(_safe_snapshot_counters().get("dlna.scan.errors", 0) or 0) == 0:
                    logger.progress("[DLNA] No se han encontrado items de vídeo.")
                    return

                logger.progress(f"[DLNA] Candidatos detectados (streaming): {total_candidates}")

        # =========================================================================
        # filtered.csv (solo si hay filas) + resumen final
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

        if SILENT_MODE and DEBUG_MODE and analyze_snapshot_start is not None:
            analyze_delta = _metrics_diff(analyze_snapshot_start, get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: delta:", metrics=analyze_delta)

        # Contadores coherentes con dlna_client.py (resilient_call):
        counters = _safe_snapshot_counters()
        dlna_scan_errors = int(counters.get("dlna.scan.errors", 0) or 0) + int(counters.get("dlna.browse.errors", 0) or 0)
        dlna_circuit_blocks = int(counters.get("dlna.browse.blocked_by_circuit", 0) or 0) + int(
            counters.get("dlna.xml_fetch.blocked_by_circuit", 0) or 0
        )

        if SILENT_MODE:
            logger.progress(
                "[DLNA] Resumen final: "
                f"server={server_label} containers={len(selected_containers)} "
                f"scan_workers={_compute_scan_workers()} analyze_workers={_compute_max_workers(PLEX_ANALYZE_WORKERS)} "
                f"inflight_cap={_compute_max_inflight(_compute_max_workers(PLEX_ANALYZE_WORKERS))} "
                f"time={elapsed:.1f}s | "
                f"scan_errors={dlna_scan_errors} circuit_blocks={dlna_circuit_blocks} analysis_errors={total_items_errors} | "
                f"items={total_items_processed} rows={total_rows_written} "
                f"(KEEP={decisions_count['KEEP']} MAYBE={decisions_count['MAYBE']} "
                f"DELETE={decisions_count['DELETE']} UNKNOWN={decisions_count['UNKNOWN']}) | "
                f"filtered_rows={filtered_len} filtered_csv={filtered_csv_status} "
                f"suggestions={total_suggestions_written}"
            )

            if global_dedupe.skipped_global > 0:
                logger.progress(
                    f"[DLNA] Dedupe global: evitados {global_dedupe.skipped_global} items duplicados entre contenedores."
                )

            logger.progress(
                "[DLNA] CSVs: "
                f"all={REPORT_ALL_PATH} | suggestions={METADATA_FIX_PATH} | filtered={REPORT_FILTERED_PATH}"
            )

            _log_omdb_metrics(prefix="[DLNA][DEBUG] Global:")

            if DEBUG_MODE and container_omdb_delta_analyze:
                logger.progress("[DLNA][DEBUG] Rankings (analyze deltas):")
                _log_omdb_rankings(container_omdb_delta_analyze, min_groups=2)

        logger.info("[DLNA] Análisis completado.", always=True)

    finally:
        # Flush agregado (una vez por run) aunque haya returns/excepciones
        try:
            flush_external_caches()
        except Exception as exc:  # pragma: no cover
            if DEBUG_MODE:
                logger.debug_ctx("DLNA", f"flush_external_caches failed: {exc!r}")


__all__ = ["analyze_dlna_server"]