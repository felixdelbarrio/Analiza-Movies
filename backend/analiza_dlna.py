"""
backend/analiza_dlna.py

Orquestador principal de análisis DLNA/UPnP (streaming).

MEJORAS (objetivo: menos UNKNOWN):
1) Extraer year también desde resource_url/filename (y/o display_file), no solo desde el título UPnP.
2) Extraer imdb_id_hint si aparece tt\\d+ en nombre/URL (suele ocurrir en librerías bien nombradas).
3) Mejorar el “título candidato” usando el filename cuando it.title es genérico
   (ej. “Aardman Classics”, “Vol.2”, “Scrat Pack”, etc.).

Nota: apoyado en backend/title_utils.py para no duplicar heurísticas.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from typing import Protocol, TypeAlias
from urllib.parse import unquote, urlparse

from backend import logger as logger
from backend.collection_analysis import analyze_movie, flush_external_caches
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.config_omdb import (
    OMDB_HTTP_MAX_CONCURRENCY,
    OMDB_HTTP_MIN_INTERVAL_SECONDS,
)
from backend.config_plex import PLEX_ANALYZE_WORKERS
from backend.config_reports import (
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from backend.decision_logic import sort_filtered_rows
from backend.movie_input import MovieInput
from backend.omdb_client import get_omdb_metrics_snapshot, reset_omdb_metrics
from backend.reporting import (
    open_all_csv_writer,
    open_filtered_csv_writer_only_if_rows,
    open_suggestions_csv_writer,
)
from backend.title_utils import (
    clean_title_candidate,
    extract_imdb_id_from_text,
    extract_year_from_text,
    filename_stem,
    split_title_and_year_from_text,
)

from backend.dlna_client import (
    DLNAClient,
    DLNADevice,
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

        def add_error(
            self, subsystem: str, action: str, *, endpoint: str | None, detail: str
        ) -> None:
            return

        def snapshot(self) -> dict[str, object]:
            return {"counters": {}, "derived": {}, "timings_ms": {}, "errors": []}

    METRICS = _NoopMetrics()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Config best-effort para scan workers
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
# Parsing legacy: título/año (sufijo "(YYYY)")
# ---------------------------------------------------------------------------
_TITLE_YEAR_SUFFIX_RE: re.Pattern[str] = re.compile(
    r"(?P<base>.*?)" r"(?P<sep>\s*\.?\s*)" r"\(\s*(?P<year>\d{4})\s*\)\s*$"
)

# ---------------------------------------------------------------------------
# Heurística: títulos “genéricos” (contienen palabras tipo collection/volume/etc.)
#   Conservador: si duda => NO considerarlo genérico.
# ---------------------------------------------------------------------------
_GENERIC_TITLE_RE: re.Pattern[str] = re.compile(
    r"""
    \b(
        collection|collections|classics|anthology|anthologies|pack|
        volume|vol|vol\.|season|series|saga|
        disc|disk|cd|dvd|bd|bluray|
        extras?|specials?|bonus|
        part|chapter|episode|
        various|varios|varias|mix|mixed|
        films?|movies?|videos?
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_GENERIC_SHORT_RE: re.Pattern[str] = re.compile(
    r"^(?:vol(?:\.|ume)?\s*\d+|disc\s*\d+|cd\s*\d+|part\s*\d+)$", re.IGNORECASE
)

# ============================================================================
# TIPADO (writers)
# ============================================================================

_Row = dict[str, object]


class _RowWriter(Protocol):
    """Interfaz mínima que exponen nuestros CSV writers del módulo reporting."""

    def write_row(self, row: _Row) -> None: ...


# Result y Future types (evita líos de inference)
_AnalyzeResult: TypeAlias = tuple[_Row | None, _Row | None, list[str]]
_AnalyzeFuture: TypeAlias = Future[_AnalyzeResult]

# Scan result types (para no mezclar con AnalyzeResult)
_ScanResult: TypeAlias = tuple[str, list[DlnaVideoItem]]
_ScanFuture: TypeAlias = Future[_ScanResult]


# ============================================================================
# Dedupe global por run
# ============================================================================


@dataclass(slots=True)
class _GlobalDedupeState:
    """Dedupe global por run (entre contenedores seleccionados)."""

    seen_item_urls: set[str]
    seen_item_ids: set[str]
    skipped_global: int = 0


# ============================================================================
# Helpers “typing-safe”
# ============================================================================


def _counter_int(counters: Mapping[str, object], key: str, default: int = 0) -> int:
    """
    Convierte un contador (object) a int de forma segura.
    Acepta int/float/bool/str numérico. Cualquier otra cosa => default.
    """
    v = counters.get(key, default)

    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        return default
    return default


def _as_object_mapping(d: Mapping[str, int]) -> Mapping[str, object]:
    # dict[str, int] es un Mapping, pero Pyright/Pylance no siempre permite “upcast”
    return {k: v for k, v in d.items()}


# ============================================================================
# OMDb metrics helpers (solo SILENT+DEBUG)
# ============================================================================


def _metrics_get_int(m: Mapping[str, object], key: str) -> int:
    v = m.get(key, 0)
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        return int(s) if s.isdigit() else 0
    return 0


def _metrics_diff(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, int]:
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


def _log_omdb_rankings(
    deltas_by_group: dict[str, dict[str, int]], *, min_groups: int = 2
) -> None:
    if not (SILENT_MODE and DEBUG_MODE):
        return
    if len(deltas_by_group) < min_groups:
        return

    usable = [(name, _compute_cost_score(d)) for name, d in deltas_by_group.items()]
    usable = [(n, v) for (n, v) in usable if v > 0]
    usable.sort(key=lambda x: x[1], reverse=True)

    if not usable:
        return

    line = " | ".join(
        [f"{i + 1}) {name}: {val}" for i, (name, val) in enumerate(usable[:5])]
    )
    logger.progress(f"[DLNA][DEBUG] Top containers by TOTAL_COST: {line}")


# ============================================================================
# Concurrency helpers
# ============================================================================


def _compute_max_workers(requested: int) -> int:
    max_workers = int(requested)
    if max_workers < 1:
        max_workers = 1
    if max_workers > _MAX_WORKERS_CAP:
        max_workers = _MAX_WORKERS_CAP

    omdb_cap = max(4, int(OMDB_HTTP_MAX_CONCURRENCY) * 8)
    return max(1, min(max_workers, omdb_cap))


def _compute_max_inflight(max_workers: int) -> int:
    inflight = max_workers * _DEFAULT_MAX_INFLIGHT_FACTOR
    return max(max_workers, inflight)


def _compute_scan_workers() -> int:
    try:
        w = int(DLNA_SCAN_WORKERS)
    except Exception:
        w = 2
    return max(1, min(8, w))


# ============================================================================
# Normalización título/año + file display
# ============================================================================


def _extract_year_from_title(title: str) -> tuple[str, int | None]:
    # Legacy: intenta "(YYYY)" al final
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

    base = match.group("base").strip().rstrip(".").strip()
    return (base or title), year


def _dlna_display_file(
    client: DLNAClient, library: str, raw_title: str, resource_url: str
) -> tuple[str, str]:
    ext = client.extract_ext_from_resource_url(resource_url)
    base = raw_title.strip() or "UNKNOWN"
    if ext and not base.lower().endswith(ext.lower()):
        base = f"{base}{ext}"
    lib = library.strip() or "DLNA"
    return f"{lib}/{base}", resource_url


def _format_item_progress_line(
    *, index: int, total: int, title: str, year: int | None, file_size_bytes: int | None
) -> str:
    base = title.strip() or "UNKNOWN"
    if year is not None:
        base = f"{base} ({year})"
    if DEBUG_MODE and file_size_bytes is not None and file_size_bytes >= 0:
        base = f"{base} [{file_size_bytes} bytes]"
    return f"({index}/{total}) {base}"


# ============================================================================
# URL/filename extraction helpers (apoyados en title_utils)
# ============================================================================


def _url_to_filename(url: str) -> str:
    """
    Extrae un filename “humano” desde una URL (best-effort), sin tocar disco.
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        path = unquote(p.path or "")
    except Exception:
        path = u
    # El último segmento suele ser el filename
    seg = path.rsplit("/", 1)[-1]
    return seg.strip()


def _best_filename_stem_from_url(url: str) -> str:
    """
    Devuelve un stem usable desde la URL:
      - parse + unquote
      - filename_stem() para quitar extensión
      - clean_title_candidate() para quitar ruido conservador (sin lookup-agresivo)
    """
    fn = _url_to_filename(url)
    st = filename_stem(fn)
    return clean_title_candidate(st)


def _title_from_filename(url: str) -> tuple[str, int | None]:
    """
    Best-effort: extrae un (title, year) desde el "filename" contenido en una URL/path.
    En DLNA la resource_url suele incluir el nombre del fichero (o parte del path),
    aunque no sea una URL "semántica".

    Estrategia:
      1) Parse + unquote del path
      2) basename (último segmento) como filename
      3) filename_stem() para quitar extensión
      4) clean_title_candidate() para normalizar separadores y quitar ruido conservador
      5) split_title_and_year_from_text() para separar "Title (1999)" / "Title - 1999" / etc.
      6) fallback year: extract_year_from_text(url) si no salió del stem limpio

    Devuelve:
      - title: "" si no se puede inferir nada
      - year: int o None
    """
    u = (url or "").strip()
    if not u:
        return "", None

    # 1) intenta parsear como URL estándar (http/file/etc.)
    # si falla, tratamos el input como un path "crudo"
    try:
        parsed = urlparse(u)
        path = unquote(parsed.path or "") or u
    except Exception:
        path = u

    # 2) filename = último segmento del path (si hay /)
    # nota: si no hay '/', rsplit devuelve el mismo string
    filename = path.rsplit("/", 1)[-1].strip()
    if not filename:
        # fallback: intenta con el input original por si era algo raro
        filename = u.rsplit("/", 1)[-1].strip()

    # 3) stem sin extensión
    stem = filename_stem(filename)
    if not stem:
        # nada que rascar
        year_fallback = extract_year_from_text(u)
        return "", year_fallback

    # 4) limpieza conservadora (separadores, bracket-noise si knob, etc.)
    cleaned = clean_title_candidate(stem)
    if not cleaned:
        year_fallback = extract_year_from_text(u)
        return "", year_fallback

    # 5) split título/año si viene trailing (o año “en medio” conservador)
    title_part, year = split_title_and_year_from_text(cleaned)
    title_part = (title_part or "").strip()

    # 6) fallback year desde URL completa si no salió del stem limpio
    if year is None:
        year = extract_year_from_text(u)

    # si el split devolvió vacío (raro), usa cleaned como título
    if not title_part:
        title_part = cleaned.strip()

    return title_part, year


def _is_generic_title(title: str) -> bool:
    """
    Heurística local para decidir si el it.title es genérico y conviene reemplazarlo por filename.
    Conservador: devuelve True solo si hay señales claras.
    """
    t = (title or "").strip()
    if not t:
        return True
    low = t.lower().strip()

    # Muy corto y con pinta de “vol2/disc1”
    if _GENERIC_SHORT_RE.match(low):
        return True

    # Tiene tokens típicos de colecciones/volúmenes
    if _GENERIC_TITLE_RE.search(low):
        # si además es muy corto, más seguro
        if len(low) <= 20:
            return True

    # Muchos dígitos y poca letra => sospechoso (ej. "01", "Movie 2")
    letters = sum(1 for ch in low if ch.isalpha())
    digits = sum(1 for ch in low if ch.isdigit())
    if digits >= 2 and letters <= 3:
        return True

    return False


def _derive_best_title_year_imdb(
    *,
    raw_title: str,
    resource_url: str,
    display_file: str,
) -> tuple[str, int | None, str | None, dict[str, object]]:
    """
    Devuelve (title, year, imdb_hint, extra_patch)
    - title: preferimos raw_title, salvo que sea genérico -> filename stem.
    - year: primero desde raw_title (legacy), luego desde filename/url/display_file.
    - imdb_hint: tt* desde url/filename/raw_title.
    - extra_patch: campos para MovieInput.extra (debug/observabilidad).
    """
    extra_patch: dict[str, object] = {}

    # 1) title/year desde el título UPnP (legacy)
    clean_title_legacy, year_legacy = _extract_year_from_title(raw_title)
    title_candidate = clean_title_legacy.strip() or raw_title.strip() or "UNKNOWN"

    # 2) filename stem (desde URL) como fuente secundaria (título/año)
    fn_split_title, fn_year = _title_from_filename(resource_url)
    fn_title = fn_split_title

    # 3) year fallback desde URL/filename/display_file
    # (extract_year_from_text es conservador)
    url_year = extract_year_from_text(resource_url)
    file_year = extract_year_from_text(display_file)

    # 4) imdb_id_hint desde cualquier texto (primero URL, luego filename/display/title)
    imdb_hint = extract_imdb_id_from_text(resource_url)
    if imdb_hint is None and fn_title:
        imdb_hint = extract_imdb_id_from_text(fn_title)
    if imdb_hint is None:
        imdb_hint = extract_imdb_id_from_text(display_file)
    if imdb_hint is None:
        imdb_hint = extract_imdb_id_from_text(raw_title)

    # 5) Decide título final: si it.title es genérico, usar filename split title si es usable
    used_filename_title = False
    if _is_generic_title(title_candidate) and fn_split_title.strip():
        title_candidate = fn_split_title.strip()
        used_filename_title = True

    # 6) Decide year final
    year_final = year_legacy
    if year_final is None:
        year_final = fn_year
    if year_final is None:
        year_final = url_year
    if year_final is None:
        year_final = file_year

    # 7) Observabilidad (no afecta lógica)
    extra_patch["raw_title"] = raw_title
    extra_patch["filename_title_candidate"] = fn_title
    extra_patch["filename_title_used"] = used_filename_title
    extra_patch["year_from_title"] = year_legacy
    extra_patch["year_from_filename"] = fn_year
    extra_patch["year_from_url"] = url_year
    extra_patch["year_from_display_file"] = file_year
    if imdb_hint:
        extra_patch["imdb_id_hint_from_text"] = imdb_hint

    return title_candidate, year_final, imdb_hint, extra_patch


# ============================================================================
# Snapshot counters helper (robust)
# ============================================================================


def _safe_snapshot_counters() -> Mapping[str, object]:
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


def analyze_dlna_server(device: DLNADevice | None = None) -> None:
    t0 = time.monotonic()
    reset_omdb_metrics()

    client = DLNAClient()
    global_dedupe = _GlobalDedupeState(seen_item_urls=set(), seen_item_ids=set())
    limits: TraversalLimits = build_traversal_limits()

    try:
        if device is None:
            picked = client.ask_user_to_select_device()
            if picked is None:
                return
            device = picked

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

        picked_containers = client.ask_user_to_select_video_containers(device)
        if picked_containers is None:
            return

        chosen_root, selected_containers = picked_containers

        if not SILENT_MODE:
            logger.progress(f"[DLNA] Raíz de vídeo: {chosen_root.title}")

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

        filtered_rows: list[_Row] = []
        decisions_count: dict[str, int] = {
            "KEEP": 0,
            "MAYBE": 0,
            "DELETE": 0,
            "UNKNOWN": 0,
        }
        total_items_processed = 0
        total_items_errors = 0
        total_rows_written = 0
        total_suggestions_written = 0
        container_omdb_delta_analyze: dict[str, dict[str, int]] = {}

        analyze_snapshot_start: dict[str, object] | None = None
        if SILENT_MODE and DEBUG_MODE:
            analyze_snapshot_start = dict(get_omdb_metrics_snapshot())
            _log_omdb_metrics(prefix="[DLNA][DEBUG] analyze: start:")

        def _maybe_print_item_logs(logs: list[str]) -> None:
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

        def _apply_global_dedupe(
            container_title: str, items: list[DlnaVideoItem]
        ) -> list[DlnaVideoItem]:
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
                    f"global_dedupe: container={container_title!r} in={len(items)} out={len(out)} skipped={len(items) - len(out)}",
                )
            return out

        def _analyze_one(
            *,
            raw_title: str,
            resource_url: str,
            file_size: int | None,
            library: str,
        ) -> _AnalyzeResult:
            # Construimos un "file_path" lógico + URL para inferencias
            display_file, file_url = _dlna_display_file(
                client, library, raw_title, resource_url
            )

            # ✅ NUEVO: title/year/imdb_hint best-effort desde title + filename/url/display_file
            best_title, best_year, imdb_hint, extra_patch = (
                _derive_best_title_year_imdb(
                    raw_title=raw_title,
                    resource_url=file_url,
                    display_file=display_file,
                )
            )

            movie_input = MovieInput(
                source="dlna",
                library=library,
                title=best_title,
                year=best_year,
                file_path=display_file,
                file_size_bytes=file_size,
                imdb_id_hint=imdb_hint,
                plex_guid=None,
                rating_key=None,
                thumb_url=None,
                extra={
                    "source_url": file_url,
                    "display_title": best_title,
                    "display_year": best_year,
                    **extra_patch,
                },
            )

            row, meta_sugg, logs = analyze_movie(movie_input, source_movie=None)

            if row is not None:
                row["file"] = display_file
                row["file_url"] = file_url

            return row, meta_sugg, logs

        def _handle_result(
            res: _AnalyzeResult,
            *,
            all_writer: _RowWriter,
            sugg_writer: _RowWriter,
        ) -> None:
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

        with (
            open_all_csv_writer(REPORT_ALL_PATH) as all_writer,
            open_suggestions_csv_writer(METADATA_FIX_PATH) as sugg_writer,
        ):

            def _scan_container(c: DlnaContainer) -> _ScanResult:
                METRICS.incr("dlna.scan.containers")
                t_scan0 = time.monotonic()
                items, _stats = client.iter_video_items_recursive(
                    device, c.object_id, limits=limits
                )
                METRICS.observe_ms(
                    "dlna.scan.container_latency_ms",
                    (time.monotonic() - t_scan0) * 1000.0,
                )
                return c.title, items

            # ============================================================
            # MODO INTERACTIVO (no-silent): escaneamos secuencial y analizamos por contenedor
            # ============================================================
            if not SILENT_MODE:
                candidates_by_container: dict[str, list[DlnaVideoItem]] = {}
                total_candidates = 0

                total_containers = len(selected_containers)
                for idx, container in enumerate(selected_containers, start=1):
                    logger.progress(
                        f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {container.title}"
                    )

                    items, _stats = client.iter_video_items_recursive(
                        device, container.object_id, limits=limits
                    )
                    items = _apply_global_dedupe(container.title, items)

                    if DEBUG_MODE:
                        logger.debug_ctx(
                            "DLNA", f"scan {container.title!r} items={len(items)}"
                        )

                    bucket = candidates_by_container.setdefault(container.title, [])
                    bucket.extend(items)
                    total_candidates += len(items)

                if total_candidates == 0:
                    logger.progress("[DLNA] No se han encontrado items de vídeo.")
                    return

                logger.progress(f"[DLNA] Candidatos a analizar: {total_candidates}")
                analyzed_so_far = 0

                for container_title, items in candidates_by_container.items():
                    if not items:
                        continue

                    logger.progress(
                        f"[DLNA] Analizando contenedor: {container_title} (items={len(items)})"
                    )

                    future_to_index: dict[_AnalyzeFuture, int] = {}
                    pending_by_index: dict[int, _AnalyzeResult] = {}
                    next_to_write = 1

                    # ✅ renombrado para evitar [no-redef] con el modo SILENT
                    inflight_interactive: set[_AnalyzeFuture] = set()

                    def _drain_completed_interactive(*, drain_all: bool) -> None:
                        nonlocal \
                            next_to_write, \
                            total_items_processed, \
                            total_items_errors

                        while inflight_interactive:
                            done, _ = wait(
                                inflight_interactive, return_when=FIRST_COMPLETED
                            )
                            if not done:
                                return

                            for fut in done:
                                inflight_interactive.discard(fut)
                                idx_local = future_to_index.get(fut, -1)

                                try:
                                    res = fut.result()
                                except Exception as exc:  # pragma: no cover
                                    total_items_errors += 1
                                    METRICS.incr("dlna.analyze.future_errors")
                                    METRICS.add_error(
                                        "dlna",
                                        "analyze_future",
                                        endpoint=None,
                                        detail=repr(exc),
                                    )
                                    logger.error(
                                        f"[DLNA] Error analizando item (future): {exc!r}",
                                        always=True,
                                    )
                                    total_items_processed += 1
                                    continue

                                if idx_local >= 1:
                                    pending_by_index[idx_local] = res

                                while next_to_write in pending_by_index:
                                    ready = pending_by_index.pop(next_to_write)
                                    _handle_result(
                                        ready,
                                        all_writer=all_writer,
                                        sugg_writer=sugg_writer,
                                    )
                                    total_items_processed += 1
                                    next_to_write += 1

                            if not drain_all:
                                return

                    workers_here = min(analyze_workers, max(1, len(items)))
                    inflight_cap_here = min(max_inflight, max(1, len(items)))

                    with ThreadPoolExecutor(max_workers=workers_here) as executor:
                        for item_index, it in enumerate(items, start=1):
                            analyzed_so_far += 1

                            # ✅ progreso: usa year derivable del propio it.title (rápido), sin parsear URL aquí
                            clean_title_preview, extracted_year_preview = (
                                _extract_year_from_title(it.title)
                            )
                            display_title = (
                                it.title if DEBUG_MODE else clean_title_preview
                            )

                            logger.info(
                                _format_item_progress_line(
                                    index=analyzed_so_far,
                                    total=total_candidates,
                                    title=display_title,
                                    year=extracted_year_preview,
                                    file_size_bytes=it.size_bytes,
                                )
                            )

                            fut = executor.submit(
                                _analyze_one,
                                raw_title=it.title,
                                resource_url=it.resource_url,
                                file_size=it.size_bytes,
                                library=container_title,
                            )
                            future_to_index[fut] = item_index
                            inflight_interactive.add(fut)

                            if len(inflight_interactive) >= inflight_cap_here:
                                _drain_completed_interactive(drain_all=False)

                        _drain_completed_interactive(drain_all=True)

                    while next_to_write in pending_by_index:
                        ready = pending_by_index.pop(next_to_write)
                        _handle_result(
                            ready, all_writer=all_writer, sugg_writer=sugg_writer
                        )
                        total_items_processed += 1
                        next_to_write += 1

            # ============================================================
            # MODO SILENT: scanning concurrente + análisis en streaming
            # ============================================================
            else:
                total_containers = len(selected_containers)
                total_candidates = 0
                analyzed_so_far = 0

                with (
                    ThreadPoolExecutor(max_workers=scan_workers) as scan_pool,
                    ThreadPoolExecutor(max_workers=analyze_workers) as analyze_pool,
                ):
                    scan_futures: list[_ScanFuture] = []
                    for idx, c in enumerate(selected_containers, start=1):
                        logger.progress(
                            f"[DLNA] Escaneando contenedor ({idx}/{total_containers}): {c.title}"
                        )
                        scan_futures.append(scan_pool.submit(_scan_container, c))

                    for sf in as_completed(scan_futures):
                        try:
                            container_title, items = sf.result()
                        except Exception as exc:  # pragma: no cover
                            METRICS.incr("dlna.scan.errors")
                            METRICS.add_error(
                                "dlna",
                                "scan_container",
                                endpoint=None,
                                detail=repr(exc),
                            )
                            logger.error(
                                f"[DLNA] Error escaneando contenedor (future): {exc!r}",
                                always=True,
                            )
                            continue

                        items = _apply_global_dedupe(container_title, items)
                        container_key = container_title or "<container>"
                        total_candidates += len(items)

                        if DEBUG_MODE:
                            logger.debug_ctx(
                                "DLNA", f"scan {container_title!r} items={len(items)}"
                            )

                        if not items:
                            continue

                        logger.progress(
                            f"[DLNA] Analizando contenedor: {container_title} (items={len(items)})"
                        )

                        analyze_snap_start_container: dict[str, object] | None = None
                        if DEBUG_MODE:
                            analyze_snap_start_container = dict(
                                get_omdb_metrics_snapshot()
                            )
                            _log_omdb_metrics(
                                prefix=f"[DLNA][DEBUG] {container_key}: analyze:start:"
                            )

                        inflight_silent: set[_AnalyzeFuture] = set()
                        inflight_cap_here = min(max_inflight, max(1, len(items)))

                        def _drain_completed_silent(*, drain_all: bool) -> None:
                            nonlocal total_items_processed, total_items_errors

                            while inflight_silent:
                                done, _ = wait(
                                    inflight_silent, return_when=FIRST_COMPLETED
                                )
                                if not done:
                                    return

                                for af in done:
                                    inflight_silent.discard(af)
                                    try:
                                        res = af.result()
                                    except Exception as exc:  # pragma: no cover
                                        total_items_errors += 1
                                        METRICS.incr("dlna.analyze.future_errors")
                                        METRICS.add_error(
                                            "dlna",
                                            "analyze_future",
                                            endpoint=None,
                                            detail=repr(exc),
                                        )
                                        logger.error(
                                            f"[DLNA] Error analizando item (future): {exc!r}",
                                            always=True,
                                        )
                                        total_items_processed += 1
                                        continue

                                    _handle_result(
                                        res,
                                        all_writer=all_writer,
                                        sugg_writer=sugg_writer,
                                    )
                                    total_items_processed += 1

                                if not drain_all:
                                    return

                        for it in items:
                            analyzed_so_far += 1
                            if DEBUG_MODE and (
                                analyzed_so_far % _PROGRESS_EVERY_N_ITEMS == 0
                            ):
                                logger.progress(
                                    f"[DLNA][DEBUG] Progreso: analizados {analyzed_so_far} items..."
                                )

                            inflight_silent.add(
                                analyze_pool.submit(
                                    _analyze_one,
                                    raw_title=it.title,
                                    resource_url=it.resource_url,
                                    file_size=it.size_bytes,
                                    library=container_title,
                                )
                            )

                            if len(inflight_silent) >= inflight_cap_here:
                                _drain_completed_silent(drain_all=False)

                        _drain_completed_silent(drain_all=True)

                        if DEBUG_MODE and analyze_snap_start_container is not None:
                            analyze_delta_container = _metrics_diff(
                                analyze_snap_start_container,
                                get_omdb_metrics_snapshot(),
                            )
                            container_omdb_delta_analyze[container_key] = dict(
                                analyze_delta_container
                            )
                            _log_omdb_metrics(
                                prefix=f"[DLNA][DEBUG] {container_key}: analyze:delta:",
                                metrics=_as_object_mapping(analyze_delta_container),
                            )

                if (
                    total_candidates == 0
                    and _counter_int(_safe_snapshot_counters(), "dlna.scan.errors") == 0
                ):
                    logger.progress("[DLNA] No se han encontrado items de vídeo.")
                    return

                logger.progress(
                    f"[DLNA] Candidatos detectados (streaming): {total_candidates}"
                )

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
            analyze_delta = _metrics_diff(
                analyze_snapshot_start, get_omdb_metrics_snapshot()
            )
            _log_omdb_metrics(
                prefix="[DLNA][DEBUG] analyze: delta:",
                metrics=_as_object_mapping(analyze_delta),
            )

        counters = _safe_snapshot_counters()
        dlna_scan_errors = _counter_int(counters, "dlna.scan.errors") + _counter_int(
            counters, "dlna.browse.errors"
        )
        dlna_circuit_blocks = _counter_int(
            counters, "dlna.browse.blocked_by_circuit"
        ) + _counter_int(counters, "dlna.xml_fetch.blocked_by_circuit")

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
                _log_omdb_rankings(container_omdb_delta_analyze, min_groups=2)

        logger.info("[DLNA] Análisis completado.", always=True)

    finally:
        try:
            flush_external_caches()
        except Exception as exc:  # pragma: no cover
            if DEBUG_MODE:
                logger.debug_ctx("DLNA", f"flush_external_caches failed: {exc!r}")


__all__ = ["analyze_dlna_server"]
