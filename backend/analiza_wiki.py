from __future__ import annotations

"""
backend/analiza_wiki.py

Orquestador de enriquecimiento Wikipedia/Wikidata (cache warmup / batch).
"""

import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, SupportsIndex, SupportsInt

from backend import logger as logger
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.movie_input import MovieInput, coalesce_movie_identity
from backend.title_utils import normalize_title_for_lookup
from backend.wiki_client import (
    flush_wiki_cache,
    get_wiki,
    get_wiki_for_input,
    get_wiki_metrics_snapshot,
    log_wiki_metrics_summary,
    reset_wiki_metrics,
)

# --------------------------------------------------------------------------------------
# Best-effort config.py knobs
# --------------------------------------------------------------------------------------
try:
    from backend.config import WIKI_ANALYZE_WORKERS  # type: ignore
except Exception:  # pragma: no cover
    WIKI_ANALYZE_WORKERS = 8  # type: ignore

try:
    from backend.config import WIKI_MAX_INFLIGHT_FACTOR  # type: ignore
except Exception:  # pragma: no cover
    WIKI_MAX_INFLIGHT_FACTOR = 4  # type: ignore

try:
    from backend.config import WIKI_PROGRESS_EVERY_N_ITEMS  # type: ignore
except Exception:  # pragma: no cover
    WIKI_PROGRESS_EVERY_N_ITEMS = 200  # type: ignore

_MAX_WORKERS_CAP: int = 64


# ======================================================================================
# Helpers “typing-safe”
# ======================================================================================


def _safe_int(v: object, default: int) -> int:
    """
    Convierte v a int de forma segura (sin int(object) genérico para Pyright).
    Acepta: None/bool/int/float/str-numérico/SupportsInt/SupportsIndex.
    """
    try:
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
            if not s:
                return default
            try:
                return int(s)
            except ValueError:
                # "8.0"
                try:
                    return int(float(s))
                except Exception:
                    return default

        # last resort tipado
        if isinstance(v, SupportsInt):
            return int(v)
        if isinstance(v, SupportsIndex):
            return int(v)

        return default
    except Exception:
        return default


def _clip_hint(s: str, max_len: int = 512) -> str:
    s2 = (s or "").strip()
    if not s2:
        return ""
    return s2[:max_len]


# ======================================================================================
# Dedupe global por run
# ======================================================================================


@dataclass(slots=True)
class _WikiRunDedupe:
    """
    Dedupe global por run:
    - imdb:<imdb_id_lower>
    - ty:<norm_title>|<year_str>
    """

    seen_keys: set[str]
    skipped: int = 0


def _norm_imdb(imdb_id: str | None) -> str | None:
    if not isinstance(imdb_id, str):
        return None
    v = imdb_id.strip().lower()
    return v or None


def _ty_key(title_norm: str, year: int | None) -> str:
    return f"{title_norm}|{str(year) if year is not None else ''}"


def _request_key(title: str, year: int | None, imdb_id: str | None) -> str | None:
    imdb = _norm_imdb(imdb_id)
    if imdb:
        return f"imdb:{imdb}"
    title_norm = normalize_title_for_lookup(title or "")
    if not title_norm:
        return None
    return f"ty:{_ty_key(title_norm, year)}"


# ======================================================================================
# Concurrency knobs
# ======================================================================================


def _compute_max_workers(requested: int) -> int:
    w = int(requested) if int(requested) > 0 else 1
    return max(1, min(_MAX_WORKERS_CAP, w))


def _compute_max_inflight(max_workers: int) -> int:
    factor = _safe_int(WIKI_MAX_INFLIGHT_FACTOR, 4)
    if factor <= 0:
        factor = 4
    inflight = max_workers * factor
    return max(max_workers, inflight)


# ======================================================================================
# Public API
# ======================================================================================


def analyze_wiki_for_movie_inputs(
    movie_inputs: Iterable[MovieInput],
    *,
    label: str = "WIKI",
) -> dict[str, Any]:
    """
    Orquesta un run de enriquecimiento Wiki para un iterable de MovieInput.
    """
    t0 = time.monotonic()
    reset_wiki_metrics()

    max_workers = _compute_max_workers(_safe_int(WIKI_ANALYZE_WORKERS, 8))
    max_inflight = _compute_max_inflight(max_workers)
    progress_every = max(50, _safe_int(WIKI_PROGRESS_EVERY_N_ITEMS, 200))

    dedupe = _WikiRunDedupe(seen_keys=set())

    total_seen = 0
    total_submitted = 0
    total_processed = 0
    total_errors = 0
    total_skipped_dedupe = 0

    def _iter_inputs(it: Iterable[MovieInput]) -> Iterator[MovieInput]:
        for x in it:
            yield x

    def _extract_path_hints(mi: MovieInput) -> tuple[str, str]:
        file_path = str(getattr(mi, "file_path", "") or "")
        extra = getattr(mi, "extra", {}) or {}
        source_url = ""
        if isinstance(extra, dict):
            v = extra.get("source_url")
            if isinstance(v, str):
                source_url = v
        return file_path, source_url

    def _coalesce_identity(mi: MovieInput) -> tuple[str, int | None, str | None]:
        title_raw = str(getattr(mi, "title", "") or "")
        year_raw = getattr(mi, "year", None)
        imdb_raw = getattr(mi, "imdb_id_hint", None)

        file_path, source_url = _extract_path_hints(mi)
        hint = _clip_hint(f"{file_path} {source_url}".strip(), max_len=512)

        title2, year2, imdb2 = coalesce_movie_identity(
            title=title_raw,
            year=year_raw,
            file_path=hint,
            imdb_id_hint=imdb_raw,
        )
        return title2, year2, imdb2

    def _fetch_one(mi: MovieInput) -> None:
        nonlocal total_errors
        try:
            title2, year2, imdb2 = _coalesce_identity(mi)
            title_norm = normalize_title_for_lookup(title2)

            res = None
            if title_norm:
                res = get_wiki_for_input(movie_input=mi, title=title_norm, year=year2, imdb_id=imdb2)

            # fallback muy conservador
            if res is None and title2 and (not title_norm or len(title_norm) < 3):
                _ = get_wiki_for_input(movie_input=mi, title=title2, year=year2, imdb_id=imdb2)

        except Exception as exc:  # pragma: no cover
            total_errors += 1
            logger.error(f"[{label}] Error en wiki enrichment: {exc!r}", always=True)

    if SILENT_MODE:
        logger.progress(f"[{label}] Inicio (SILENT) | workers={max_workers} inflight_cap={max_inflight}")
    else:
        logger.progress(f"[{label}] Inicio | workers={max_workers} inflight_cap={max_inflight}")

    if DEBUG_MODE:
        logger.debug_ctx("WIKI", f"analyze_wiki_for_movie_inputs: workers={max_workers} inflight={max_inflight}")

    try:
        # NO-SILENT: determinista
        if not SILENT_MODE:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                inflight: set[Future[None]] = set()

                for idx, mi in enumerate(_iter_inputs(movie_inputs), start=1):
                    total_seen += 1

                    title2, year2, imdb2 = _coalesce_identity(mi)
                    key = _request_key(title2, year2, imdb2)
                    if not key or key in dedupe.seen_keys:
                        dedupe.skipped += 1
                        total_skipped_dedupe += 1
                        continue
                    dedupe.seen_keys.add(key)

                    if DEBUG_MODE:
                        logger.info(f"[{label}] ({idx}) {title2} ({year2 if year2 is not None else '?'})")
                    else:
                        logger.info(f"[{label}] ({idx}) {title2}")

                    inflight.add(pool.submit(_fetch_one, mi))
                    total_submitted += 1

                    if len(inflight) >= max_inflight:
                        done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                        for f in done:
                            inflight.discard(f)
                            try:
                                f.result()
                            except Exception:
                                total_errors += 1
                            total_processed += 1

                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for f in done:
                        inflight.discard(f)
                        try:
                            f.result()
                        except Exception:
                            total_errors += 1
                        total_processed += 1

        # SILENT: streaming
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                inflight2: set[Future[None]] = set()

                for mi in _iter_inputs(movie_inputs):
                    total_seen += 1

                    title2, year2, imdb2 = _coalesce_identity(mi)
                    key = _request_key(title2, year2, imdb2)
                    if not key or key in dedupe.seen_keys:
                        dedupe.skipped += 1
                        total_skipped_dedupe += 1
                        continue
                    dedupe.seen_keys.add(key)

                    inflight2.add(pool.submit(_fetch_one, mi))
                    total_submitted += 1

                    if DEBUG_MODE and (total_submitted % progress_every == 0):
                        logger.progress(f"[{label}][DEBUG] Progreso: submitted={total_submitted} processed={total_processed}...")

                    if len(inflight2) >= max_inflight:
                        done, _ = wait(inflight2, return_when=FIRST_COMPLETED)
                        for f in done:
                            inflight2.discard(f)
                            try:
                                f.result()
                            except Exception:
                                total_errors += 1
                            total_processed += 1

                while inflight2:
                    done, _ = wait(inflight2, return_when=FIRST_COMPLETED)
                    for f in done:
                        inflight2.discard(f)
                        try:
                            f.result()
                        except Exception:
                            total_errors += 1
                        total_processed += 1

        elapsed = time.monotonic() - t0
        snap = get_wiki_metrics_snapshot()

        logger.progress(
            f"[{label}] Fin | time={elapsed:.1f}s "
            f"seen={total_seen} submitted={total_submitted} processed={total_processed} "
            f"dedupe_skipped={total_skipped_dedupe} errors={total_errors}"
        )

        try:
            log_wiki_metrics_summary()
        except Exception:
            pass

        return {
            "seen": total_seen,
            "submitted": total_submitted,
            "processed": total_processed,
            "dedupe_skipped": total_skipped_dedupe,
            "errors": total_errors,
            "elapsed_s": elapsed,
            "wiki_metrics": snap,
        }

    finally:
        try:
            flush_wiki_cache()
        except Exception as exc:  # pragma: no cover
            if DEBUG_MODE:
                logger.debug_ctx("WIKI", f"flush_wiki_cache failed: {exc!r}")


def analyze_wiki_single(*, title: str, year: int | None, imdb_id: str | None) -> None:
    reset_wiki_metrics()
    logger.progress("[WIKI] Single lookup: start")
    try:
        _ = get_wiki(title=title, year=year, imdb_id=imdb_id)
    except Exception as exc:  # pragma: no cover
        logger.error(f"[WIKI] Single lookup error: {exc!r}", always=True)
    finally:
        try:
            flush_wiki_cache()
        except Exception:
            pass
        try:
            log_wiki_metrics_summary(force=False)
        except Exception:
            pass
        logger.progress("[WIKI] Single lookup: end")


__all__ = ["analyze_wiki_for_movie_inputs", "analyze_wiki_single"]