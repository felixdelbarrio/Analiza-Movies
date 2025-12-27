from __future__ import annotations

"""
backend/analiza_wiki.py

Orquestador de enriquecimiento Wikipedia/Wikidata (cache warmup / batch).

Responsabilidades (post-split)
-----------------------------
✅ Se queda con:
- Orquestación de un “run” de Wiki: iteración -> get_wiki_for_input/get_wiki -> métricas.
- Concurrencia: ThreadPoolExecutor + bounded inflight (backpressure).
- Dedupe GLOBAL por run (por imdb o title/year) para evitar trabajo duplicado.
- Logging de ejecución (inicio/progreso/fin) sin “spam”.
- flush_wiki_cache() y log_wiki_metrics_summary() en finally.

🚫 Se delega en backend.wiki_client:
- HTTP + retry/circuit breaker + throttle WDQS.
- Cache persistente + SWR + single-flight interno.
- Heurísticas de búsqueda/ranking + parsers.
- Métricas internas del cliente (get_wiki_metrics_snapshot).

Política de logs (alineada con backend/logger.py)
------------------------------------------------
- Estado global (inicio / progreso / fin) sin “spam” -> logger.progress(...)
- Debug contextual -> logger.debug_ctx("WIKI", "...") (respeta DEBUG/SILENT)
- Errores de pipeline siempre visibles -> logger.error(..., always=True)
- Este módulo NO presenta menús/prompt (si se añadieran, always=True).

Notas de diseño
---------------
- Este orquestador es “best-effort”: no rompe el pipeline si Wikipedia/Wikidata fallan.
- En NO-SILENT prioriza determinismo (orden estable de items procesados).
- En SILENT usa streaming concurrente para rendimiento.
"""

import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from backend import logger as logger
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.movie_input import MovieInput, normalize_title_for_lookup

from backend.wiki_client import (
    flush_wiki_cache,
    get_wiki,
    get_wiki_for_input,
    get_wiki_metrics_snapshot,
    log_wiki_metrics_summary,
    reset_wiki_metrics,
)

# --------------------------------------------------------------------------------------
# Best-effort config.py knobs (preferimos backend.config agregador, como en DLNA/Plex)
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

# Cap defensivo global (no queremos reventar CPU/FDs si config viene mal)
_MAX_WORKERS_CAP: int = 64


# ======================================================================================
# Dedupe global por run
# ======================================================================================

@dataclass(slots=True)
class _WikiRunDedupe:
    """
    Dedupe global por run.

    Evita refrescar/buscar lo mismo N veces si el input trae duplicados:
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
    factor = int(WIKI_MAX_INFLIGHT_FACTOR) if int(WIKI_MAX_INFLIGHT_FACTOR) > 0 else 4
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

    Uso típico:
      - warmup de cache antes de análisis pesado
      - runs batch sobre una lista de items (Plex/DLNA/CSV import)

    Devuelve un dict con estadísticas del run (para tests o reporting opcional).
    """
    t0 = time.monotonic()
    reset_wiki_metrics()

    max_workers = _compute_max_workers(int(WIKI_ANALYZE_WORKERS))
    max_inflight = _compute_max_inflight(max_workers)
    progress_every = max(50, int(WIKI_PROGRESS_EVERY_N_ITEMS))

    dedupe = _WikiRunDedupe(seen_keys=set())

    total_seen = 0
    total_submitted = 0
    total_processed = 0
    total_errors = 0
    total_skipped_dedupe = 0

    # Para no depender de que el caller pase list, iteramos “una sola vez” (streaming).
    def _iter_inputs(it: Iterable[MovieInput]) -> Iterator[MovieInput]:
        for x in it:
            yield x

    def _fetch_one(mi: MovieInput) -> None:
        """
        Best-effort: llama al cliente y no lanza exceptions hacia arriba.
        """
        try:
            title = str(getattr(mi, "title", "") or "")
            year = getattr(mi, "year", None)
            imdb_id = getattr(mi, "imdb_id_hint", None)

            # Preferimos el path “aware” del input (idioma por librería/contexto).
            _ = get_wiki_for_input(movie_input=mi, title=title, year=year, imdb_id=imdb_id)
        except Exception as exc:  # pragma: no cover
            # Errores siempre visibles, pero sin reventar el run.
            nonlocal total_errors
            total_errors += 1
            logger.error(f"[{label}] Error en wiki enrichment: {exc!r}", always=True)

    if SILENT_MODE:
        logger.progress(f"[{label}] Inicio (SILENT) | workers={max_workers} inflight_cap={max_inflight}")
    else:
        logger.progress(f"[{label}] Inicio | workers={max_workers} inflight_cap={max_inflight}")

    if DEBUG_MODE:
        logger.debug_ctx("WIKI", f"analyze_wiki_for_movie_inputs: workers={max_workers} inflight={max_inflight}")

    try:
        # ----------------------------------------------------------------------------
        # Modo NO-SILENT: determinista (orden estable)
        # ----------------------------------------------------------------------------
        if not SILENT_MODE:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                inflight: set[Future[None]] = set()

                for idx, mi in enumerate(_iter_inputs(movie_inputs), start=1):
                    total_seen += 1

                    title = str(getattr(mi, "title", "") or "")
                    year = getattr(mi, "year", None)
                    imdb_id = getattr(mi, "imdb_id_hint", None)

                    key = _request_key(title, year, imdb_id)
                    if not key or key in dedupe.seen_keys:
                        dedupe.skipped += 1
                        total_skipped_dedupe += 1
                        continue
                    dedupe.seen_keys.add(key)

                    # Progreso “humano” (visible)
                    # Nota: no hacemos “spam” de URLs ni payloads.
                    if DEBUG_MODE:
                        logger.info(f"[{label}] ({idx}) {title} ({year if year is not None else '?'})")
                    else:
                        # En NO-DEBUG, línea más compacta.
                        logger.info(f"[{label}] ({idx}) {title}")

                    inflight.add(pool.submit(_fetch_one, mi))
                    total_submitted += 1

                    if len(inflight) >= max_inflight:
                        done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                        for f in done:
                            inflight.discard(f)
                            try:
                                f.result()
                            except Exception:
                                # _fetch_one ya hace best-effort, pero por seguridad:
                                total_errors += 1
                            total_processed += 1

                # drain
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for f in done:
                        inflight.discard(f)
                        try:
                            f.result()
                        except Exception:
                            total_errors += 1
                        total_processed += 1

        # ----------------------------------------------------------------------------
        # Modo SILENT: streaming concurrente + progreso “cada N”
        # ----------------------------------------------------------------------------
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                inflight2: set[Future[None]] = set()

                for mi in _iter_inputs(movie_inputs):
                    total_seen += 1

                    title = str(getattr(mi, "title", "") or "")
                    year = getattr(mi, "year", None)
                    imdb_id = getattr(mi, "imdb_id_hint", None)

                    key = _request_key(title, year, imdb_id)
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

                # drain
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

        # Métricas: el cliente ya decide si loguearlas (según config_wiki + silent/debug).
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
        # Flush cache + métricas siempre al final (best-effort)
        try:
            flush_wiki_cache()
        except Exception as exc:  # pragma: no cover
            if DEBUG_MODE:
                logger.debug_ctx("WIKI", f"flush_wiki_cache failed: {exc!r}")


def analyze_wiki_single(
    *,
    title: str,
    year: int | None,
    imdb_id: str | None,
) -> None:
    """
    Helper simple para CLI/tests: enriquece un único título.

    Nota: usa get_wiki (sin MovieInput), pensado para debugging.
    """
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


__all__ = [
    "analyze_wiki_for_movie_inputs",
    "analyze_wiki_single",
]