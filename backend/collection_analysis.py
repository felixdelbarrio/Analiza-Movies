from __future__ import annotations

"""
backend/collection_analysis.py

Orquestador “por item” que conecta el pipeline completo para UNA película.

✅ Esta iteración añade / asegura:
- Bounded in-memory caches (LRU) para resultados OMDb / wiki minimal (intra-run).
- Negative caching intra-run (sentinel) para evitar reintentos dentro del mismo run.
- TODO configurable desde backend/config.py (y por env):
    - COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS
    - COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS
    - COLLECTION_TRACE_LINE_MAX_CHARS
    - COLLECTION_ENABLE_LAZY_WIKI
    - COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE

🎯 Responsabilidad (y lo que NO hace)
------------------------------------
Este módulo COORDINA (no decide reglas de negocio) el pipeline completo:

- Entrada unificada (MovieInput) desde Plex / DLNA / etc.
- Resolución de datos externos (OMDb / Wiki) de forma LAZY (bajo demanda).
- Ejecución del core genérico (analyze_input_movie) con trazas opcionales.
- Sugerencias de metadata (solo Plex) y enriquecimiento final para reporting.

NO hace:
- I/O pesado de scanning (lo hacen orquestadores: collection_analiza_plex/dlna).
- Decisiones “hard” (eso está en scoring/decision_logic).
- Flush por item (sería carísimo en throughput).

🧠 Optimizaciones principales (IO + APIs externas)
--------------------------------------------------
1) Lazy OMDb:
   - No hay prefetch.
   - OMDb se resuelve solo si el core lo solicita (callback fetch_omdb),
     o si se necesita explícitamente para persistir wiki minimal.

2) Lazy Wiki:
   - Wiki solo se intenta si:
        a) OMDb NO trae __wiki minimal ya persistido, y
        b) el item se beneficia de wiki (heurística conservadora), y
        c) COLLECTION_ENABLE_LAZY_WIKI=True
   - Esto evita llamar a Wikipedia/Wikidata innecesariamente.

3) Blindaje del consumo de Wiki:
   - Consideramos “útil” un wiki_item solo si:
        - status == "ok", o
        - compat legacy: contiene señales mínimas (qid / wikibase_item).

4) Negative caching “real” (INTRA-RUN) para OMDb/Wiki:
   - OMDb: si omdb_query_with_cache devuelve None, cacheamos MISS y devolvemos {} al core.
   - Wiki: si intentamos wiki y NO es usable (None/negativo/fallo), cacheamos MISS.

5) Bounded in-memory caches (LRU):
   - Evita crecimiento infinito de memoria en runs grandes.
   - Política:
       - LRU global por módulo (no por thread).
       - Las entradas se refrescan al acceder.
       - Al sobrepasar MAX, se expulsan las menos usadas.

6) Flush explícito:
   - Se expone flush_external_caches() para que el orquestador lo llame UNA vez al final del run.

🪵 Filosofía de logs (alineada con backend/logger.py)
----------------------------------------------------
- Este módulo NO imprime progreso por item (lo hacen orquestadores).
- Devuelve `logs: list[str]` acotado por item, para que el orquestador lo muestre
  según modo (SILENT/DEBUG).
- No implementa “política propia”: usa utilidades centralizadas en logger.py.
"""

import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Final, TypeAlias

from backend import logger
from backend.analyze_input_core import AnalysisRow, analyze_input_movie
from backend.config import (
    COLLECTION_ENABLE_LAZY_WIKI,
    COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS,
    COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE,
    COLLECTION_TRACE_LINE_MAX_CHARS,
    COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS,
)
from backend.metadata_fix import generate_metadata_suggestions_row
from backend.movie_input import MovieInput, normalize_title_for_lookup
from backend.omdb_client import omdb_query_with_cache
from backend.wiki_client import get_wiki, get_wiki_for_input

# ---------------------------------------------------------------------------
# Opcionales (compat / despliegues parciales)
# ---------------------------------------------------------------------------

try:
    from backend.omdb_client import patch_cached_omdb_record  # type: ignore
except Exception:  # pragma: no cover
    patch_cached_omdb_record = None  # type: ignore[assignment]

try:
    from backend.omdb_client import flush_omdb_cache  # type: ignore
except Exception:  # pragma: no cover
    flush_omdb_cache = None  # type: ignore[assignment]

try:
    from backend.wiki_client import flush_wiki_cache  # type: ignore
except Exception:  # pragma: no cover
    flush_wiki_cache = None  # type: ignore[assignment]


# ============================================================
# Cache local (módulo) + locks (LRU + INTRA-RUN negative caching)
# ============================================================

_OmdbCacheKey: TypeAlias = tuple[str, str, str | None]
# key: (norm_title, norm_year_str, imdb_hint_norm)

# Sentinel explícito para “ya lo intenté y no hay nada usable”.
_CACHE_MISS: Final[object] = object()

# Tamaños máximos configurables desde config.py (y env).
_OMDB_LOCAL_CACHE_MAX_ITEMS: Final[int] = int(COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS)
_WIKI_LOCAL_CACHE_MAX_ITEMS: Final[int] = int(COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS)

# Guardamos dict o _CACHE_MISS.
# Usamos OrderedDict para política LRU:
# - get() refresca orden (move_to_end)
# - set() inserta y recorta por tamaño.
_OMDB_LOCAL_CACHE: "OrderedDict[_OmdbCacheKey, object]" = OrderedDict()
_WIKI_LOCAL_CACHE: "OrderedDict[_OmdbCacheKey, object]" = OrderedDict()

_LOCAL_CACHE_LOCK = threading.Lock()

# Solo para patch/write-back a omdb_cache.json (evita race multi-thread)
_OMDB_CACHE_WRITE_LOCK = threading.Lock()


def _lru_get(cache: "OrderedDict[_OmdbCacheKey, object]", key: _OmdbCacheKey) -> object | None:
    """
    LRU get (NO thread-safe por sí solo; usar _LOCAL_CACHE_LOCK).

    Returns:
      - dict (hit)
      - _CACHE_MISS (negative cached)
      - None (not present)
    """
    v = cache.get(key)
    if v is None:
        return None
    try:
        cache.move_to_end(key, last=True)
    except Exception:
        pass
    return v


def _lru_set(
    cache: "OrderedDict[_OmdbCacheKey, object]",
    key: _OmdbCacheKey,
    value: object,
    *,
    max_items: int,
) -> None:
    """
    LRU set (NO thread-safe por sí solo; usar _LOCAL_CACHE_LOCK).

    - Inserta/actualiza
    - Refresca orden
    - Recorta por max_items expulsando LRU (least-recently-used)

    Nota:
    - Si max_items <= 0 => cache desactivado (se limpia).
    """
    cache[key] = value
    try:
        cache.move_to_end(key, last=True)
    except Exception:
        pass

    if max_items <= 0:
        cache.clear()
        return

    while len(cache) > max_items:
        try:
            cache.popitem(last=False)
        except Exception:
            break


# ============================================================
# Logging helpers (centralizados)
# ============================================================

_TRACE_LINE_MAX_CHARS: Final[int] = int(COLLECTION_TRACE_LINE_MAX_CHARS)


def _append_log(logs: list[str], line: object, *, force: bool = False, tag: str | None = None) -> None:
    """
    Añade una línea al buffer `logs` usando la política central.

    Importante:
    - Logging nunca debe romper el pipeline.
    - force=True: útil para errores críticos incluso en SILENT.
    """
    try:
        logger.append_bounded_log(logs, line, force=force, tag=tag)
    except Exception:
        return


def _append_trace(logs: list[str], line: object) -> None:
    """
    Traza corta por item (solo en DEBUG_MODE, inyectada por el core).
    """
    try:
        text = logger.truncate_line(str(line), max_chars=_TRACE_LINE_MAX_CHARS)
    except Exception:
        text = "<unprintable>"
    _append_log(logs, f"[TRACE] {text}")


def _dbg_ctx(msg: object) -> None:
    """Debug contextual del orquestador (COLLECTION)."""
    try:
        logger.debug_ctx("COLLECTION", msg)
    except Exception:
        return


# ============================================================
# Utilidades de parse / normalización
# ============================================================

def _safe_float(value: object) -> float | None:
    """Convierte a float de forma defensiva (int/float/str), o None."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s or s.upper() == "N/A":
                return None
            return float(s)
        return None
    except Exception:
        return None


def _norm_title(title: str) -> str:
    """
    Normaliza un título para keys/caches.
    - Usa normalize_title_for_lookup (canonical del proyecto).
    """
    try:
        out = normalize_title_for_lookup(title or "")
        return out or (title or "").strip().lower()
    except Exception:
        return (title or "").strip().lower()


def _norm_year_str(year: int | None) -> str:
    """Normaliza year a string estable para la key del cache."""
    return str(year) if year is not None else ""


def _norm_imdb_hint(raw: object) -> str | None:
    """
    Normaliza imdb_id_hint (puede venir None/str).

    Importante:
    - Lo bajamos a lowercase para consistencia con caches externos (omdb/wiki).
    """
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    return v or None


def _cache_key(title_for_fetch: str, year_for_fetch: int | None, imdb_hint: str | None) -> _OmdbCacheKey:
    """Key compacta y estable para caches locales (in-memory)."""
    return (_norm_title(title_for_fetch), _norm_year_str(year_for_fetch), imdb_hint)


def _extract_wiki_meta(omdb_record: Mapping[str, object] | None) -> dict[str, object]:
    """
    OMDb cache puede contener __wiki minimal (persistido por este orquestador).

    IMPORTANTE:
    - Es un “minimal block” (IDs + provenance), no el payload completo.
    """
    if not omdb_record:
        return {}
    wiki_raw = omdb_record.get("__wiki")
    if isinstance(wiki_raw, Mapping):
        return dict(wiki_raw)
    return {}


def _build_lookup_key(title_for_fetch: str, year_for_fetch: int | None, imdb_hint: str | None) -> str:
    """
    Clave humana para provenance/diagnóstico.

    Preferimos imdb_id cuando existe porque es la clave más estable.
    """
    t = (title_for_fetch or "").strip()
    if imdb_hint:
        return f"imdb_id:{imdb_hint}"
    if year_for_fetch is not None:
        return f"title_year:{t}|{year_for_fetch}"
    return f"title:{t}"


def _build_wiki_lookup_info(
    *,
    title_for_fetch: str,
    year_for_fetch: int | None,
    imdb_used: str | None,
) -> dict[str, object]:
    """Describe con qué parámetros se consultó a Wiki (persistencia minimal)."""
    title_clean = (title_for_fetch or "").strip()
    if imdb_used:
        return {"via": "imdb_id", "imdb_id": imdb_used}
    if year_for_fetch is not None:
        return {"via": "title_year", "title": title_clean, "year": year_for_fetch}
    return {"via": "title", "title": title_clean}


def _build_minimal_wiki_block(
    *,
    imdb_id: str | None,
    wikidata_id: str | None,
    wikipedia_title: str | None,
    wiki_lookup: Mapping[str, object],
    source_language: str | None = None,
) -> dict[str, object]:
    """
    Bloque mínimo que persistimos dentro de omdb_cache.json.

    Regla de oro:
    - Solo IDs + provenance
    - Nunca payloads grandes (extracts/summaries/images)
    """
    out: dict[str, object] = {"wiki_lookup": dict(wiki_lookup)}
    if imdb_id:
        out["imdb_id"] = imdb_id
    if wikidata_id:
        out["wikidata_id"] = wikidata_id
    if wikipedia_title:
        out["wikipedia_title"] = wikipedia_title
    if source_language:
        out["source_language"] = source_language
    return out


def _persist_minimal_wiki_into_omdb_cache(
    *,
    title_for_fetch: str,
    year_for_fetch: int | None,
    imdb_id_for_cache: str | None,
    imdb_hint: str | None,
    minimal_wiki: Mapping[str, object],
    lookup_key: str,
) -> None:
    """
    Persiste __wiki MINIMAL dentro de omdb_cache.json.

    - Respeta flag COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE.
    - No hace nada si patch_cached_omdb_record no existe.
    - Serializa writes con _OMDB_CACHE_WRITE_LOCK para evitar races.
    """
    if not COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE:
        return
    if patch_cached_omdb_record is None:
        return

    imdb_id_final = imdb_id_for_cache or imdb_hint
    norm_title = _norm_title(title_for_fetch)
    norm_year = _norm_year_str(year_for_fetch)

    wikidata_id: str | None = None
    raw_wdid = minimal_wiki.get("wikidata_id") if isinstance(minimal_wiki, Mapping) else None
    if isinstance(raw_wdid, str) and raw_wdid.strip():
        wikidata_id = raw_wdid.strip()

    prov_patch: dict[str, object] = {"lookup_key": lookup_key, "had_imdb_hint": bool(imdb_hint)}
    if wikidata_id:
        prov_patch["wiki_wikidata_id"] = wikidata_id

    try:
        with _OMDB_CACHE_WRITE_LOCK:
            patch_cached_omdb_record(
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_id_final,
                patch={"__wiki": dict(minimal_wiki), "__prov": prov_patch},
            )
        _dbg_ctx(
            "Persisted minimal wiki into OMDb cache | "
            f"lookup_key={lookup_key} imdb_id={imdb_id_final or 'n/a'} wikidata_id={wikidata_id or 'n/a'}"
        )
    except Exception as exc:  # pragma: no cover
        _dbg_ctx(f"patch_cached_omdb_record failed | exc={exc!r}")


def _wiki_item_is_usable(wiki_item: Mapping[str, object]) -> bool:
    """
    Blindaje: aceptamos el item de wiki como “usable” solo si parece OK.

    Compatibilidad:
    - Implementaciones antiguas podían no tener `status`.
      En ese caso, exigimos señales mínimas:
        - wikidata.qid o wiki.wikibase_item
    """
    status = wiki_item.get("status")
    if isinstance(status, str):
        return status == "ok"

    wiki_block = wiki_item.get("wiki")
    wikidata_block = wiki_item.get("wikidata")
    if not isinstance(wiki_block, Mapping) or not isinstance(wikidata_block, Mapping):
        return False

    qid = wikidata_block.get("qid")
    if isinstance(qid, str) and qid.strip():
        return True

    wb = wiki_block.get("wikibase_item")
    if isinstance(wb, str) and wb.strip():
        return True

    return False


def _should_fetch_wiki_for_reporting(base_row: Mapping[str, object]) -> bool:
    """
    Heurística conservadora para evitar llamadas a Wiki.

    Activamos Wiki si:
    - misidentified_hint existe (caso “sospechoso”), o
    - decision es DELETE/MAYBE (casos que suelen necesitar verificación).
    """
    misidentified_hint = base_row.get("misidentified_hint")
    if isinstance(misidentified_hint, str) and misidentified_hint.strip():
        return True

    decision = base_row.get("decision")
    if isinstance(decision, str) and decision.strip().upper() in ("DELETE", "MAYBE"):
        return True

    return False


# ============================================================
# API: flush explícito
# ============================================================

def flush_external_caches() -> None:
    """
    Flush explícito (opcional) para invocarse UNA vez al final del run.

    Contrato:
    - Seguro llamarlo aunque los backends no expongan flush_* (compat).
    - Seguro llamarlo varias veces (idempotencia razonable).
    """
    try:
        if callable(flush_omdb_cache):
            flush_omdb_cache()
    except Exception as exc:  # pragma: no cover
        _dbg_ctx(f"flush_omdb_cache failed: {exc!r}")

    try:
        if callable(flush_wiki_cache):
            flush_wiki_cache()
    except Exception as exc:  # pragma: no cover
        _dbg_ctx(f"flush_wiki_cache failed: {exc!r}")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def analyze_movie(
    movie_input: MovieInput,
    *,
    source_movie: object | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[str]]:
    """
    Analiza un MovieInput y devuelve:
      - row: dict para report_all.csv (o None si no se pudo analizar)
      - meta_sugg: dict de sugerencias metadata (solo Plex) o None
      - logs: lista de strings acotada por item

    Importante:
    - Este módulo NO imprime progreso por item; eso lo hacen los orquestadores.
    - OMDb/Wiki se resuelven de forma LAZY.
    """
    logs: list[str] = []

    # ------------------------------------------------------------------
    # 0) Precedencia de título/año para reporting (Plex manda si está)
    # ------------------------------------------------------------------
    display_title_raw = movie_input.extra.get("display_title")
    display_year_raw = movie_input.extra.get("display_year")

    display_title = (
        display_title_raw if isinstance(display_title_raw, str) and display_title_raw.strip() else movie_input.title
    )

    display_year: int | None
    if isinstance(display_year_raw, int):
        display_year = display_year_raw
    else:
        display_year = movie_input.year

    imdb_hint = _norm_imdb_hint(movie_input.imdb_id_hint)

    # ------------------------------------------------------------------
    # 1) Callback fetch_omdb: OMDb LAZY + read-through de __wiki
    #    + bounded LRU caches + negative caching intra-run
    # ------------------------------------------------------------------
    omdb_data: dict[str, object] | None = None
    wiki_meta: dict[str, object] = {}

    def fetch_omdb(title_for_fetch: str, year_for_fetch: int | None) -> Mapping[str, object]:
        """
        Callback inyectado al core.

        Implementa:
        - LRU cache local (bounded) por key estable.
        - Consulta a omdb_query_with_cache (persistente).
        - Read-through del __wiki minimal (si existe) sin tocar Wiki real.
        - Negative caching intra-run: si OMDb devuelve None -> cache MISS y devolvemos {}.

        Contrato esperado por analyze_input_movie:
        - Retorna Mapping (posiblemente vacío) y nunca lanza.
        """
        nonlocal omdb_data, wiki_meta

        key = _cache_key(title_for_fetch, year_for_fetch, imdb_hint)

        with _LOCAL_CACHE_LOCK:
            cached_omdb = _lru_get(_OMDB_LOCAL_CACHE, key)
            cached_wiki = _lru_get(_WIKI_LOCAL_CACHE, key)

        # 1) OMDb MISS cacheado intra-run
        if cached_omdb is _CACHE_MISS:
            omdb_data = {}
            wiki_meta = {}
            return {}

        # 2) OMDb HIT
        if isinstance(cached_omdb, dict):
            omdb_data = cached_omdb
            wiki_meta = dict(cached_wiki) if isinstance(cached_wiki, dict) else {}
            return cached_omdb

        # 3) Cache miss real -> vamos a persistente/red
        lookup_key = _build_lookup_key(title_for_fetch, year_for_fetch, imdb_hint)

        try:
            record = omdb_query_with_cache(
                title=title_for_fetch,
                year=year_for_fetch,
                imdb_id=imdb_hint,
                provenance={"lookup_key": lookup_key, "had_imdb_hint": bool(imdb_hint)},
            )
        except Exception as exc:  # pragma: no cover
            _dbg_ctx(f"omdb_query_with_cache failed | lookup_key={lookup_key} exc={exc!r}")
            record = None

        if record is None:
            with _LOCAL_CACHE_LOCK:
                _lru_set(_OMDB_LOCAL_CACHE, key, _CACHE_MISS, max_items=_OMDB_LOCAL_CACHE_MAX_ITEMS)
                _lru_set(_WIKI_LOCAL_CACHE, key, {}, max_items=_WIKI_LOCAL_CACHE_MAX_ITEMS)
            omdb_data = {}
            wiki_meta = {}
            _append_log(logs, f"lookup_key={lookup_key}", tag="OMDB_NONE")
            return {}

        omdb_dict = dict(record)
        wiki_dict = _extract_wiki_meta(omdb_dict)

        with _LOCAL_CACHE_LOCK:
            _lru_set(_OMDB_LOCAL_CACHE, key, omdb_dict, max_items=_OMDB_LOCAL_CACHE_MAX_ITEMS)
            _lru_set(_WIKI_LOCAL_CACHE, key, wiki_dict, max_items=_WIKI_LOCAL_CACHE_MAX_ITEMS)

        omdb_data = omdb_dict
        wiki_meta = wiki_dict

        if wiki_dict:
            _dbg_ctx(f"Using __wiki from OMDb cache | lookup_key={lookup_key}")

        return omdb_dict

    # ------------------------------------------------------------------
    # 2) Plex rating (si aplica)
    # ------------------------------------------------------------------
    plex_rating: float | None = None
    if movie_input.source == "plex" and source_movie is not None:
        plex_user_rating = getattr(source_movie, "userRating", None)
        plex_rating_raw = getattr(source_movie, "rating", None)
        plex_rating = _safe_float(plex_user_rating) or _safe_float(plex_rating_raw)

    # ------------------------------------------------------------------
    # 3) Core genérico (con trazas opcionales)
    # ------------------------------------------------------------------
    def _analysis_trace(line: str) -> None:
        _append_trace(logs, line)
        _dbg_ctx(line)

    analysis_trace_cb: Callable[[str], None] | None = _analysis_trace if logger.is_debug_mode() else None

    try:
        base_row: AnalysisRow = analyze_input_movie(
            movie_input,
            fetch_omdb,
            plex_title=display_title,
            plex_year=display_year,
            plex_rating=plex_rating,
            metacritic_score=None,
            analysis_trace=analysis_trace_cb,
        )
    except Exception as exc:  # pragma: no cover
        msg = (
            f"[ERROR] {movie_input.library} / {display_title} ({display_year}): "
            f"fallo en core de análisis: {exc!r}"
        )
        logger.error(msg, always=True)
        _append_log(logs, msg, force=True)
        return None, None, logs

    if not base_row:
        _append_log(
            logs,
            f"{movie_input.library} / {display_title} ({display_year}): core devolvió fila vacía.",
            force=not logger.is_silent_mode(),
            tag="WARN",
        )
        return None, None, logs

    # ------------------------------------------------------------------
    # 4) Lazy Wiki (solo si compensa y si OMDb no trae __wiki)
    #    + bounded LRU cache + negative caching intra-run
    # ------------------------------------------------------------------
    if (
        COLLECTION_ENABLE_LAZY_WIKI
        and not wiki_meta
        and _should_fetch_wiki_for_reporting(base_row)
    ):
        key_post = _cache_key(movie_input.title, movie_input.year, imdb_hint)

        with _LOCAL_CACHE_LOCK:
            wiki_local = _lru_get(_WIKI_LOCAL_CACHE, key_post)

        if wiki_local is _CACHE_MISS:
            _dbg_ctx("Lazy Wiki skipped: negative-cached intra-run.")
        elif isinstance(wiki_local, dict) and wiki_local:
            wiki_meta = dict(wiki_local)
        else:
            # Si el core no tocó OMDb pero necesitamos Wiki, forzamos OMDb post-core para:
            #   - extraer imdbID si existe
            #   - persistir __wiki minimal en omdb_cache.json
            if omdb_data is None:
                _dbg_ctx("Lazy Wiki requires OMDb record -> forcing OMDb fetch (post-core).")
                _ = fetch_omdb(movie_input.title, movie_input.year)

            omdb_dict_for_wiki = omdb_data or {}
            lookup_key = _build_lookup_key(movie_input.title, movie_input.year, imdb_hint)

            imdb_id_from_omdb: str | None = None
            imdb_raw = omdb_dict_for_wiki.get("imdbID")
            if isinstance(imdb_raw, str) and imdb_raw.strip():
                imdb_id_from_omdb = imdb_raw.strip().lower()

            imdb_used_for_wiki = imdb_id_from_omdb or imdb_hint

            if imdb_used_for_wiki:
                wiki_item: Mapping[str, object] | None
                try:
                    wiki_item = get_wiki_for_input(
                        movie_input=movie_input,
                        title=movie_input.title,
                        year=movie_input.year,
                        imdb_id=imdb_used_for_wiki,
                    )
                    if wiki_item is None:
                        wiki_item = get_wiki(
                            title=movie_input.title,
                            year=movie_input.year,
                            imdb_id=imdb_used_for_wiki,
                        )
                except Exception as exc:  # pragma: no cover
                    _dbg_ctx(f"Lazy Wiki call failed | lookup_key={lookup_key} exc={exc!r}")
                    wiki_item = None

                if isinstance(wiki_item, Mapping) and _wiki_item_is_usable(wiki_item):
                    wiki_block = wiki_item.get("wiki")
                    wikidata_block = wiki_item.get("wikidata")

                    imdb_from_wiki: str | None = None
                    imdb_cached = wiki_item.get("imdbID")
                    if isinstance(imdb_cached, str) and imdb_cached.strip():
                        imdb_from_wiki = imdb_cached.strip().lower()

                    wikidata_id: str | None = None
                    if isinstance(wikidata_block, Mapping):
                        qid = wikidata_block.get("qid")
                        if isinstance(qid, str) and qid.strip():
                            wikidata_id = qid.strip()

                    wikipedia_title: str | None = None
                    source_language: str | None = None
                    if isinstance(wiki_block, Mapping):
                        wt = wiki_block.get("wikipedia_title")
                        if isinstance(wt, str) and wt.strip():
                            wikipedia_title = wt.strip()
                        sl = wiki_block.get("source_language")
                        if isinstance(sl, str) and sl.strip():
                            source_language = sl.strip()

                    wiki_lookup = _build_wiki_lookup_info(
                        title_for_fetch=movie_input.title,
                        year_for_fetch=movie_input.year,
                        imdb_used=imdb_used_for_wiki,
                    )

                    minimal_wiki = _build_minimal_wiki_block(
                        imdb_id=imdb_from_wiki or imdb_used_for_wiki,
                        wikidata_id=wikidata_id,
                        wikipedia_title=wikipedia_title,
                        wiki_lookup=wiki_lookup,
                        source_language=source_language,
                    )

                    # Inyectamos en memoria y persistimos al cache OMDb.
                    try:
                        omdb_dict_for_wiki["__wiki"] = minimal_wiki
                    except Exception:
                        omdb_dict_for_wiki = dict(omdb_dict_for_wiki)
                        omdb_dict_for_wiki["__wiki"] = minimal_wiki

                    _persist_minimal_wiki_into_omdb_cache(
                        title_for_fetch=movie_input.title,
                        year_for_fetch=movie_input.year,
                        imdb_id_for_cache=imdb_id_from_omdb,
                        imdb_hint=imdb_hint,
                        minimal_wiki=minimal_wiki,
                        lookup_key=lookup_key,
                    )

                    with _LOCAL_CACHE_LOCK:
                        _lru_set(
                            _OMDB_LOCAL_CACHE,
                            key_post,
                            dict(omdb_dict_for_wiki),
                            max_items=_OMDB_LOCAL_CACHE_MAX_ITEMS,
                        )
                        _lru_set(
                            _WIKI_LOCAL_CACHE,
                            key_post,
                            dict(minimal_wiki),
                            max_items=_WIKI_LOCAL_CACHE_MAX_ITEMS,
                        )

                    omdb_data = dict(omdb_dict_for_wiki)
                    wiki_meta = dict(minimal_wiki)
                    _dbg_ctx(f"Lazy Wiki success | lookup_key={lookup_key}")
                else:
                    with _LOCAL_CACHE_LOCK:
                        _lru_set(_WIKI_LOCAL_CACHE, key_post, _CACHE_MISS, max_items=_WIKI_LOCAL_CACHE_MAX_ITEMS)
                    _dbg_ctx(f"Lazy Wiki negative/None -> intra-run MISS cached | lookup_key={lookup_key}")
            else:
                with _LOCAL_CACHE_LOCK:
                    _lru_set(_WIKI_LOCAL_CACHE, key_post, _CACHE_MISS, max_items=_WIKI_LOCAL_CACHE_MAX_ITEMS)
                _dbg_ctx("Lazy Wiki skipped: no imdb id available (omdb/imdb_hint).")

    # ------------------------------------------------------------------
    # 5) Sugerencias de metadata (solo Plex)
    # ------------------------------------------------------------------
    omdb_dict: dict[str, object] = dict(omdb_data) if isinstance(omdb_data, dict) else {}
    meta_sugg: dict[str, object] | None = None

    if movie_input.source == "plex" and source_movie is not None:
        try:
            meta_candidate = generate_metadata_suggestions_row(movie_input, omdb_dict or None)
            if isinstance(meta_candidate, dict):
                meta_sugg = meta_candidate
                _append_log(logs, f"{movie_input.library} / {display_title} ({display_year})", tag="METADATA_SUGG")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"generate_metadata_suggestions_row falló para {display_title!r}: {exc!r}", always=True)

    misidentified_hint = base_row.get("misidentified_hint")
    if isinstance(misidentified_hint, str) and misidentified_hint.strip():
        _append_log(
            logs,
            f"{movie_input.library} / {display_title} ({display_year}): {misidentified_hint}",
            tag="MISIDENTIFIED",
        )

    # ------------------------------------------------------------------
    # 6) Enriquecimiento estándar para reporting
    # ------------------------------------------------------------------
    poster_url: str | None = None
    trailer_url: str | None = None
    imdb_id: str | None = None
    omdb_json_str: str | None = None

    if omdb_dict:
        poster_raw = omdb_dict.get("Poster")
        trailer_raw = omdb_dict.get("Website")
        imdb_id_raw = omdb_dict.get("imdbID")

        poster_url = poster_raw if isinstance(poster_raw, str) else None
        trailer_url = trailer_raw if isinstance(trailer_raw, str) else None
        imdb_id = imdb_id_raw.strip().lower() if isinstance(imdb_id_raw, str) and imdb_id_raw.strip() else None

    if imdb_id is None and imdb_hint is not None:
        imdb_id = imdb_hint

    # omdb_json puede ser grande:
    # - En modo normal: útil para inspección
    # - En SILENT: lo omitimos salvo DEBUG_MODE=True (política central en logger)
    if omdb_dict and (not logger.is_silent_mode() or logger.is_debug_mode()):
        try:
            omdb_json_str = json.dumps(omdb_dict, ensure_ascii=False)
        except Exception:
            omdb_json_str = str(omdb_dict)

    # ------------------------------------------------------------------
    # 7) Construcción fila final (Plex prevalece en campos nativos)
    # ------------------------------------------------------------------
    row: dict[str, object] = dict(base_row)

    row["source"] = movie_input.source
    row["library"] = movie_input.library
    row["title"] = display_title
    row["year"] = display_year

    file_size_bytes = row.get("file_size_bytes")
    row["file_size"] = file_size_bytes if isinstance(file_size_bytes, int) else movie_input.file_size_bytes

    row["file"] = movie_input.file_path or row.get("file", "")

    source_url_obj = movie_input.extra.get("source_url")
    row["source_url"] = source_url_obj if isinstance(source_url_obj, str) else ""

    row["rating_key"] = movie_input.rating_key
    row["guid"] = movie_input.plex_guid
    row["thumb"] = movie_input.thumb_url

    row["imdb_id"] = imdb_id
    row["poster_url"] = poster_url
    row["trailer_url"] = trailer_url
    row["omdb_json"] = omdb_json_str

    # wiki_meta es minimal: wikidata_id y wikipedia_title salen directos.
    wikidata_id = wiki_meta.get("wikidata_id")
    if not isinstance(wikidata_id, str) or not wikidata_id.strip():
        wikidata_id = wiki_meta.get("wikadata_id")  # compat typo histórico

    row["wikidata_id"] = wikidata_id
    row["wikipedia_title"] = wiki_meta.get("wikipedia_title")
    row["source_language"] = wiki_meta.get("source_language")

    return row, meta_sugg, logs