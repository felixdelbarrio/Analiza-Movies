from __future__ import annotations

"""
backend/collection_analysis.py

Orquestador “por item” que conecta el pipeline completo para UNA película.

Este módulo coordina (no decide reglas de negocio):
- Entrada unificada (MovieInput).
- Resolución LAZY de OMDb (callback fetch_omdb).
- Resolución LAZY de Wiki (solo si compensa) con negative caching intra-run.
- Enriquecimiento final para reporting (poster, imdb_id, wiki minimal...).
- Sugerencias de metadata (solo Plex).

✅ Optimizaciones / robustez incluidas
-------------------------------------
1) LRU bounded (in-memory) + negative caching intra-run:
   - OMDb local cache: dict (HIT) / _CACHE_MISS (ya intentado y no hay nada).
   - Wiki local cache: dict (HIT) / _CACHE_MISS.
   - Tamaños configurables:
        COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS
        COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS
     Si <= 0 => cache deshabilitado.

2) Lazy Wiki conservador:
   - Solo si:
        a) config COLLECTION_ENABLE_LAZY_WIKI
        b) OMDb no trae __wiki minimal
        c) _should_fetch_wiki_for_reporting(base_row) = True
   - Si falla o es negativo => cachea MISS intra-run para evitar reintentos.

   ✅ NUEVO: fallback opcional sin imdb_id
   - Config: COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK
   - Si no hay imdb_id (ni omdb ni hint) y está habilitado:
        - permite intentar Wiki por (title, year) como último recurso.

3) Persistencia opcional de wiki minimal en OMDb cache:
   - config COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE
   - usa patch_cached_omdb_record si está disponible (compat).

4) Logs:
   - NO imprime progreso por item.
   - Devuelve logs acotados; orquestadores deciden si mostrar según SILENT/DEBUG.
   - Se apoya en backend/logger.py (append_bounded_log, debug_ctx, truncate_line).

   ✅ NUEVO: trazas también a debug_ctx (opcional)
   - Config: COLLECTION_TRACE_ALSO_DEBUG_CTX
   - Si False: las trazas se quedan solo en el buffer `logs` del item.

5) Payload OMDb JSON en report:
   ✅ NUEVO: COLLECTION_OMDB_JSON_MODE
   - "auto"   (default): comportamiento previo (solo si not silent o debug)
   - "never"  : nunca incluir omdb_json
   - "always" : siempre incluir omdb_json (⚠️ tamaño reports)

⚠️ Este módulo NO hace flush por item:
- expone flush_external_caches() para que el orquestador lo llame UNA vez al final.
"""

import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, Final, TypeAlias, cast

from backend import logger
from backend.analyze_input_core import AnalysisRow, analyze_input_movie
from backend.config_collection import (
    COLLECTION_ENABLE_LAZY_WIKI,
    COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK,
    COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE,
    COLLECTION_OMDB_JSON_MODE,
    COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS,
    COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE,
    COLLECTION_TRACE_ALSO_DEBUG_CTX,
    COLLECTION_TRACE_LINE_MAX_CHARS,
    COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS,
)
from backend.metadata_fix import generate_metadata_suggestions_row
from backend.movie_input import MovieInput, normalize_title_for_lookup
from backend.omdb_client import omdb_query_with_cache
from backend.wiki_client import get_wiki, get_wiki_for_input

# -----------------------------------------------------------------------------
# ✅ Anti-"unreachable" definitivo:
# 1) Evitamos `return` en helpers (Pyright puede marcarlos unreachable si
#    constant-fold de configs o callable()).
# 2) Usamos wrapper _cfg_bool que "ensancha" a bool para que no se trate como Literal.
# -----------------------------------------------------------------------------
_LOGGER: Any = logger


def _cfg_bool(v: object) -> bool:
    # cast -> bool (no Literal) a ojos del type-checker
    return bool(cast(bool, v))


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


# ============================================================================
# Cache local (módulo): LRU + negative caching intra-run
# ============================================================================

_OmdbCacheKey: TypeAlias = tuple[str, str, str | None]
# key: (norm_title, norm_year_str, imdb_hint_norm)

_CACHE_MISS: Final[object] = object()

_OMDB_LOCAL_CACHE_MAX: Final[int] = int(COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS)
_WIKI_LOCAL_CACHE_MAX: Final[int] = int(COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS)

# Guardamos dict (HIT) o _CACHE_MISS (negative cached).
_OMDB_LOCAL_CACHE: "OrderedDict[_OmdbCacheKey, object]" = OrderedDict()
_WIKI_LOCAL_CACHE: "OrderedDict[_OmdbCacheKey, object]" = OrderedDict()

# Un único lock para operaciones LRU (consistencia de move_to_end + popitem).
_LOCAL_CACHE_LOCK = threading.Lock()

# Serializa write-back a omdb_cache.json para evitar condiciones de carrera.
_OMDB_CACHE_WRITE_LOCK = threading.Lock()


def _lru_get(cache: "OrderedDict[_OmdbCacheKey, object]", key: _OmdbCacheKey) -> object | None:
    """
    LRU get (NO thread-safe; llamar bajo _LOCAL_CACHE_LOCK).

    Returns:
      - dict (hit)
      - _CACHE_MISS (negative-cached)
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
    LRU set (NO thread-safe; llamar bajo _LOCAL_CACHE_LOCK).

    - Si max_items <= 0 => cache deshabilitado: limpiamos y no almacenamos.
    - Si habilitado => set + recorte por tamaño expulsando LRU.
    """
    if max_items <= 0:
        cache.clear()
        return

    cache[key] = value
    try:
        cache.move_to_end(key, last=True)
    except Exception:
        pass

    while len(cache) > max_items:
        try:
            cache.popitem(last=False)
        except Exception:
            break


# ============================================================================
# Logging helpers (alineados con backend/logger.py)
# ============================================================================

_TRACE_LINE_MAX_CHARS: Final[int] = int(COLLECTION_TRACE_LINE_MAX_CHARS)


def _append_log(logs: list[str], line: object, *, force: bool = False, tag: str | None = None) -> None:
    """
    ✅ NO early-return: evita "Statement is unreachable" si Pyright constant-fold.
    """
    fn_obj: object = getattr(_LOGGER, "append_bounded_log", None)
    if callable(fn_obj):
        try:
            cast(Callable[..., object], fn_obj)(logs, line, force=force, tag=tag)
        except Exception:
            pass


def _append_trace(logs: list[str], line: object) -> None:
    try:
        trunc_fn: object = getattr(_LOGGER, "truncate_line", None)
        if callable(trunc_fn):
            text = cast(Callable[..., str], trunc_fn)(str(line), max_chars=_TRACE_LINE_MAX_CHARS)
        else:
            text = str(line)
    except Exception:
        text = "<unprintable>"
    _append_log(logs, f"[TRACE] {text}")


def _dbg_ctx(msg: object) -> None:
    fn_obj: object = getattr(_LOGGER, "debug_ctx", None)
    if callable(fn_obj):
        try:
            cast(Callable[..., object], fn_obj)("COLLECTION", msg)
        except Exception:
            pass


def _dbg_trace(msg: object) -> None:
    # ✅ sin `return` -> no hay "return unreachable"
    if _cfg_bool(COLLECTION_TRACE_ALSO_DEBUG_CTX):
        _dbg_ctx(msg)


def _log_error_always(message: object) -> None:
    fn_obj: object = getattr(_LOGGER, "error", None)
    if callable(fn_obj):
        try:
            cast(Callable[..., object], fn_obj)(message, always=True)
        except Exception:
            pass


def _log_warning_always(message: object) -> None:
    fn_obj: object = getattr(_LOGGER, "warning", None)
    if callable(fn_obj):
        try:
            cast(Callable[..., object], fn_obj)(message, always=True)
        except Exception:
            pass


# ============================================================================
# Normalización / helpers defensivos
# ============================================================================


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.upper() == "N/A":
            return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None
    return None


def _norm_title(title: str) -> str:
    try:
        out = normalize_title_for_lookup(title or "")
        return out or (title or "").strip().lower()
    except Exception:
        return (title or "").strip().lower()


def _norm_year_str(year: int | None) -> str:
    return str(year) if year is not None else ""


def _norm_imdb_hint(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    return v or None


def _cache_key(title_for_fetch: str, year_for_fetch: int | None, imdb_hint: str | None) -> _OmdbCacheKey:
    return (_norm_title(title_for_fetch), _norm_year_str(year_for_fetch), imdb_hint)


def _extract_wiki_meta(omdb_record: Mapping[str, object] | None) -> dict[str, object]:
    if not omdb_record:
        return {}
    wiki_raw = omdb_record.get("__wiki")
    if isinstance(wiki_raw, Mapping):
        return dict(wiki_raw)
    return {}


def _build_lookup_key(title_for_fetch: str, year_for_fetch: int | None, imdb_hint: str | None) -> str:
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
    if _cfg_bool(COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE) and patch_cached_omdb_record is not None:
        imdb_id_final = imdb_id_for_cache or imdb_hint
        norm_title = _norm_title(title_for_fetch)
        norm_year = _norm_year_str(year_for_fetch)

        try:
            with _OMDB_CACHE_WRITE_LOCK:
                patch_cached_omdb_record(
                    norm_title=norm_title,
                    norm_year=norm_year,
                    imdb_id=imdb_id_final,
                    patch={
                        "__wiki": dict(minimal_wiki),
                        "__prov": {
                            "lookup_key": lookup_key,
                            "had_imdb_hint": bool(imdb_hint),
                        },
                    },
                )
            _dbg_ctx(f"Persisted minimal wiki into OMDb cache | lookup_key={lookup_key}")
        except Exception as exc:  # pragma: no cover
            _dbg_ctx(f"patch_cached_omdb_record failed | lookup_key={lookup_key} exc={exc!r}")


def _wiki_item_is_usable(wiki_item: Mapping[str, object]) -> bool:
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
    misidentified_hint = base_row.get("misidentified_hint")
    if isinstance(misidentified_hint, str) and misidentified_hint.strip():
        return True

    decision = base_row.get("decision")
    if isinstance(decision, str) and decision.strip().upper() in ("DELETE", "MAYBE"):
        return True

    return False


def _should_include_omdb_json() -> bool:
    mode = (COLLECTION_OMDB_JSON_MODE or "auto").strip().lower()
    if mode == "never":
        return False
    if mode == "always":
        return True
    return (not logger.is_silent_mode()) or logger.is_debug_mode()


# ============================================================================
# API: flush explícito (para orquestadores)
# ============================================================================


def flush_external_caches() -> None:
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


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================


def analyze_movie(
    movie_input: MovieInput,
    *,
    source_movie: object | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[str]]:
    logs: list[str] = []

    # 0) Preferencia de display (Plex puede sobrescribir)
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

    # 1) Callback fetch_omdb: LRU + negative intra-run + read-through de __wiki
    omdb_data: dict[str, object] | None = None
    wiki_meta: dict[str, object] = {}

    def fetch_omdb(title_for_fetch: str, year_for_fetch: int | None) -> Mapping[str, object]:
        nonlocal omdb_data, wiki_meta

        key = _cache_key(title_for_fetch, year_for_fetch, imdb_hint)

        with _LOCAL_CACHE_LOCK:
            cached_omdb = _lru_get(_OMDB_LOCAL_CACHE, key)
            cached_wiki = _lru_get(_WIKI_LOCAL_CACHE, key)

        if cached_omdb is _CACHE_MISS:
            omdb_data = {}
            wiki_meta = {}
            return {}

        if isinstance(cached_omdb, dict):
            omdb_data = cached_omdb
            wiki_meta = dict(cached_wiki) if isinstance(cached_wiki, dict) else {}
            return cached_omdb

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
                _lru_set(_OMDB_LOCAL_CACHE, key, _CACHE_MISS, max_items=_OMDB_LOCAL_CACHE_MAX)
                _lru_set(_WIKI_LOCAL_CACHE, key, {}, max_items=_WIKI_LOCAL_CACHE_MAX)
            omdb_data = {}
            wiki_meta = {}
            _append_log(logs, f"lookup_key={lookup_key}", tag="OMDB_NONE")
            return {}

        omdb_dict_local = dict(record)
        wiki_dict_local = _extract_wiki_meta(omdb_dict_local)

        with _LOCAL_CACHE_LOCK:
            _lru_set(_OMDB_LOCAL_CACHE, key, omdb_dict_local, max_items=_OMDB_LOCAL_CACHE_MAX)
            _lru_set(_WIKI_LOCAL_CACHE, key, wiki_dict_local, max_items=_WIKI_LOCAL_CACHE_MAX)

        omdb_data = omdb_dict_local
        wiki_meta = wiki_dict_local

        if wiki_dict_local:
            _dbg_ctx(f"Using __wiki from OMDb cache | lookup_key={lookup_key}")

        return omdb_dict_local

    # 2) Plex rating (si aplica)
    plex_rating: float | None = None
    if movie_input.source == "plex" and source_movie is not None:
        plex_user_rating = getattr(source_movie, "userRating", None)
        plex_rating_raw = getattr(source_movie, "rating", None)
        plex_rating = _safe_float(plex_user_rating) or _safe_float(plex_rating_raw)

    # 3) Core
    def _analysis_trace(line: str) -> None:
        _append_trace(logs, line)
        _dbg_trace(line)

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
        _log_error_always(msg)
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

    # 4) Lazy Wiki
    if _cfg_bool(COLLECTION_ENABLE_LAZY_WIKI) and not wiki_meta and _should_fetch_wiki_for_reporting(base_row):
        key_post = _cache_key(movie_input.title, movie_input.year, imdb_hint)

        with _LOCAL_CACHE_LOCK:
            wiki_local = _lru_get(_WIKI_LOCAL_CACHE, key_post)

        if wiki_local is _CACHE_MISS:
            _dbg_ctx("Lazy Wiki skipped: negative-cached intra-run.")
        elif isinstance(wiki_local, dict) and wiki_local:
            wiki_meta = dict(wiki_local)
        else:
            if omdb_data is None and _cfg_bool(COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE):
                _dbg_ctx("Lazy Wiki -> forcing OMDb fetch (post-core) due to config.")
                _ = fetch_omdb(movie_input.title, movie_input.year)

            omdb_dict_for_wiki = omdb_data or {}
            lookup_key = _build_lookup_key(movie_input.title, movie_input.year, imdb_hint)

            imdb_id_from_omdb: str | None = None
            imdb_raw = omdb_dict_for_wiki.get("imdbID")
            if isinstance(imdb_raw, str) and imdb_raw.strip():
                imdb_id_from_omdb = imdb_raw.strip().lower()

            imdb_used_for_wiki = imdb_id_from_omdb or imdb_hint

            if not imdb_used_for_wiki and not _cfg_bool(COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK):
                with _LOCAL_CACHE_LOCK:
                    _lru_set(_WIKI_LOCAL_CACHE, key_post, _CACHE_MISS, max_items=_WIKI_LOCAL_CACHE_MAX)
                _dbg_ctx("Lazy Wiki skipped: no imdb id available (omdb/imdb_hint) and fallback disabled.")
            else:
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

                    imdb_id_from_wiki: str | None = None
                    imdb_cached = wiki_item.get("imdbID")
                    if isinstance(imdb_cached, str) and imdb_cached.strip():
                        imdb_id_from_wiki = imdb_cached.strip().lower()

                    wikidata_id_from_wiki: str | None = None
                    if isinstance(wikidata_block, Mapping):
                        qid_val = wikidata_block.get("qid")
                        if isinstance(qid_val, str) and qid_val.strip():
                            wikidata_id_from_wiki = qid_val.strip()

                    wikipedia_title_from_wiki: str | None = None
                    source_language_from_wiki: str | None = None
                    if isinstance(wiki_block, Mapping):
                        wt = wiki_block.get("wikipedia_title")
                        if isinstance(wt, str) and wt.strip():
                            wikipedia_title_from_wiki = wt.strip()
                        sl = wiki_block.get("source_language")
                        if isinstance(sl, str) and sl.strip():
                            source_language_from_wiki = sl.strip()

                    wiki_lookup = _build_wiki_lookup_info(
                        title_for_fetch=movie_input.title,
                        year_for_fetch=movie_input.year,
                        imdb_used=imdb_used_for_wiki or imdb_id_from_wiki,
                    )

                    minimal_wiki = _build_minimal_wiki_block(
                        imdb_id=(imdb_id_from_wiki or imdb_used_for_wiki),
                        wikidata_id=wikidata_id_from_wiki,
                        wikipedia_title=wikipedia_title_from_wiki,
                        wiki_lookup=wiki_lookup,
                        source_language=source_language_from_wiki,
                    )

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
                        _lru_set(_OMDB_LOCAL_CACHE, key_post, dict(omdb_dict_for_wiki), max_items=_OMDB_LOCAL_CACHE_MAX)
                        _lru_set(_WIKI_LOCAL_CACHE, key_post, dict(minimal_wiki), max_items=_WIKI_LOCAL_CACHE_MAX)

                    omdb_data = dict(omdb_dict_for_wiki)
                    wiki_meta = dict(minimal_wiki)
                    _dbg_ctx(f"Lazy Wiki success | lookup_key={lookup_key}")
                else:
                    with _LOCAL_CACHE_LOCK:
                        _lru_set(_WIKI_LOCAL_CACHE, key_post, _CACHE_MISS, max_items=_WIKI_LOCAL_CACHE_MAX)
                    _dbg_ctx(f"Lazy Wiki negative/None -> intra-run MISS cached | lookup_key={lookup_key}")

    # 5) Sugerencias metadata (solo Plex)
    omdb_dict: dict[str, object] = dict(omdb_data) if isinstance(omdb_data, dict) else {}
    meta_sugg: dict[str, object] | None = None

    if movie_input.source == "plex" and source_movie is not None:
        try:
            meta_candidate = generate_metadata_suggestions_row(movie_input, omdb_dict or None)
            if isinstance(meta_candidate, dict):
                meta_sugg = meta_candidate
                _append_log(logs, f"{movie_input.library} / {display_title} ({display_year})", tag="METADATA_SUGG")
        except Exception as exc:  # pragma: no cover
            _log_warning_always(f"generate_metadata_suggestions_row falló para {display_title!r}: {exc!r}")

    misidentified_hint = base_row.get("misidentified_hint")
    if isinstance(misidentified_hint, str) and misidentified_hint.strip():
        _append_log(
            logs,
            f"{movie_input.library} / {display_title} ({display_year}): {misidentified_hint}",
            tag="MISIDENTIFIED",
        )

    # 6) Enriquecimiento para reporting
    poster_url: str | None = None
    trailer_url: str | None = None
    imdb_id: str | None = None
    omdb_json_str: str | None = None

    if omdb_dict:
        poster_raw = omdb_dict.get("Poster")
        trailer_raw = omdb_dict.get("Website")
        imdb_id_raw = omdb_dict.get("imdbID")

        poster_url = poster_raw.strip() if isinstance(poster_raw, str) and poster_raw.strip() else None
        trailer_url = trailer_raw.strip() if isinstance(trailer_raw, str) and trailer_raw.strip() else None
        imdb_id = imdb_id_raw.strip().lower() if isinstance(imdb_id_raw, str) and imdb_id_raw.strip() else None

    if imdb_id is None and imdb_hint is not None:
        imdb_id = imdb_hint

    if omdb_dict and _should_include_omdb_json():
        try:
            omdb_json_str = json.dumps(omdb_dict, ensure_ascii=False)
        except Exception:
            omdb_json_str = str(omdb_dict)

    # 7) Fila final
    row: dict[str, object] = dict(base_row)

    row["source"] = movie_input.source
    row["library"] = movie_input.library
    row["title"] = display_title
    row["year"] = display_year

    file_size_bytes = row.get("file_size_bytes")
    row["file_size"] = file_size_bytes if isinstance(file_size_bytes, int) else movie_input.file_size_bytes

    file_from_row = row.get("file")
    file_from_row_str = file_from_row if isinstance(file_from_row, str) else ""
    row["file"] = (movie_input.file_path or file_from_row_str)

    source_url_obj = movie_input.extra.get("source_url")
    row["source_url"] = source_url_obj if isinstance(source_url_obj, str) else ""

    row["rating_key"] = movie_input.rating_key
    row["guid"] = movie_input.plex_guid
    row["thumb"] = movie_input.thumb_url

    row["imdb_id"] = imdb_id
    row["poster_url"] = poster_url
    row["trailer_url"] = trailer_url
    row["omdb_json"] = omdb_json_str

    wikidata_id_raw = wiki_meta.get("wikidata_id")
    wikidata_id: str | None = (
        wikidata_id_raw.strip() if isinstance(wikidata_id_raw, str) and wikidata_id_raw.strip() else None
    )
    if wikidata_id is None:
        wikadata_id_raw = wiki_meta.get("wikadata_id")
        wikidata_id = wikadata_id_raw.strip() if isinstance(wikadata_id_raw, str) and wikadata_id_raw.strip() else None

    wikipedia_title_raw = wiki_meta.get("wikipedia_title")
    wikipedia_title: str | None = (
        wikipedia_title_raw.strip() if isinstance(wikipedia_title_raw, str) and wikipedia_title_raw.strip() else None
    )

    source_language_raw = wiki_meta.get("source_language")
    source_language: str | None = (
        source_language_raw.strip() if isinstance(source_language_raw, str) and source_language_raw.strip() else None
    )

    row["wikidata_id"] = wikidata_id
    row["wikipedia_title"] = wikipedia_title
    row["source_language"] = source_language

    return row, meta_sugg, logs