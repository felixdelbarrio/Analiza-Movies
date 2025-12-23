from __future__ import annotations

"""
backend/wiki_client.py (schema v6)

Cliente “best-effort” para enriquecer títulos con Wikipedia/Wikidata con caché persistente,
sin wrappers OO (no hay WikiClient / get_wiki_client).

✅ Objetivos (v6)
----------------
1) API estable y simple:
   - get_wiki(title, year, imdb_id) -> WikiCacheItem | None
   - get_wiki_for_input(movie_input, title, year, imdb_id) -> WikiCacheItem | None
   - flush_wiki_cache() -> None   (para flush_external_caches() en el orquestador)

2) Negative caching REAL (persistente + TTL):
   - Si NO encontramos QID o el QID NO parece film -> guardamos item negativo (status != "ok")
   - Evita repetir SPARQL / Wikipedia calls caros en futuros runs.

3) Rendimiento / robustez:
   - Schema v6 optimizado: records + indexes (sin duplicación imdb:* y ty:* apuntando al mismo dict)
       - records[rid] -> WikiCacheItem (1 record por película)
       - index_imdb[imdb_id] -> rid
       - index_ty["<norm_title>|<norm_year>"] -> rid
   - TTL por estado:
       - ok: TTL largo
       - negativos: TTL más corto (pero suficiente para evitar martilleo)
       - imdb->qid negativos: TTL separado y conservador
   - Write-back batching: flush por umbral de escrituras sucias o por tiempo.
   - Escritura atómica del JSON (temp + replace).
   - Compaction/GC antes de escribir:
       - elimina expirados
       - caps (records / imdb_qid / is_film / entities)
       - rebuild indexes coherentes
       - prune entities a QIDs realmente referenciados

4) Política de schema:
   - NO migración automática (por diseño de esta iteración).
   - schema mismatch => se recrea cache vacío (se regenerará desde cero).

5) Logs alineados con backend/logger.py:
   - NO prints.
   - Debug contextual SOLO con logger.debug_ctx("WIKI", ...).
   - Info SOLO si NO silent y aporta valor.
   - Error siempre visible con logger.error(..., always=True).

Notas
-----
- “Best-effort”: puede devolver None si hay fallos transitorios de red.
  (No cacheamos fallos de red como negativos para no “congelar” errores.)
- ThreadPool safe: locks globales alrededor de IO y mutaciones del cache.
- SPARQL throttle global con lock (evita tormentas de requests concurrentes).
"""

import atexit
import json
import re
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend import logger
from backend.config import (
    # Paths / idioma
    WIKI_CACHE_PATH,
    WIKI_LANGUAGE,
    WIKI_FALLBACK_LANGUAGE,
    # Throttle / TTLs
    WIKI_SPARQL_MIN_INTERVAL_SECONDS,
    WIKI_CACHE_TTL_OK_SECONDS,
    WIKI_CACHE_TTL_NEGATIVE_SECONDS,
    WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS,
    WIKI_IS_FILM_TTL_SECONDS,
    # Write-back batching
    WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES,
    WIKI_CACHE_FLUSH_MAX_SECONDS,
    # Compaction caps (centralizado; opt-in tuning desde env pero parseado en config.py)
    ANALIZA_WIKI_CACHE_MAX_RECORDS,
    ANALIZA_WIKI_CACHE_MAX_IMDB_QID,
    ANALIZA_WIKI_CACHE_MAX_IS_FILM,
    ANALIZA_WIKI_CACHE_MAX_ENTITIES,
    # Debug extra (complemento; no fuerza DEBUG_MODE)
    ANALIZA_WIKI_DEBUG,
)
from backend.movie_input import normalize_title_for_lookup


# ============================================================================
# Tipos: dict-like (compatibilidad con el pipeline)
# ============================================================================

class WikidataEntity(dict):
    """
    Entidad etiquetada (qid -> label/description/type).

    Se persiste en cache["entities"] para hidratar listas de QIDs (directores/países/géneros)
    con labels humanas sin volver a pedirlas en runs posteriores.

    Keys típicas:
      - label: str
      - description: str | None
      - type: str (person/country/genre) (opcional)
    """


class WikiBlock(dict):
    """
    Bloque Wikipedia (REST summary) normalizado.

    Keys típicas:
      - language, fallback_language: idiomas “preferidos” para este item (config/per-item)
      - source_language: idioma real del summary usado
      - wikipedia_title, wikipedia_pageid, wikibase_item (QID)
      - summary, description
      - urls, images
    """


class WikidataBlock(dict):
    """
    Bloque Wikidata:
      - qid: str
      - directors/countries/genres: list[str] de QIDs
    """


WikiItemStatus = Literal["ok", "no_qid", "not_film", "imdb_no_qid"]


class WikiCacheItem(dict):
    """
    Entrada principal cacheable (records[rid]).

    Campos esperados:
      - Title: str (normalizado por normalize_title_for_lookup)
      - Year: str ("" si no hay año)
      - imdbID: str | None (normalizado: lowercase)
      - wiki: WikiBlock
      - wikidata: WikidataBlock
      - fetched_at: int (epoch seconds)
      - ttl_s: int (seconds)
      - status: ok | no_qid | not_film | imdb_no_qid
    """


class ImdbQidCacheEntry(dict):
    """
    Cache imdbID -> QID (P345), con negative caching (qid=None).

    Keys:
      - qid: str | None
      - fetched_at: int
      - ttl_s: int
    """


class IsFilmCacheEntry(dict):
    """
    Cache QID -> is_film (heurística rápida, persistente).

    Keys:
      - is_film: bool
      - fetched_at: int
      - ttl_s: int
    """


class WikiCacheFile(dict):
    """
    Schema v6 (optimizado, sin duplicación):

    Keys:
      - schema
      - language/fallback_language
      - records: dict[rid, WikiCacheItem]
      - index_imdb: dict[imdb_id_norm, rid]
      - index_ty: dict["<norm_title>|<norm_year>", rid]
      - entities: dict[qid, WikidataEntity]
      - imdb_qid: dict[imdb_id_norm, ImdbQidCacheEntry]
      - is_film: dict[qid, IsFilmCacheEntry]
    """


# ============================================================================
# Protocol para idioma por item (evita acoplamiento fuerte)
# ============================================================================

class MovieInputLangProto(Protocol):
    """
    Interfaz mínima para extraer idioma/contexto por biblioteca o por heurística.
    Evita acoplar el cliente a una clase concreta.
    """
    def plex_library_language(self) -> str | None: ...
    def is_spanish_context(self) -> bool: ...
    def is_english_context(self) -> bool: ...
    def is_italian_context(self) -> bool: ...
    def is_french_context(self) -> bool: ...
    def is_japanese_context(self) -> bool: ...
    def is_korean_context(self) -> bool: ...
    def is_chinese_context(self) -> bool: ...


# ============================================================================
# Constantes / estado interno
# ============================================================================

_SCHEMA_VERSION: Final[int] = 6

# Centralizado en config.py (evita duplicación y facilita overrides)
_CACHE_PATH: Final[Path] = Path(WIKI_CACHE_PATH)

# HTTP Session singleton (lazy-init) + lock
_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()

# Cache singleton + lock
_CACHE: WikiCacheFile | None = None
_CACHE_LOCK = threading.RLock()

# Dirty tracking / batching
_CACHE_DIRTY: bool = False
_CACHE_DIRTY_WRITES: int = 0
_CACHE_LAST_FLUSH_TS: float = 0.0

# SPARQL throttle global (para ser buenos ciudadanos)
_LAST_SPARQL_TS: float = 0.0
_SPARQL_THROTTLE_LOCK = threading.Lock()
_SPARQL_MIN_INTERVAL_S: Final[float] = float(WIKI_SPARQL_MIN_INTERVAL_SECONDS)

# TTLs (seconds)
_TTL_OK_S: Final[int] = int(WIKI_CACHE_TTL_OK_SECONDS)
_TTL_NEGATIVE_S: Final[int] = int(WIKI_CACHE_TTL_NEGATIVE_SECONDS)
_TTL_IMDB_QID_NEGATIVE_S: Final[int] = int(WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS)
_TTL_IS_FILM_S: Final[int] = int(WIKI_IS_FILM_TTL_SECONDS)

# Flush knobs (write-back batching)
_FLUSH_MAX_DIRTY_WRITES: Final[int] = int(WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES)
_FLUSH_MAX_SECONDS: Final[float] = float(WIKI_CACHE_FLUSH_MAX_SECONDS)

# Caps de compaction (centralizados en config.py)
_COMPACT_MAX_RECORDS: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_RECORDS)
_COMPACT_MAX_IMDB_QID: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_IMDB_QID)
_COMPACT_MAX_IS_FILM: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_IS_FILM)
_COMPACT_MAX_ENTITIES: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_ENTITIES)

# Debug extra (no fuerza logger.is_debug_mode; solo habilita mensajes “más verbosos” si procede)
_WIKI_DEBUG_EXTRA: Final[bool] = bool(ANALIZA_WIKI_DEBUG)

# “Film instances” allowlist (P31) — conservador (prefiere false-negative a false-positive).
_FILM_INSTANCE_ALLOWLIST: Final[set[str]] = {
    "Q11424",   # film
    "Q24862",   # feature film
    "Q202866",  # animated film
    "Q226730",  # television film
    "Q93204",   # short film
}

# Tokenizador básico para canonicalización de búsquedas
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Timeouts (en segundos) — separados para facilidad de tuning futuro
_HTTP_TIMEOUT_SHORT: Final[float] = 10.0
_HTTP_TIMEOUT_SPARQL: Final[tuple[float, float]] = (5.0, 45.0)  # (connect, read)


# ============================================================================
# Logging helpers (SIN política propia)
# ============================================================================

def _dbg(msg: object) -> None:
    """
    Debug contextual.

    - Solo emite si logger.is_debug_mode() True.
    - El “cómo” (silent->progress vs normal->info) lo decide logger.debug_ctx.
    - _WIKI_DEBUG_EXTRA habilita emitir más señales (sin forzar DEBUG_MODE).
      Aun así, si DEBUG_MODE está apagado, aquí no se emitirá nada.
    """
    if not logger.is_debug_mode():
        return
    # En debug mode, siempre permitimos debug básico. El flag “extra” queda para
    # futuras ampliaciones (p.ej. dumps de contadores), sin cambiar la política.
    logger.debug_ctx("WIKI", msg)


def _info(msg: str) -> None:
    """Info: solo si NO silent."""
    if logger.is_silent_mode():
        return
    logger.info(msg)


def _err(msg: str) -> None:
    """Errores: siempre visibles."""
    logger.error(msg, always=True)


# ============================================================================
# Session / HTTP
# ============================================================================

def _get_session() -> requests.Session:
    """
    requests.Session singleton con Retry.

    Motivos:
    - Pooling de conexiones: crítico en ThreadPool.
    - Retry conservador para 429/5xx.
    - Respeta Retry-After cuando existe.

    Nota:
    - NO usamos prints; todo logging pasa por logger.debug_ctx.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Analiza-Movies/1.0 (local)",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": f"{WIKI_LANGUAGE},{WIKI_FALLBACK_LANGUAGE};q=0.8,en;q=0.6,es;q=0.5",
            }
        )

        retries = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _SESSION = session
        return session


# ============================================================================
# Helpers generales (defensivos)
# ============================================================================

def _now_epoch() -> int:
    """Epoch seconds (int)."""
    return int(time.time())


def _safe_str(value: object) -> str | None:
    """Devuelve str strip() o None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v or None


def _safe_int(value: object) -> int | None:
    """Parsea int de forma tolerante (int/float/str numérico)."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _normalize_lang_code(lang: str) -> str:
    """
    Normaliza códigos comunes:

      - es-ES / es_ES -> es
      - spa -> es
      - eng -> en
      - jpn/jp -> ja
      - kor -> ko
      - zho/chi -> zh

    Devuelve "" si no hay entrada usable.
    """
    l = lang.strip().lower().replace("_", "-")
    if not l:
        return ""

    iso_map: dict[str, str] = {
        "spa": "es",
        "eng": "en",
        "ita": "it",
        "fra": "fr",
        "fre": "fr",
        "jpn": "ja",
        "jp": "ja",
        "kor": "ko",
        "zho": "zh",
        "chi": "zh",
    }
    base = l.split("-", 1)[0]
    return iso_map.get(base, base)


def _norm_imdb(imdb_id: str | None) -> str | None:
    """
    Normaliza imdbID para keys/caches:
      - strip + lower
    """
    if not isinstance(imdb_id, str):
        return None
    v = imdb_id.strip().lower()
    return v or None


def _ty_key(norm_title: str, norm_year: str) -> str:
    """
    Key compacta para index_ty.
    """
    return f"{norm_title}|{norm_year}"


def _is_expired(fetched_at: int, ttl_s: int, now_epoch: int) -> bool:
    """
    Expiración simple por TTL:
    - si faltan campos o ttl inválido => consideramos expirado (defensivo).
    """
    if fetched_at <= 0 or ttl_s <= 0:
        return True
    return (now_epoch - fetched_at) > ttl_s


def _mark_dirty_unlocked() -> None:
    """Marca el cache como dirty y cuenta escrituras (para batching)."""
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES
    _CACHE_DIRTY = True
    _CACHE_DIRTY_WRITES += 1


# ============================================================================
# Cache IO (atomic write + load/validate)
# ============================================================================

def _save_cache_file_atomic(cache: WikiCacheFile) -> None:
    """
    Escritura atómica del JSON:
    - temp file en el mismo directorio
    - fsync (best-effort)
    - os.replace (atómico)

    Nota:
    - Usamos NamedTemporaryFile(delete=False) para poder os.replace en Windows/macOS.
    """
    dirpath = _CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(dirpath)) as tf:
            json.dump(cache, tf, ensure_ascii=False, indent=2)
            tf.flush()
            try:
                import os
                os.fsync(tf.fileno())
            except Exception:
                pass
            temp_name = tf.name

        import os
        os.replace(temp_name, str(_CACHE_PATH))
    finally:
        if temp_name:
            try:
                import os
                if os.path.exists(temp_name) and temp_name != str(_CACHE_PATH):
                    os.remove(temp_name)
            except Exception:
                pass


def _maybe_flush_unlocked(force: bool) -> None:
    """
    Flush del cache si procede:
    - force=True: flush explícito / salida
    - si dirty y supera umbral de writes
    - si dirty y supera umbral de tiempo desde último flush

    Importante:
    - Antes de persistir, hacemos compaction/GC para evitar crecimiento indefinido.
    """
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES, _CACHE_LAST_FLUSH_TS, _CACHE
    if _CACHE is None:
        return
    if not _CACHE_DIRTY and not force:
        return

    now = time.monotonic()
    should_flush = force
    if not should_flush:
        if _CACHE_DIRTY_WRITES >= _FLUSH_MAX_DIRTY_WRITES:
            should_flush = True
        elif (now - _CACHE_LAST_FLUSH_TS) >= _FLUSH_MAX_SECONDS:
            should_flush = True

    if not should_flush:
        return

    try:
        _compact_cache_unlocked(_CACHE, force=force)
        _save_cache_file_atomic(_CACHE)

        _CACHE_DIRTY = False
        _CACHE_DIRTY_WRITES = 0
        _CACHE_LAST_FLUSH_TS = now
    except Exception as exc:
        _dbg(f"cache flush failed: {exc!r}")


def flush_wiki_cache() -> None:
    """
    API pública: flush explícito del cache Wiki (si está dirty).

    Se llama típicamente al final del run mediante flush_external_caches().
    También se registra en atexit como red de seguridad.
    """
    try:
        with _CACHE_LOCK:
            _maybe_flush_unlocked(force=True)
    except Exception as exc:
        _dbg(f"flush_wiki_cache failed: {exc!r}")


atexit.register(flush_wiki_cache)


def _empty_cache() -> WikiCacheFile:
    """
    Construye una estructura de cache vacía (schema v6).

    Nota:
    - language/fallback_language se guardan como metadata informativa.
    """
    return {
        "schema": _SCHEMA_VERSION,
        "language": str(WIKI_LANGUAGE),
        "fallback_language": str(WIKI_FALLBACK_LANGUAGE),
        "records": {},
        "index_imdb": {},
        "index_ty": {},
        "entities": {},
        "imdb_qid": {},
        "is_film": {},
    }


def _load_cache_unlocked() -> WikiCacheFile:
    """
    Carga cache una sola vez (singleton). Si está corrupto / schema mismatch -> recrea.

    Política v6:
    - NO migración automática: si el schema no coincide, arrancamos vacío.
    """
    global _CACHE, _CACHE_DIRTY, _CACHE_DIRTY_WRITES, _CACHE_LAST_FLUSH_TS
    if _CACHE is not None:
        return _CACHE

    _CACHE_DIRTY = False
    _CACHE_DIRTY_WRITES = 0
    _CACHE_LAST_FLUSH_TS = time.monotonic()

    if not _CACHE_PATH.exists():
        _CACHE = _empty_cache()
        return _CACHE

    try:
        raw_obj = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        _dbg(f"cache read failed ({exc!r}); recreating")
        _CACHE = _empty_cache()
        return _CACHE

    if not isinstance(raw_obj, Mapping) or raw_obj.get("schema") != _SCHEMA_VERSION:
        _dbg(f"cache schema mismatch -> recreate (found={raw_obj.get('schema')!r})")
        _CACHE = _empty_cache()
        return _CACHE

    # Estructuras principales esperadas
    records_obj = raw_obj.get("records")
    index_imdb_obj = raw_obj.get("index_imdb")
    index_ty_obj = raw_obj.get("index_ty")
    entities_obj = raw_obj.get("entities")
    imdb_qid_obj = raw_obj.get("imdb_qid")
    is_film_obj = raw_obj.get("is_film")

    if (
        not isinstance(records_obj, Mapping)
        or not isinstance(index_imdb_obj, Mapping)
        or not isinstance(index_ty_obj, Mapping)
        or not isinstance(entities_obj, Mapping)
        or not isinstance(imdb_qid_obj, Mapping)
        or not isinstance(is_film_obj, Mapping)
    ):
        _CACHE = _empty_cache()
        return _CACHE

    # Validación/normalización ligera (best-effort).
    # Importante: index_* puede contener entradas huérfanas; compaction lo reparará.
    records: dict[str, WikiCacheItem] = {}
    for rid, v in records_obj.items():
        if not isinstance(rid, str) or not isinstance(v, Mapping):
            continue

        title = v.get("Title")
        year = v.get("Year")
        imdb_id = v.get("imdbID")
        wiki = v.get("wiki")
        wikidata = v.get("wikidata")
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        status = v.get("status")

        if not isinstance(title, str) or not isinstance(year, str):
            continue
        if imdb_id is not None and not isinstance(imdb_id, str):
            continue
        if not isinstance(wiki, Mapping) or not isinstance(wikidata, Mapping):
            continue
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
            continue
        if status not in ("ok", "no_qid", "not_film", "imdb_no_qid"):
            continue

        records[rid] = WikiCacheItem(
            Title=title,
            Year=year,
            imdbID=_norm_imdb(imdb_id),
            wiki=dict(wiki),
            wikidata=dict(wikidata),
            fetched_at=fetched_at,
            ttl_s=ttl_s,
            status=status,
        )

    index_imdb: dict[str, str] = {}
    for k, v in index_imdb_obj.items():
        imdb = _norm_imdb(k if isinstance(k, str) else None)
        if not imdb or not isinstance(v, str):
            continue
        index_imdb[imdb] = v

    index_ty: dict[str, str] = {}
    for k, v in index_ty_obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip():
            index_ty[k] = v

    entities: dict[str, WikidataEntity] = {}
    for k, v in entities_obj.items():
        qid = _safe_str(k)
        if not qid or not isinstance(v, Mapping):
            continue
        entities[qid] = WikidataEntity(dict(v))

    imdb_qid: dict[str, ImdbQidCacheEntry] = {}
    for k, v in imdb_qid_obj.items():
        imdb = _norm_imdb(k if isinstance(k, str) else None)
        if not imdb or not isinstance(v, Mapping):
            continue
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
            continue
        qid_val = v.get("qid")
        qid = _safe_str(qid_val) if qid_val is not None else None
        imdb_qid[imdb] = ImdbQidCacheEntry(qid=qid, fetched_at=fetched_at, ttl_s=ttl_s)

    is_film: dict[str, IsFilmCacheEntry] = {}
    for k, v in is_film_obj.items():
        qid = _safe_str(k)
        if not qid or not isinstance(v, Mapping):
            continue
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        is_film_val = v.get("is_film")
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int) or not isinstance(is_film_val, bool):
            continue
        is_film[qid] = IsFilmCacheEntry(is_film=is_film_val, fetched_at=fetched_at, ttl_s=ttl_s)

    _CACHE = WikiCacheFile(
        schema=_SCHEMA_VERSION,
        language=str(raw_obj.get("language") or WIKI_LANGUAGE),
        fallback_language=str(raw_obj.get("fallback_language") or WIKI_FALLBACK_LANGUAGE),
        records=records,
        index_imdb=index_imdb,
        index_ty=index_ty,
        entities=entities,
        imdb_qid=imdb_qid,
        is_film=is_film,
    )

    # Compaction ligera al cargar (best-effort):
    # - repara índices huérfanos
    # - elimina expirados
    # No forzamos escritura aquí (evitamos I/O en load), pero dejamos memoria saneada.
    try:
        _compact_cache_unlocked(_CACHE, force=False)
    except Exception:
        pass

    return _CACHE


# ============================================================================
# Compaction / GC (schema v6)
# ============================================================================

def _compact_cache_unlocked(cache: WikiCacheFile, *, force: bool) -> None:
    """
    Compaction/GC:
    - Elimina expirados
    - Cap records / imdb_qid / is_film / entities
    - Rebuild indexes coherentes
    - Prune entities a QIDs realmente usados

    Filosofía:
    - Best-effort: nunca debe romper el pipeline.
    - En flush(force=True) buscamos dejar disco lo más compacto posible.
    - En flush(force=False) hacemos lo justo para evitar crecimiento indefinido.
    """
    try:
        now_epoch = _now_epoch()

        # 1) Records: filtrar expirados / inválidos + normalizar imdbID
        records_obj = cache.get("records")
        if not isinstance(records_obj, Mapping):
            cache["records"] = {}
            records_obj = cache["records"]

        records: dict[str, WikiCacheItem] = {}
        for rid, it in records_obj.items():
            if not isinstance(rid, str) or not isinstance(it, Mapping):
                continue
            fetched_at = it.get("fetched_at")
            ttl_s = it.get("ttl_s")
            if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                continue
            if _is_expired(fetched_at, ttl_s, now_epoch):
                continue

            d = dict(it)
            imdb_norm = _norm_imdb(d.get("imdbID") if isinstance(d.get("imdbID"), str) else None)
            if imdb_norm:
                d["imdbID"] = imdb_norm
            records[rid] = WikiCacheItem(d)

        # 2) Cap records (más nuevos primero)
        if _COMPACT_MAX_RECORDS > 0 and len(records) > _COMPACT_MAX_RECORDS:
            ranked = sorted(records.items(), key=lambda kv: int(kv[1].get("fetched_at", 0)), reverse=True)
            records = dict(ranked[:_COMPACT_MAX_RECORDS])

        cache["records"] = records

        # 3) Rebuild indexes coherentes
        index_imdb: dict[str, str] = {}
        index_ty: dict[str, str] = {}
        for rid, it in records.items():
            title = it.get("Title")
            year = it.get("Year")
            if isinstance(title, str) and isinstance(year, str):
                index_ty[_ty_key(title, year)] = rid

            imdb = _norm_imdb(it.get("imdbID") if isinstance(it.get("imdbID"), str) else None)
            if imdb:
                index_imdb[imdb] = rid

        cache["index_imdb"] = index_imdb
        cache["index_ty"] = index_ty

        # 4) imdb_qid: eliminar expirados + cap
        imdb_qid_obj = cache.get("imdb_qid")
        imdb_qid: dict[str, ImdbQidCacheEntry] = {}
        if isinstance(imdb_qid_obj, Mapping):
            for k, v in imdb_qid_obj.items():
                imdb = _norm_imdb(k if isinstance(k, str) else None)
                if not imdb or not isinstance(v, Mapping):
                    continue
                fetched_at = v.get("fetched_at")
                ttl_s = v.get("ttl_s")
                if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                    continue
                if _is_expired(fetched_at, ttl_s, now_epoch):
                    continue
                imdb_qid[imdb] = ImdbQidCacheEntry(dict(v))

        if _COMPACT_MAX_IMDB_QID > 0 and len(imdb_qid) > _COMPACT_MAX_IMDB_QID:
            ranked = sorted(imdb_qid.items(), key=lambda kv: int(kv[1].get("fetched_at", 0)), reverse=True)
            imdb_qid = dict(ranked[:_COMPACT_MAX_IMDB_QID])
        cache["imdb_qid"] = imdb_qid

        # 5) is_film: eliminar expirados + cap
        is_film_obj = cache.get("is_film")
        is_film: dict[str, IsFilmCacheEntry] = {}
        if isinstance(is_film_obj, Mapping):
            for k, v in is_film_obj.items():
                qid = _safe_str(k)
                if not qid or not isinstance(v, Mapping):
                    continue
                fetched_at = v.get("fetched_at")
                ttl_s = v.get("ttl_s")
                if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                    continue
                if _is_expired(fetched_at, ttl_s, now_epoch):
                    continue
                is_film[qid] = IsFilmCacheEntry(dict(v))

        if _COMPACT_MAX_IS_FILM > 0 and len(is_film) > _COMPACT_MAX_IS_FILM:
            ranked = sorted(is_film.items(), key=lambda kv: int(kv[1].get("fetched_at", 0)), reverse=True)
            is_film = dict(ranked[:_COMPACT_MAX_IS_FILM])
        cache["is_film"] = is_film

        # 6) entities: prune a QIDs referenciados por records
        referenced_qids: set[str] = set()

        for it in records.values():
            wikidata_block = it.get("wikidata")
            if isinstance(wikidata_block, Mapping):
                qid = wikidata_block.get("qid")
                if isinstance(qid, str) and qid.strip():
                    referenced_qids.add(qid.strip())

                for prop in ("directors", "countries", "genres"):
                    qids = wikidata_block.get(prop)
                    if isinstance(qids, list):
                        for q in qids:
                            if isinstance(q, str) and q.strip():
                                referenced_qids.add(q.strip())

            wiki_block = it.get("wiki")
            if isinstance(wiki_block, Mapping):
                wb = wiki_block.get("wikibase_item")
                if isinstance(wb, str) and wb.strip():
                    referenced_qids.add(wb.strip())

        entities_obj = cache.get("entities")
        entities: dict[str, WikidataEntity] = {}
        if isinstance(entities_obj, Mapping):
            for qid, ent in entities_obj.items():
                qs = _safe_str(qid)
                if not qs or qs not in referenced_qids:
                    continue
                if isinstance(ent, Mapping):
                    entities[qs] = WikidataEntity(dict(ent))

        if _COMPACT_MAX_ENTITIES > 0 and len(entities) > _COMPACT_MAX_ENTITIES:
            # Sin timestamps por entity => cap estable por orden de key.
            keep_keys = sorted(entities.keys())[:_COMPACT_MAX_ENTITIES]
            entities = {k: entities[k] for k in keep_keys}
        cache["entities"] = entities

        if _WIKI_DEBUG_EXTRA:
            _dbg(
                "cache compacted"
                + (" (force)" if force else "")
                + f" | records={len(records)} idx_imdb={len(index_imdb)} idx_ty={len(index_ty)} "
                  f"entities={len(entities)} imdb_qid={len(imdb_qid)} is_film={len(is_film)}"
            )
        else:
            # Debug “básico” (más compacto)
            _dbg(
                f"cache compacted{' (force)' if force else ''} | "
                f"records={len(records)} entities={len(entities)}"
            )

    except Exception as exc:
        _dbg(f"cache compaction failed: {exc!r}")


# ============================================================================
# Idiomas por item
# ============================================================================

def _detect_language_chain_from_input(movie_input: MovieInputLangProto | None) -> list[str]:
    """
    Cadena priorizada de idiomas para este item:

    1) Plex library_language() si existe (p.ej. "es", "es-ES", "spa")
    2) heurísticas is_*_context()
    3) WIKI_LANGUAGE / WIKI_FALLBACK_LANGUAGE
    4) siempre incluye 'en'

    Devuelve lista deduplicada preservando orden.
    """
    chain: list[str] = []

    if movie_input is not None:
        try:
            lang = movie_input.plex_library_language()
        except Exception:
            lang = None

        if lang:
            base = _normalize_lang_code(lang)
            if base:
                chain.append(base)

        if not chain:
            try:
                if movie_input.is_spanish_context():
                    chain.append("es")
                elif movie_input.is_italian_context():
                    chain.append("it")
                elif movie_input.is_french_context():
                    chain.append("fr")
                elif movie_input.is_japanese_context():
                    chain.append("ja")
                elif movie_input.is_korean_context():
                    chain.append("ko")
                elif movie_input.is_chinese_context():
                    chain.append("zh")
                elif movie_input.is_english_context():
                    chain.append("en")
            except Exception:
                pass

    for cfg_lang in (_normalize_lang_code(WIKI_LANGUAGE), _normalize_lang_code(WIKI_FALLBACK_LANGUAGE)):
        if cfg_lang and cfg_lang not in chain:
            chain.append(cfg_lang)

    if "en" not in chain:
        chain.append("en")

    out: list[str] = []
    seen: set[str] = set()
    for l in chain:
        if l and l not in seen:
            seen.add(l)
            out.append(l)
    return out


def _best_wikipedia_languages_for_item(movie_input: MovieInputLangProto | None) -> tuple[str, str]:
    """
    (primary, fallback) según cadena de idiomas.

    - primary: primer idioma de la cadena
    - fallback: primer idioma distinto a primary (si existe), o fallback global.
    """
    chain = _detect_language_chain_from_input(movie_input)
    primary = chain[0] if chain else (_normalize_lang_code(WIKI_LANGUAGE) or "en")

    fallback = ""
    for l in chain[1:]:
        if l != primary:
            fallback = l
            break

    if not fallback:
        fallback = _normalize_lang_code(WIKI_FALLBACK_LANGUAGE) or ("en" if primary != "en" else "")

    if fallback == primary:
        fallback = "en" if primary != "en" else ""

    return primary, fallback


# ============================================================================
# Canonicalización para ranking de búsquedas
# ============================================================================

def _strip_accents(text: str) -> str:
    """Elimina acentos (para comparar títulos sin penalizar diacríticos)."""
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _canon_cmp(text: str) -> str:
    """
    Canonicaliza texto para comparación:
    - lower
    - sin acentos
    - tokenización alfanumérica
    """
    base = _strip_accents(text).lower()
    tokens = _WORD_RE.findall(base)
    return " ".join(tokens)


# ============================================================================
# Wikipedia REST + Search
# ============================================================================

def _fetch_wikipedia_summary_by_title(title: str, language: str) -> Mapping[str, object] | None:
    """
    Wikipedia REST summary:
      https://{lang}.wikipedia.org/api/rest_v1/page/summary/{Title}

    Devuelve Mapping o None si:
    - status != 200
    - disambiguation
    - JSON inválido
    """
    safe_title = quote(title.replace(" ", "_"), safe="")
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

    _dbg(f"wikipedia.summary -> lang={language} title={title!r}")

    try:
        resp = _get_session().get(url, timeout=_HTTP_TIMEOUT_SHORT)
    except Exception as exc:
        _dbg(f"wikipedia.summary EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _dbg(f"wikipedia.summary <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        _dbg(f"wikipedia.summary JSON EXC: {exc!r}")
        return None

    if not isinstance(data, Mapping):
        return None

    if _safe_str(data.get("type")) == "disambiguation":
        _dbg("wikipedia.summary -> disambiguation (skip)")
        return None

    return data


def _wikipedia_search(*, query: str, language: str, limit: int = 8) -> list[dict[str, object]]:
    """
    MediaWiki search API:
      https://{lang}.wikipedia.org/w/api.php?action=query&list=search...

    Devuelve hits con keys típicas: title, snippet, ...
    """
    params: dict[str, str] = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    url = f"https://{language}.wikipedia.org/w/api.php"

    _dbg(f"wikipedia.search -> lang={language} q={query!r}")

    try:
        resp = _get_session().get(url, params=params, timeout=_HTTP_TIMEOUT_SHORT)
    except Exception as exc:
        _dbg(f"wikipedia.search EXC: {exc!r}")
        return []

    if resp.status_code != 200:
        _dbg(f"wikipedia.search <- status={resp.status_code}")
        return []

    try:
        payload = resp.json()
    except Exception as exc:
        _dbg(f"wikipedia.search JSON EXC: {exc!r}")
        return []

    if not isinstance(payload, Mapping):
        return []

    q = payload.get("query")
    if not isinstance(q, Mapping):
        return []

    search_obj = q.get("search")
    if not isinstance(search_obj, list):
        return []

    out: list[dict[str, object]] = []
    for it in search_obj:
        if isinstance(it, Mapping):
            out.append(dict(it))
    return out


def _score_search_hit(*, hit_title: str, hit_snippet: str, wanted_title: str, year: int | None) -> float:
    """
    Score heurístico para elegir mejores resultados:

    - coincidencia canonical (sin acentos, tokens)
    - bonus si snippet sugiere film
    - bonus por año
    - penaliza desambiguación
    """
    score = 0.0

    want = _canon_cmp(wanted_title)
    got = _canon_cmp(hit_title)

    if want == got:
        score += 10.0
    elif want and got and (want in got or got in want):
        score += 6.0
    else:
        wt = set(want.split())
        gt = set(got.split())
        if wt and gt:
            score += min(5.0, len(wt & gt) * 0.8)

    sn = _canon_cmp(hit_snippet)
    if "pelicula" in sn or "película" in sn or "film" in sn or "movie" in sn or "largometraje" in sn:
        score += 2.0

    if year is not None:
        y = str(year)
        if y in hit_title:
            score += 2.0
        if y in hit_snippet:
            score += 1.0

    if "desambiguacion" in sn or "(desambiguacion" in _canon_cmp(hit_title):
        score -= 4.0

    return score


def _rank_wikipedia_candidates(*, lookup_title: str, year: int | None, language: str) -> list[str]:
    """
    Busca y rankea candidatos devolviendo títulos ordenados por relevancia.

    Estrategia:
    - queries con "film/movie/película" para empujar resultados de cine.
    - score mínimo para filtrar ruido.
    """
    clean_title = " ".join(lookup_title.strip().split())

    queries: list[str] = []
    if year is not None:
        queries.append(f"{clean_title} {year} film")
        queries.append(f"{clean_title} {year} movie")
        if language == "es":
            queries.append(f"{clean_title} {year} película")

    queries.append(f"{clean_title} film")
    queries.append(f"{clean_title} movie")
    if language == "es":
        queries.append(f"{clean_title} película")
    queries.append(clean_title)

    scored: dict[str, float] = {}
    for q in queries:
        for hit in _wikipedia_search(query=q, language=language, limit=10):
            ht = _safe_str(hit.get("title"))
            if not ht:
                continue
            snippet_raw = hit.get("snippet")
            hs = str(snippet_raw) if snippet_raw is not None else ""
            s = _score_search_hit(hit_title=ht, hit_snippet=hs, wanted_title=clean_title, year=year)
            prev = scored.get(ht)
            if prev is None or s > prev:
                scored[ht] = s

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [t for (t, s) in ranked if s >= 4.0]


def _choose_wikipedia_summary_candidates(
    *,
    title_for_lookup: str,
    year: int | None,
    primary_language: str,
    fallback_language: str,
) -> list[tuple[str, str]]:
    """
    Devuelve candidatos (title, lang) a probar para Wikipedia REST summary.

    - Primero primary_language.
    - Luego fallback_language (si existe y distinto).
    """
    out: list[tuple[str, str]] = []
    for t in _rank_wikipedia_candidates(lookup_title=title_for_lookup, year=year, language=primary_language)[:8]:
        out.append((t, primary_language))

    if fallback_language and fallback_language != primary_language:
        for t in _rank_wikipedia_candidates(lookup_title=title_for_lookup, year=year, language=fallback_language)[:8]:
            out.append((t, fallback_language))

    return out


# ============================================================================
# Wikidata entity + labels
# ============================================================================

def _fetch_wikidata_entity_json(qid: str) -> Mapping[str, object] | None:
    """
    Descarga el JSON de entidad desde Wikidata:

      https://www.wikidata.org/wiki/Special:EntityData/Qxxxx.json

    Ventajas:
    - Directo, no requiere SPARQL para claims.
    - Incluye sitelinks para elegir título en el idioma preferido.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    _dbg(f"wikidata.entity -> qid={qid}")

    try:
        resp = _get_session().get(url, timeout=_HTTP_TIMEOUT_SHORT)
    except Exception as exc:
        _dbg(f"wikidata.entity EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _dbg(f"wikidata.entity <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        _dbg(f"wikidata.entity JSON EXC: {exc!r}")
        return None

    if not isinstance(data, Mapping):
        return None

    entities = data.get("entities")
    if not isinstance(entities, Mapping):
        return None

    entity = entities.get(qid)
    if not isinstance(entity, Mapping):
        return None

    return entity


def _extract_qids_from_claims(entity: Mapping[str, object], property_id: str) -> list[str]:
    """
    Extrae QIDs de una propiedad Wikidata (claims).

    property_id ejemplos:
      - P31: instance of
      - P57: director
      - P495: country of origin
      - P136: genre
    """
    claims_obj = entity.get("claims")
    if not isinstance(claims_obj, Mapping):
        return []

    prop_claims = claims_obj.get(property_id)
    if not isinstance(prop_claims, list):
        return []

    qids: list[str] = []
    for claim in prop_claims:
        if not isinstance(claim, Mapping):
            continue
        mainsnak = claim.get("mainsnak")
        if not isinstance(mainsnak, Mapping):
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, Mapping):
            continue
        value = datavalue.get("value")
        if not isinstance(value, Mapping):
            continue
        qid = _safe_str(value.get("id"))
        if qid and qid not in qids:
            qids.append(qid)
    return qids


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    """Generador simple de chunks."""
    step = max(1, size)
    for i in range(0, len(values), step):
        yield values[i : i + step]


def _fetch_wikidata_labels(qids: list[str], language: str, fallback_language: str) -> dict[str, WikidataEntity]:
    """
    Obtiene labels/descriptions (wbgetentities) para un conjunto de QIDs.

    - Batch de 50 para evitar URLs enormes.
    - Pide labels/descriptions en (language|fallback_language).
    """
    if not qids:
        return {}

    out: dict[str, WikidataEntity] = {}
    languages = f"{language}|{fallback_language}" if fallback_language and fallback_language != language else language

    for batch in _chunked(qids, 50):
        ids = "|".join(batch)
        params: dict[str, str] = {
            "action": "wbgetentities",
            "ids": ids,
            "props": "labels|descriptions",
            "languages": languages,
            "format": "json",
        }

        try:
            resp = _get_session().get("https://www.wikidata.org/w/api.php", params=params, timeout=_HTTP_TIMEOUT_SHORT)
        except Exception as exc:
            _dbg(f"wikidata.labels EXC: {exc!r}")
            continue

        if resp.status_code != 200:
            _dbg(f"wikidata.labels <- status={resp.status_code}")
            continue

        try:
            payload = resp.json()
        except Exception as exc:
            _dbg(f"wikidata.labels JSON EXC: {exc!r}")
            continue

        if not isinstance(payload, Mapping):
            continue

        entities_obj = payload.get("entities")
        if not isinstance(entities_obj, Mapping):
            continue

        for qid in batch:
            ent = entities_obj.get(qid)
            if not isinstance(ent, Mapping):
                continue

            label = qid
            labels_obj = ent.get("labels")
            if isinstance(labels_obj, Mapping):
                lp = labels_obj.get(language)
                lf = labels_obj.get(fallback_language) if fallback_language else None
                if isinstance(lp, Mapping):
                    label = str(lp.get("value") or qid)
                elif isinstance(lf, Mapping):
                    label = str(lf.get("value") or qid)

            desc: str | None = None
            descriptions_obj = ent.get("descriptions")
            if isinstance(descriptions_obj, Mapping):
                dp = descriptions_obj.get(language)
                df = descriptions_obj.get(fallback_language) if fallback_language else None
                if isinstance(dp, Mapping):
                    desc = _safe_str(dp.get("value"))
                elif isinstance(df, Mapping):
                    desc = _safe_str(df.get("value"))

            out[qid] = WikidataEntity(label=label, description=desc)

    return out


# ============================================================================
# SPARQL (con throttle global)
# ============================================================================

def _sparql_throttle() -> None:
    """
    Throttle mínimo entre llamadas SPARQL (global para el proceso).

    Importante:
    - Protegido por lock para evitar ráfagas concurrentes en ThreadPool.
    - WDQS penaliza el abuso; esto evita “storm” cuando hay muchos misses.
    """
    global _LAST_SPARQL_TS
    with _SPARQL_THROTTLE_LOCK:
        now = time.time()
        delta = now - _LAST_SPARQL_TS
        if delta < _SPARQL_MIN_INTERVAL_S:
            time.sleep(_SPARQL_MIN_INTERVAL_S - delta)
        _LAST_SPARQL_TS = time.time()


def _wikidata_sparql(query: str) -> Mapping[str, object] | None:
    """
    Ejecuta SPARQL en WDQS.
    Timeout más alto porque WDQS puede tardar.

    Nota:
    - Best-effort: si hay fallo, devolvemos None (y NO cacheamos como negativo).
    """
    url = "https://query.wikidata.org/sparql"
    params = {"format": "json", "query": query}

    _sparql_throttle()
    _dbg(f"wikidata.sparql -> len={len(query)}")

    try:
        resp = _get_session().get(url, params=params, timeout=_HTTP_TIMEOUT_SPARQL)
    except Exception as exc:
        _dbg(f"wikidata.sparql EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _dbg(f"wikidata.sparql <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        _dbg(f"wikidata.sparql JSON EXC: {exc!r}")
        return None

    return data if isinstance(data, Mapping) else None


# ============================================================================
# “is film” heurística (cacheada)
# ============================================================================

def _looks_like_film_from_wikipedia(wiki_raw: Mapping[str, object], language: str) -> bool:
    """
    Heurística conservadora basada en `description` del summary.
    """
    desc = str(wiki_raw.get("description") or "").strip().lower()
    if not desc:
        return False

    hints = (
        "película",
        "largometraje",
        "film",
        "movie",
        "feature film",
        "animated film",
        "television film",
        "motion picture",
        "映画",
        "영화",
        "电影",
        "電影",
    )
    if any(h.lower() in desc for h in hints):
        return True

    # refuerzos por idioma (redundantes pero baratos)
    if language == "ja" and "映画" in desc:
        return True
    if language == "ko" and "영화" in desc:
        return True
    if language == "zh" and ("电影" in desc or "電影" in desc):
        return True

    return False


def _is_film_without_sparql(
    *,
    wd_entity: Mapping[str, object],
    wiki_raw: Mapping[str, object] | None,
    wiki_lang: str,
) -> bool:
    """
    Decide “is film” SIN SPARQL:
      1) P31 ∈ allowlist (film/feature/animated/tv/short)
      2) fallback a description de Wikipedia (si existe)

    Diseñado para ser barato y no depender de WDQS en el path principal.
    """
    p31 = set(_extract_qids_from_claims(wd_entity, "P31"))
    if p31 & _FILM_INSTANCE_ALLOWLIST:
        return True

    if wiki_raw is not None and _looks_like_film_from_wikipedia(wiki_raw, wiki_lang):
        return True

    return False


def _is_film_cached(
    *,
    cache: WikiCacheFile,
    qid: str,
    wd_entity: Mapping[str, object],
    wiki_raw: Mapping[str, object] | None,
    wiki_lang: str,
) -> bool:
    """
    Cachea (qid -> is_film) con TTL.

    Importante:
    - Este cache se usa tanto para casos OK como negativos.
    - Guardar false ayuda a evitar volver a construir negativos repetidamente.
    """
    now_epoch = _now_epoch()

    cached = cache["is_film"].get(qid)
    if cached is not None and not _is_expired(int(cached["fetched_at"]), int(cached["ttl_s"]), now_epoch):
        return bool(cached.get("is_film") is True)

    ok = _is_film_without_sparql(wd_entity=wd_entity, wiki_raw=wiki_raw, wiki_lang=wiki_lang)
    cache["is_film"][qid] = IsFilmCacheEntry(is_film=bool(ok), fetched_at=now_epoch, ttl_s=int(_TTL_IS_FILM_S))
    _mark_dirty_unlocked()
    _maybe_flush_unlocked(force=False)
    return ok


# ============================================================================
# imdbID -> QID (P345) + negative caching
# ============================================================================

def _fetch_qid_by_imdb_id(cache: WikiCacheFile, imdb_id: str) -> str | None:
    """
    Resuelve imdbID -> QID (P345) con SPARQL.

    - Usa cache vigente si existe.
    - Cachea None (negative) con TTL específico para evitar martilleo.

    Nota “best-effort”:
    - Si WDQS falla (None), NO cacheamos negativo (podría ser transitorio).
    """
    imdb_norm = _norm_imdb(imdb_id)
    if not imdb_norm:
        return None

    now_epoch = _now_epoch()
    cached = cache["imdb_qid"].get(imdb_norm)
    if cached is not None and not _is_expired(int(cached["fetched_at"]), int(cached["ttl_s"]), now_epoch):
        return cached.get("qid")

    query = f"""
SELECT ?item WHERE {{
  ?item wdt:P345 "{imdb_norm}" .
}}
LIMIT 2
""".strip()

    data = _wikidata_sparql(query)
    if not data:
        return None

    results = data.get("results")
    if not isinstance(results, Mapping):
        return None
    bindings = results.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        cache["imdb_qid"][imdb_norm] = ImdbQidCacheEntry(
            qid=None,
            fetched_at=now_epoch,
            ttl_s=int(_TTL_IMDB_QID_NEGATIVE_S),
        )
        _mark_dirty_unlocked()
        _maybe_flush_unlocked(force=False)
        return None

    first = bindings[0]
    if not isinstance(first, Mapping):
        return None
    item = first.get("item")
    if not isinstance(item, Mapping):
        return None
    val = _safe_str(item.get("value"))
    if not val:
        return None

    m = re.search(r"/entity/(Q\d+)$", val)
    qid = m.group(1) if m else None
    if not qid:
        return None

    cache["imdb_qid"][imdb_norm] = ImdbQidCacheEntry(qid=qid, fetched_at=now_epoch, ttl_s=int(_TTL_OK_S))
    _mark_dirty_unlocked()
    _maybe_flush_unlocked(force=False)
    return qid


def _extract_sitelink_title(entity: Mapping[str, object], language: str) -> str | None:
    """Extrae el título del sitelink {language}wiki (si existe)."""
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, Mapping):
        return None
    sl = sitelinks.get(f"{language}wiki")
    if not isinstance(sl, Mapping):
        return None
    return _safe_str(sl.get("title"))


def _pick_best_sitelink_title(entity: Mapping[str, object], languages: list[str]) -> tuple[str | None, str]:
    """
    Elige el mejor sitelink_title según una lista ordenada de idiomas.

    Returns:
      - (title_or_none, lang_used)
    """
    for lang in languages:
        title = _extract_sitelink_title(entity, lang)
        if title:
            return title, lang
    return None, (languages[0] if languages else (_normalize_lang_code(WIKI_LANGUAGE) or "en"))


# ============================================================================
# Cache lookup/store (schema v6)
# ============================================================================

def _get_cached_item(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
) -> WikiCacheItem | None:
    """
    Lookup en cache (incluye items negativos).

    Estrategia:
    1) Si hay imdb_id -> index_imdb -> rid -> record
    2) Si no (o no está) -> index_ty -> rid -> record

    Nota:
    - Aunque indexes apunten a records, puede haber inconsistencias por bugs o cortes antiguos.
      Compaction debería repararlo, pero aquí también validamos de forma defensiva.
    """
    now_epoch = _now_epoch()

    records = cache["records"]
    idx_imdb = cache["index_imdb"]
    idx_ty = cache["index_ty"]

    if imdb_id:
        rid = idx_imdb.get(imdb_id)
        if isinstance(rid, str):
            it = records.get(rid)
            if it is not None and not _is_expired(int(it["fetched_at"]), int(it["ttl_s"]), now_epoch):
                return it

    rid2 = idx_ty.get(_ty_key(norm_title, norm_year))
    if isinstance(rid2, str):
        it2 = records.get(rid2)
        if it2 is not None and not _is_expired(int(it2["fetched_at"]), int(it2["ttl_s"]), now_epoch):
            return it2

    return None


def _rid_for_item(item: WikiCacheItem) -> str:
    """
    RID estable:
    - si imdbID => "imdb:<id>"
    - else => "ty:<title>|<year>"

    Esto permite:
    - deduplicación natural: todo lo que tenga imdb cae en el mismo rid.
    - estabilidad de key en disco (útil para compaction).
    """
    imdb = _norm_imdb(item.get("imdbID") if isinstance(item.get("imdbID"), str) else None)
    if imdb:
        return f"imdb:{imdb}"

    title = item.get("Title")
    year = item.get("Year")
    t = title if isinstance(title, str) else ""
    y = year if isinstance(year, str) else ""
    return f"ty:{t}|{y}"


def _store_item_unlocked(cache: WikiCacheFile, item: WikiCacheItem) -> None:
    """
    Inserta/actualiza item en schema v6:
      - records[rid] = item
      - index_ty[title|year] = rid
      - index_imdb[imdb] = rid (si existe)

    Se llama tanto en OK como en negativos.
    """
    rid = _rid_for_item(item)
    cache["records"][rid] = item

    title = item.get("Title")
    year = item.get("Year")
    if isinstance(title, str) and isinstance(year, str):
        cache["index_ty"][_ty_key(title, year)] = rid

    imdb = _norm_imdb(item.get("imdbID") if isinstance(item.get("imdbID"), str) else None)
    if imdb:
        cache["index_imdb"][imdb] = rid

    _mark_dirty_unlocked()
    _maybe_flush_unlocked(force=False)


# ============================================================================
# Builders
# ============================================================================

def _build_negative_item(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    status: WikiItemStatus,
    wikibase_item: str | None,
    primary_language: str,
    fallback_language: str,
) -> WikiCacheItem:
    """
    Construye un item negativo cacheable (NEGATIVE CACHING REAL).

    - status explica el motivo:
        - imdb_no_qid: imdb_id no resolvió a QID
        - no_qid: no se pudo inferir QID por búsqueda
        - not_film: QID existe pero heurística dice “no film”
    - TTL: _TTL_NEGATIVE_S (configurable)

    Guardamos 'wikibase_item' si lo tenemos (QID) para debug/provenance,
    incluso si decidimos not_film.
    """
    now_epoch = _now_epoch()

    wiki_block: WikiBlock = WikiBlock(
        language=primary_language,
        fallback_language=fallback_language,
        source_language="",
        wikipedia_title=None,
        wikipedia_pageid=None,
        wikibase_item=wikibase_item,
        summary="",
        description="",
        urls={},
    )

    wikidata_block: WikidataBlock = WikidataBlock()
    if wikibase_item:
        wikidata_block["qid"] = wikibase_item

    return WikiCacheItem(
        Title=norm_title,
        Year=norm_year,
        imdbID=imdb_id,
        wiki=wiki_block,
        wikidata=wikidata_block,
        fetched_at=now_epoch,
        ttl_s=int(_TTL_NEGATIVE_S),
        status=status,
    )


def _build_ok_item_and_merge_entities(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    wiki_raw: Mapping[str, object],
    source_language: str,
    wikibase_item: str,
    wd_entity: Mapping[str, object],
    primary_language: str,
    fallback_language: str,
) -> WikiCacheItem:
    """
    Construye un item OK:
    - Normaliza summary de Wikipedia
    - Extrae claims básicos de Wikidata (director, país, género)
    - Hidrata labels/descriptions y los mergea en cache["entities"]

    Nota:
    - Este item puede ser “grande” (summary, images, urls).
      Para compactación adicional, un siguiente paso sería persistir solo IDs
      y mantener el payload completo en memoria. Por ahora respetamos diseño.
    """
    titles_obj = wiki_raw.get("titles")
    wikipedia_title: str | None = None
    if isinstance(titles_obj, Mapping):
        wikipedia_title = _safe_str(titles_obj.get("normalized")) or _safe_str(titles_obj.get("canonical"))

    wiki_block: WikiBlock = WikiBlock(
        language=primary_language,
        fallback_language=fallback_language,
        source_language=source_language,
        wikipedia_title=wikipedia_title,
        wikipedia_pageid=_safe_int(wiki_raw.get("pageid")),
        wikibase_item=wikibase_item,
        summary=str(wiki_raw.get("extract") or ""),
        description=str(wiki_raw.get("description") or ""),
        urls=dict(wiki_raw.get("content_urls")) if isinstance(wiki_raw.get("content_urls"), Mapping) else {},
    )

    if "originalimage" in wiki_raw or "thumbnail" in wiki_raw:
        images: dict[str, object] = {}
        original = wiki_raw.get("originalimage")
        thumb = wiki_raw.get("thumbnail")
        if isinstance(original, Mapping):
            images["original"] = dict(original)
        if isinstance(thumb, Mapping):
            images["thumbnail"] = dict(thumb)
        wiki_block["images"] = images

    wikidata_block: WikidataBlock = WikidataBlock(qid=wikibase_item)

    directors = _extract_qids_from_claims(wd_entity, "P57")   # director
    countries = _extract_qids_from_claims(wd_entity, "P495")  # country of origin
    genres = _extract_qids_from_claims(wd_entity, "P136")     # genre

    if directors:
        wikidata_block["directors"] = directors
    if countries:
        wikidata_block["countries"] = countries
    if genres:
        wikidata_block["genres"] = genres

    # Labels para QIDs relevantes (batch)
    to_label = list({*directors, *countries, *genres})
    labeled = _fetch_wikidata_labels(to_label, primary_language, fallback_language)

    for qid, ent in labeled.items():
        etype: str | None = None
        if qid in directors:
            etype = "person"
        elif qid in countries:
            etype = "country"
        elif qid in genres:
            etype = "genre"

        merged = WikidataEntity(ent)
        if etype:
            merged["type"] = etype
        cache["entities"][qid] = merged

    item = WikiCacheItem(
        Title=norm_title,
        Year=norm_year,
        imdbID=imdb_id,
        wiki=wiki_block,
        wikidata=wikidata_block,
        fetched_at=_now_epoch(),
        ttl_s=int(_TTL_OK_S),
        status="ok",
    )

    if not logger.is_silent_mode():
        year_label = norm_year if norm_year else "?"
        _info(f"[WIKI] cached ({source_language}): {norm_title} ({year_label})")

    return item


# ============================================================================
# Core único (una sola implementación interna)
# ============================================================================

def _get_wiki_impl(
    *,
    title: str,
    year: int | None,
    imdb_id: str | None,
    movie_input: MovieInputLangProto | None,
    primary: str,
    fallback: str,
) -> WikiCacheItem | None:
    """
    Única implementación:

    0) canonicaliza título
    1) cache lookup (imdb->record o ty->record)
    2) si imdb_id:
         - imdb(P345) -> QID (SPARQL) con negative caching
         - sitelink según chain de idiomas
         - Wikipedia summary
         - is_film (cacheado) -> OK/NEGATIVO
       si no imdb_id:
         - búsqueda Wikipedia -> summary -> wikibase_item -> is_film -> OK/NEGATIVO

    Best-effort:
    - Fallos de red => None, NO se cachean como negativos.
    """
    lookup_title = normalize_title_for_lookup(title)
    if not lookup_title:
        _dbg(f"empty lookup_title from title={title!r}")
        return None

    norm_title = lookup_title
    norm_year = str(year) if year is not None else ""
    imdb_norm = _norm_imdb(imdb_id)

    # 1) cache HIT (incluye negativos)
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        cached = _get_cached_item(cache=cache, norm_title=norm_title, norm_year=norm_year, imdb_id=imdb_norm)
        if cached is not None:
            _dbg("cache HIT")
            return cached

    # 2) camino imdb -> qid (más preciso)
    if imdb_norm:
        with _CACHE_LOCK:
            cache = _load_cache_unlocked()
            qid = _fetch_qid_by_imdb_id(cache, imdb_norm)

        if not qid:
            # Negative caching: imdb_no_qid
            neg = _build_negative_item(
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
                status="imdb_no_qid",
                wikibase_item=None,
                primary_language=primary,
                fallback_language=fallback,
            )
            with _CACHE_LOCK:
                cache = _load_cache_unlocked()
                _store_item_unlocked(cache, neg)
            return neg

        wd_entity = _fetch_wikidata_entity_json(qid)
        if wd_entity is None:
            return None

        chain = _detect_language_chain_from_input(movie_input)
        sl_title, sl_lang = _pick_best_sitelink_title(wd_entity, chain)
        wiki_raw = _fetch_wikipedia_summary_by_title(sl_title, sl_lang) if sl_title else None

        with _CACHE_LOCK:
            cache = _load_cache_unlocked()

            # Si determinamos “no film”, se cachea NEGATIVO.
            if not _is_film_cached(cache=cache, qid=qid, wd_entity=wd_entity, wiki_raw=wiki_raw, wiki_lang=sl_lang):
                neg_nf = _build_negative_item(
                    norm_title=norm_title,
                    norm_year=norm_year,
                    imdb_id=imdb_norm,
                    status="not_film",
                    wikibase_item=qid,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                _store_item_unlocked(cache, neg_nf)
                return neg_nf

            # Si no hay summary usable: best-effort -> NO cacheamos item ok incompleto
            if wiki_raw is None:
                return None

            ok_item = _build_ok_item_and_merge_entities(
                cache=cache,
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
                wiki_raw=wiki_raw,
                source_language=sl_lang,
                wikibase_item=qid,
                wd_entity=wd_entity,
                primary_language=primary,
                fallback_language=fallback,
            )
            _store_item_unlocked(cache, ok_item)
            return ok_item

    # 3) camino sin imdb: búsqueda Wikipedia
    candidates = _choose_wikipedia_summary_candidates(
        title_for_lookup=lookup_title,
        year=year,
        primary_language=primary,
        fallback_language=fallback,
    )
    _dbg(f"candidates={len(candidates)} primary={primary} fallback={fallback}")

    for cand_title, cand_lang in candidates:
        wiki_raw = _fetch_wikipedia_summary_by_title(cand_title, cand_lang)
        if wiki_raw is None:
            continue

        wikibase_item = _safe_str(wiki_raw.get("wikibase_item"))
        if not wikibase_item:
            continue

        wd_entity = _fetch_wikidata_entity_json(wikibase_item)
        if wd_entity is None:
            continue

        with _CACHE_LOCK:
            cache = _load_cache_unlocked()

            if not _is_film_cached(
                cache=cache,
                qid=wikibase_item,
                wd_entity=wd_entity,
                wiki_raw=wiki_raw,
                wiki_lang=cand_lang,
            ):
                neg_nf2 = _build_negative_item(
                    norm_title=norm_title,
                    norm_year=norm_year,
                    imdb_id=None,
                    status="not_film",
                    wikibase_item=wikibase_item,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                _store_item_unlocked(cache, neg_nf2)
                continue

            ok_item2 = _build_ok_item_and_merge_entities(
                cache=cache,
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=None,
                wiki_raw=wiki_raw,
                source_language=cand_lang,
                wikibase_item=wikibase_item,
                wd_entity=wd_entity,
                primary_language=primary,
                fallback_language=fallback,
            )
            _store_item_unlocked(cache, ok_item2)
            return ok_item2

    # 4) negativo final: no_qid (NEGATIVE CACHING REAL)
    neg = _build_negative_item(
        norm_title=norm_title,
        norm_year=norm_year,
        imdb_id=None,
        status="no_qid",
        wikibase_item=None,
        primary_language=primary,
        fallback_language=fallback,
    )
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        _store_item_unlocked(cache, neg)
    return neg


# ============================================================================
# API pública (sin wrappers OO)
# ============================================================================

def get_wiki(*, title: str, year: int | None, imdb_id: str | None) -> WikiCacheItem | None:
    """
    API clásica (sin MovieInput):
    - Usa idiomas globales (WIKI_LANGUAGE/WIKI_FALLBACK_LANGUAGE).
    """
    primary = _normalize_lang_code(WIKI_LANGUAGE) or "en"
    fallback = _normalize_lang_code(WIKI_FALLBACK_LANGUAGE) or ("en" if primary != "en" else "")
    return _get_wiki_impl(title=title, year=year, imdb_id=imdb_id, movie_input=None, primary=primary, fallback=fallback)


def get_wiki_for_input(
    *,
    movie_input: MovieInputLangProto,
    title: str,
    year: int | None,
    imdb_id: str | None,
) -> WikiCacheItem | None:
    """
    API recomendada:
    - Idioma por item (library_language + heurística).
    """
    primary, fallback = _best_wikipedia_languages_for_item(movie_input)
    return _get_wiki_impl(
        title=title,
        year=year,
        imdb_id=imdb_id,
        movie_input=movie_input,
        primary=primary,
        fallback=fallback,
    )