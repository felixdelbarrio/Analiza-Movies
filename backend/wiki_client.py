from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, TypedDict
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend import logger as _logger
from backend.config import (
    DATA_DIR,
    SILENT_MODE,
    WIKI_FALLBACK_LANGUAGE,
    WIKI_LANGUAGE,
    WIKI_DEBUG,
)
from backend.movie_input import normalize_title_for_lookup


class WikidataEntity(TypedDict, total=False):
    label: str
    description: str | None
    type: str


class WikiBlock(TypedDict, total=False):
    language: str
    fallback_language: str
    source_language: str
    wikipedia_title: str | None
    wikipedia_pageid: int | None
    wikibase_item: str | None
    summary: str
    description: str
    urls: dict[str, object]
    images: dict[str, object]


class WikidataBlock(TypedDict, total=False):
    qid: str
    directors: list[str]
    countries: list[str]
    genres: list[str]


class WikiCacheItem(TypedDict):
    Title: str
    Year: str
    imdbID: str | None
    wiki: WikiBlock
    wikidata: WikidataBlock


class WikiCacheFile(TypedDict):
    schema: int
    language: str
    fallback_language: str
    items: list[WikiCacheItem]
    entities: dict[str, WikidataEntity]
    # NUEVO: cachea resoluciones/validaciones para reducir SPARQL
    imdb_qid: dict[str, str]
    is_film: dict[str, bool]


_SCHEMA_VERSION: Final[int] = 4  # <- bump por nuevos campos
_CACHE_PATH: Final[Path] = DATA_DIR / "wiki_cache.json"

_SESSION: requests.Session | None = None

_WIKI_DEBUG_ENV: Final[bool] = os.getenv("ANALIZA_WIKI_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

# -------------------------
# SPARQL throttle
# -------------------------
_LAST_SPARQL_TS: float = 0.0
_SPARQL_MIN_INTERVAL_S: Final[float] = float(os.getenv("ANALIZA_WIKI_SPARQL_INTERVAL", "0.20") or "0.20")

# -------------------------
# Film detection (no SPARQL)
# -------------------------
# P31 allowlist (instance of) para aceptar “es una película” sin SPARQL.
# Puedes ampliarlo si detectas falsos negativos.
_FILM_INSTANCE_ALLOWLIST: Final[set[str]] = {
    "Q11424",  # film
    "Q24862",  # feature film
    "Q202866",  # animated film
    "Q226730",  # short film
    "Q506240",  # television film
    "Q93204",  # documentary film (ojo: a veces)
}


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    session = requests.Session()

    # Headers recomendados (Wikipedia/Wikidata pueden bloquear sin User-Agent “humano”)
    session.headers.update(
        {
            "User-Agent": "Analiza-Movies/1.0 (local; contact: your-email-or-site)",
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


def _log(msg: str) -> None:
    try:
        _logger.info(msg)
    except Exception:
        if not SILENT_MODE:
            print(msg)


def _dbg(msg: str) -> None:
    if not (WIKI_DEBUG or _WIKI_DEBUG_ENV):
        return
    _log(msg)


def _safe_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v or None


def _safe_int(value: object) -> int | None:
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


def _empty_cache() -> WikiCacheFile:
    return {
        "schema": _SCHEMA_VERSION,
        "language": WIKI_LANGUAGE,
        "fallback_language": WIKI_FALLBACK_LANGUAGE,
        "items": [],
        "entities": {},
        "imdb_qid": {},
        "is_film": {},
    }


def _save_cache(cache: WikiCacheFile) -> None:
    dirpath = _CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(dirpath),
    ) as tf:
        json.dump(cache, tf, ensure_ascii=False, indent=2)
        temp_name = tf.name

    Path(temp_name).replace(_CACHE_PATH)


def _load_cache() -> WikiCacheFile:
    if not _CACHE_PATH.exists():
        cache = _empty_cache()
        _save_cache(cache)
        _dbg(f"[WIKI-DEBUG] cache file created: {_CACHE_PATH}")
        return cache

    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] cache read failed ({exc}); recreating empty cache")
        cache = _empty_cache()
        _save_cache(cache)
        return cache

    if not isinstance(raw, Mapping):
        cache = _empty_cache()
        _save_cache(cache)
        return cache

    if raw.get("schema") != _SCHEMA_VERSION:
        _dbg(f"[WIKI-DEBUG] cache schema mismatch -> recreate (found={raw.get('schema')})")
        cache = _empty_cache()
        _save_cache(cache)
        return cache

    items_obj = raw.get("items")
    entities_obj = raw.get("entities")
    imdb_qid_obj = raw.get("imdb_qid")
    is_film_obj = raw.get("is_film")

    if (
        not isinstance(items_obj, list)
        or not isinstance(entities_obj, Mapping)
        or not isinstance(imdb_qid_obj, Mapping)
        or not isinstance(is_film_obj, Mapping)
    ):
        cache = _empty_cache()
        _save_cache(cache)
        return cache

    # normaliza tipos (por si vienen raros)
    imdb_qid: dict[str, str] = {}
    for k, v in imdb_qid_obj.items():
        ks = _safe_str(k)
        vs = _safe_str(v)
        if ks and vs:
            imdb_qid[ks] = vs

    is_film: dict[str, bool] = {}
    for k, v in is_film_obj.items():
        ks = _safe_str(k)
        if not ks:
            continue
        is_film[ks] = bool(v is True)

    return {
        "schema": _SCHEMA_VERSION,
        "language": str(raw.get("language") or WIKI_LANGUAGE),
        "fallback_language": str(raw.get("fallback_language") or WIKI_FALLBACK_LANGUAGE),
        "items": list(items_obj),  # type: ignore[list-item]
        "entities": dict(entities_obj),  # type: ignore[dict-item]
        "imdb_qid": imdb_qid,
        "is_film": is_film,
    }


def _find_existing(
    items: list[WikiCacheItem],
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
) -> WikiCacheItem | None:
    if imdb_id:
        for item in items:
            if item.get("imdbID") == imdb_id:
                return item

    for item in items:
        if (
            item.get("imdbID") is None
            and item.get("Title") == norm_title
            and item.get("Year") == norm_year
        ):
            return item

    return None


_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _canon_cmp(text: str) -> str:
    base = _strip_accents(text).lower()
    tokens = _WORD_RE.findall(base)
    return " ".join(tokens)


def _fetch_wikipedia_summary_by_title(title: str, language: str) -> Mapping[str, object] | None:
    safe_title = quote(title.replace(" ", "_"), safe="")
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

    _dbg(f"[WIKI-DEBUG] wikipedia.summary -> GET {url}")

    try:
        session = _get_session()
        resp = session.get(url, timeout=10)
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikipedia.summary EXC: {exc}")
        return None

    _dbg(f"[WIKI-DEBUG] wikipedia.summary <- status={resp.status_code}")

    if resp.status_code != 200:
        if resp.status_code == 403:
            _dbg(f"[WIKI-DEBUG] wikipedia.summary 403 body={resp.text[:300]!r}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikipedia.summary JSON EXC: {exc}")
        return None

    if not isinstance(data, Mapping):
        _dbg("[WIKI-DEBUG] wikipedia.summary JSON not Mapping")
        return None

    # Rechaza desambiguaciones explícitas
    if _safe_str(data.get("type")) == "disambiguation":
        _dbg("[WIKI-DEBUG] wikipedia.summary -> disambiguation (skip)")
        return None

    return data


def _wikipedia_search(*, query: str, language: str, limit: int = 6) -> list[dict[str, object]]:
    params: dict[str, str] = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }

    url = f"https://{language}.wikipedia.org/w/api.php"
    _dbg(f"[WIKI-DEBUG] wikipedia.search -> GET {url} params={params!r}")

    try:
        session = _get_session()
        resp = session.get(url, params=params, timeout=10)
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikipedia.search EXC: {exc}")
        return []

    _dbg(f"[WIKI-DEBUG] wikipedia.search <- status={resp.status_code}")

    if resp.status_code != 200:
        if resp.status_code == 403:
            _dbg(f"[WIKI-DEBUG] wikipedia.search 403 body={resp.text[:300]!r}")
        return []

    try:
        payload = resp.json()
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikipedia.search JSON EXC: {exc}")
        return []

    if not isinstance(payload, Mapping):
        _dbg("[WIKI-DEBUG] wikipedia.search JSON not Mapping")
        return []

    q = payload.get("query")
    if not isinstance(q, Mapping):
        _dbg("[WIKI-DEBUG] wikipedia.search payload.query not Mapping")
        return []

    search_obj = q.get("search")
    if not isinstance(search_obj, list):
        _dbg("[WIKI-DEBUG] wikipedia.search payload.query.search not list")
        return []

    out: list[dict[str, object]] = []
    for it in search_obj:
        if isinstance(it, Mapping):
            out.append(dict(it))

    return out


def _score_search_hit(*, hit_title: str, hit_snippet: str, wanted_title: str, year: int | None) -> float:
    score = 0.0

    want = _canon_cmp(wanted_title)
    got = _canon_cmp(hit_title)

    if want == got:
        score += 10.0
    elif want and got and (want in got or got in want):
        score += 6.0
    else:
        want_tokens = set(want.split())
        got_tokens = set(got.split())
        if want_tokens and got_tokens:
            inter = len(want_tokens & got_tokens)
            score += min(5.0, inter * 0.8)

    sn = _canon_cmp(hit_snippet)
    if "pelicula" in sn or "film" in sn or "largometraje" in sn or "movie" in sn:
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
    queries: list[str] = []
    clean_title = " ".join(lookup_title.strip().split())

    if language.lower().startswith("es"):
        if year is not None:
            queries += [f"{clean_title} {year} película", f"{clean_title} {year} film"]
        queries += [f"{clean_title} película", f"{clean_title} film", clean_title]
    else:
        if year is not None:
            queries += [f"{clean_title} {year} film", f"{clean_title} {year} movie"]
        queries += [f"{clean_title} film", f"{clean_title} movie", clean_title]

    scored: dict[str, float] = {}

    for q in queries:
        hits = _wikipedia_search(query=q, language=language, limit=10)
        for hit in hits:
            hit_title = _safe_str(hit.get("title")) or ""
            if not hit_title:
                continue
            snippet_raw = hit.get("snippet")
            hit_snippet = str(snippet_raw) if snippet_raw is not None else ""
            s = _score_search_hit(
                hit_title=hit_title,
                hit_snippet=hit_snippet,
                wanted_title=clean_title,
                year=year,
            )
            prev = scored.get(hit_title)
            if prev is None or s > prev:
                scored[hit_title] = s

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    out = [t for (t, s) in ranked if s >= 4.0]
    _dbg(f"[WIKI-DEBUG] ranked_candidates({language}) -> {out[:10]!r}")
    return out


def _choose_wikipedia_summary_candidates(title_for_lookup: str, year: int | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    for t in _rank_wikipedia_candidates(lookup_title=title_for_lookup, year=year, language=WIKI_LANGUAGE)[:8]:
        out.append((t, WIKI_LANGUAGE))

    if WIKI_FALLBACK_LANGUAGE and WIKI_FALLBACK_LANGUAGE != WIKI_LANGUAGE:
        for t in _rank_wikipedia_candidates(
            lookup_title=title_for_lookup, year=year, language=WIKI_FALLBACK_LANGUAGE
        )[:8]:
            out.append((t, WIKI_FALLBACK_LANGUAGE))

    return out


# -------------------------
# Wikidata helpers
# -------------------------

def _fetch_wikidata_entity_json(qid: str) -> Mapping[str, object] | None:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    _dbg(f"[WIKI-DEBUG] wikidata.entity -> GET {url}")

    try:
        session = _get_session()
        resp = session.get(url, timeout=10)
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikidata.entity EXC: {exc}")
        return None

    _dbg(f"[WIKI-DEBUG] wikidata.entity <- status={resp.status_code}")

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikidata.entity JSON EXC: {exc}")
        return None

    if not isinstance(data, Mapping):
        _dbg("[WIKI-DEBUG] wikidata.entity JSON not Mapping")
        return None

    entities = data.get("entities")
    if not isinstance(entities, Mapping):
        _dbg("[WIKI-DEBUG] wikidata.entity payload.entities not Mapping")
        return None

    entity = entities.get(qid)
    if not isinstance(entity, Mapping):
        _dbg("[WIKI-DEBUG] wikidata.entity payload.entities[qid] not Mapping")
        return None

    return entity


def _extract_qids_from_claims(entity: Mapping[str, object], property_id: str) -> list[str]:
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
        if not qid:
            continue
        if qid not in qids:
            qids.append(qid)

    return qids


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        yield values
        return
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _fetch_wikidata_labels(qids: list[str], language: str, fallback_language: str) -> dict[str, WikidataEntity]:
    if not qids:
        return {}

    out: dict[str, WikidataEntity] = {}
    languages = (
        f"{language}|{fallback_language}"
        if fallback_language and fallback_language != language
        else language
    )

    for batch in _chunked(qids, 50):
        ids = "|".join(batch)
        params: dict[str, str] = {
            "action": "wbgetentities",
            "ids": ids,
            "props": "labels|descriptions",
            "languages": languages,
            "format": "json",
        }

        _dbg(f"[WIKI-DEBUG] wikidata.labels -> GET w/api.php params={params!r}")

        try:
            session = _get_session()
            resp = session.get("https://www.wikidata.org/w/api.php", params=params, timeout=10)
        except Exception as exc:
            _dbg(f"[WIKI-DEBUG] wikidata.labels EXC: {exc}")
            continue

        _dbg(f"[WIKI-DEBUG] wikidata.labels <- status={resp.status_code}")

        if resp.status_code != 200:
            continue

        try:
            payload = resp.json()
        except Exception as exc:
            _dbg(f"[WIKI-DEBUG] wikidata.labels JSON EXC: {exc}")
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
                lbl_primary = labels_obj.get(language)
                lbl_fallback = labels_obj.get(fallback_language) if fallback_language else None
                if isinstance(lbl_primary, Mapping):
                    label = str(lbl_primary.get("value") or qid)
                elif isinstance(lbl_fallback, Mapping):
                    label = str(lbl_fallback.get("value") or qid)

            desc: str | None = None
            descriptions_obj = ent.get("descriptions")
            if isinstance(descriptions_obj, Mapping):
                d_primary = descriptions_obj.get(language)
                d_fallback = descriptions_obj.get(fallback_language) if fallback_language else None
                if isinstance(d_primary, Mapping):
                    desc = _safe_str(d_primary.get("value"))
                elif isinstance(d_fallback, Mapping):
                    desc = _safe_str(d_fallback.get("value"))

            out[qid] = {"label": label, "description": desc}

    _dbg(f"[WIKI-DEBUG] wikidata.labels parsed: n={len(out)}")
    return out


def _sparql_throttle() -> None:
    global _LAST_SPARQL_TS
    now = time.time()
    delta = now - _LAST_SPARQL_TS
    if delta < _SPARQL_MIN_INTERVAL_S:
        time.sleep(_SPARQL_MIN_INTERVAL_S - delta)
    _LAST_SPARQL_TS = time.time()


def _wikidata_sparql(query: str) -> Mapping[str, object] | None:
    url = "https://query.wikidata.org/sparql"
    params = {"format": "json", "query": query}

    _sparql_throttle()
    _dbg(f"[WIKI-DEBUG] wikidata.sparql -> GET query.wikidata.org (len={len(query)})")

    try:
        session = _get_session()
        # timeout (connect, read) más tolerante para endpoint SPARQL
        resp = session.get(url, params=params, timeout=(5, 45))
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikidata.sparql EXC: {exc}")
        return None

    _dbg(f"[WIKI-DEBUG] wikidata.sparql <- status={resp.status_code}")

    if resp.status_code != 200:
        # si te están rate-limitando, esto ayuda a diagnosticar
        if resp.status_code in (429, 503):
            _dbg(f"[WIKI-DEBUG] wikidata.sparql body={resp.text[:300]!r}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        _dbg(f"[WIKI-DEBUG] wikidata.sparql JSON EXC: {exc}")
        return None

    if not isinstance(data, Mapping):
        return None
    return data


def _looks_like_film_from_wikipedia(wiki_raw: Mapping[str, object]) -> bool:
    desc = str(wiki_raw.get("description") or "").strip().lower()
    if not desc:
        return False
    hints = (
        "película",
        "film",
        "animated film",
        "feature film",
        "television film",
        "motion picture",
    )
    return any(h in desc for h in hints)


def _is_film_without_sparql(*, qid: str, wd_entity: Mapping[str, object], wiki_raw: Mapping[str, object] | None) -> bool:
    p31 = set(_extract_qids_from_claims(wd_entity, "P31"))
    if p31 & _FILM_INSTANCE_ALLOWLIST:
        _dbg(f"[WIKI-DEBUG] is_film_without_sparql({qid}) -> True (P31 allowlist hit)")
        return True

    # fallback soft: description wikipedia
    if wiki_raw is not None and _looks_like_film_from_wikipedia(wiki_raw):
        _dbg(f"[WIKI-DEBUG] is_film_without_sparql({qid}) -> True (wiki description fallback)")
        return True

    _dbg(f"[WIKI-DEBUG] is_film_without_sparql({qid}) -> False")
    return False


def _is_film_cached(*, cache: WikiCacheFile, qid: str, wd_entity: Mapping[str, object], wiki_raw: Mapping[str, object] | None) -> bool:
    cached = cache["is_film"].get(qid)
    if cached is not None:
        return bool(cached is True)
    ok = _is_film_without_sparql(qid=qid, wd_entity=wd_entity, wiki_raw=wiki_raw)
    cache["is_film"][qid] = bool(ok)
    return ok


def _fetch_qid_by_imdb_id(cache: WikiCacheFile, imdb_id: str) -> str | None:
    imdb_id = imdb_id.strip()
    if not imdb_id:
        return None

    cached = cache["imdb_qid"].get(imdb_id)
    if cached:
        _dbg(f"[WIKI-DEBUG] qid_by_imdb({imdb_id}) cache -> {cached!r}")
        return cached

    query = f"""
SELECT ?item WHERE {{
  ?item wdt:P345 "{imdb_id}" .
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
    if qid:
        cache["imdb_qid"][imdb_id] = qid
        _dbg(f"[WIKI-DEBUG] qid_by_imdb({imdb_id}) -> {qid!r}")
    return qid


def _extract_sitelink_title(entity: Mapping[str, object], language: str) -> str | None:
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, Mapping):
        return None
    key = f"{language}wiki"
    sl = sitelinks.get(key)
    if not isinstance(sl, Mapping):
        return None
    return _safe_str(sl.get("title"))


# -------------------------
# Main
# -------------------------

def get_wiki_entry(title: str, year: int | None, imdb_id: str | None) -> WikiCacheItem | None:
    lookup_title = normalize_title_for_lookup(title)
    if not lookup_title:
        _dbg(f"[WIKI-DEBUG] get_wiki_entry: empty lookup_title from title={title!r}")
        return None

    norm_title = lookup_title
    norm_year = str(year) if year is not None else ""

    cache = _load_cache()
    existing = _find_existing(cache["items"], norm_title, norm_year, imdb_id)
    if existing is not None:
        _dbg("[WIKI-DEBUG] get_wiki_entry: cache HIT")
        return existing

    # ------------------------------------------------------------
    # 0) Si tenemos imdb_id: intentar resolver TODO por Wikidata (P345)
    #    SIN ASK SPARQL extra: validamos "film" con EntityData (P31 allowlist + fallback wikipedia)
    # ------------------------------------------------------------
    if imdb_id:
        qid = _fetch_qid_by_imdb_id(cache, imdb_id)
        if qid:
            wd_entity = _fetch_wikidata_entity_json(qid)
            if wd_entity is not None:
                # elegir sitelink al idioma principal/fallback (si existe)
                sl_primary = _extract_sitelink_title(wd_entity, WIKI_LANGUAGE)
                sl_fallback = (
                    _extract_sitelink_title(wd_entity, WIKI_FALLBACK_LANGUAGE)
                    if WIKI_FALLBACK_LANGUAGE
                    else None
                )
                sl_title = sl_primary or sl_fallback
                sl_lang = WIKI_LANGUAGE if sl_primary else (WIKI_FALLBACK_LANGUAGE or WIKI_LANGUAGE)

                wiki_raw = None
                if sl_title:
                    wiki_raw = _fetch_wikipedia_summary_by_title(sl_title, sl_lang)

                # valida "film" sin SPARQL (cacheado)
                if _is_film_cached(cache=cache, qid=qid, wd_entity=wd_entity, wiki_raw=wiki_raw):
                    if wiki_raw is not None:
                        item = _build_and_cache_item(
                            cache=cache,
                            norm_title=norm_title,
                            norm_year=norm_year,
                            imdb_id=imdb_id,
                            wiki_raw=wiki_raw,
                            source_language=sl_lang,
                            wikibase_item=qid,
                            wd_entity=wd_entity,
                        )
                        # guardamos caches nuevas
                        _save_cache(cache)
                        return item

        _dbg("[WIKI-DEBUG] imdb path failed -> fallback to title search")

    # ------------------------------------------------------------
    # 1) Buscar por Wikipedia (como antes), pero:
    #    - probando varios candidatos
    #    - validando que el QID final sea “película” SIN SPARQL
    # ------------------------------------------------------------
    candidates = _choose_wikipedia_summary_candidates(lookup_title, year)
    _dbg(f"[WIKI-DEBUG] candidates total={len(candidates)}")

    for (cand_title, cand_lang) in candidates:
        raw = _fetch_wikipedia_summary_by_title(cand_title, cand_lang)
        if raw is None:
            continue

        wikibase_item = _safe_str(raw.get("wikibase_item"))
        if not wikibase_item:
            _dbg(f"[WIKI-DEBUG] candidate {cand_title!r} ({cand_lang}) has no wikibase_item -> skip")
            continue

        wd_entity = _fetch_wikidata_entity_json(wikibase_item)
        if wd_entity is None:
            continue

        if not _is_film_cached(cache=cache, qid=wikibase_item, wd_entity=wd_entity, wiki_raw=raw):
            _dbg(f"[WIKI-DEBUG] candidate QID {wikibase_item} is NOT film -> try next")
            continue

        item = _build_and_cache_item(
            cache=cache,
            norm_title=norm_title,
            norm_year=norm_year,
            imdb_id=imdb_id,
            wiki_raw=raw,
            source_language=cand_lang,
            wikibase_item=wikibase_item,
            wd_entity=wd_entity,
        )
        # guardamos caches nuevas
        _save_cache(cache)
        return item

    _dbg("[WIKI-DEBUG] get_wiki_entry: no valid film candidate found")
    return None


def _build_and_cache_item(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    wiki_raw: Mapping[str, object],
    source_language: str,
    wikibase_item: str,
    wd_entity: Mapping[str, object],
) -> WikiCacheItem:
    titles_obj = wiki_raw.get("titles")
    wikipedia_title: str | None = None
    if isinstance(titles_obj, Mapping):
        wikipedia_title = _safe_str(titles_obj.get("normalized")) or _safe_str(titles_obj.get("canonical"))

    wiki_block: WikiBlock = {
        "language": WIKI_LANGUAGE,
        "fallback_language": WIKI_FALLBACK_LANGUAGE,
        "source_language": source_language,
        "wikipedia_title": wikipedia_title,
        "wikipedia_pageid": _safe_int(wiki_raw.get("pageid")),
        "wikibase_item": wikibase_item,
        "summary": str(wiki_raw.get("extract") or ""),
        "description": str(wiki_raw.get("description") or ""),
        "urls": dict(wiki_raw.get("content_urls")) if isinstance(wiki_raw.get("content_urls"), Mapping) else {},
    }

    if "originalimage" in wiki_raw or "thumbnail" in wiki_raw:
        images: dict[str, object] = {}
        if isinstance(wiki_raw.get("originalimage"), Mapping):
            images["original"] = dict(wiki_raw.get("originalimage"))  # type: ignore[assignment]
        if isinstance(wiki_raw.get("thumbnail"), Mapping):
            images["thumbnail"] = dict(wiki_raw.get("thumbnail"))  # type: ignore[assignment]
        wiki_block["images"] = images

    wikidata_block: WikidataBlock = {"qid": wikibase_item}
    new_entities: dict[str, WikidataEntity] = {}

    directors = _extract_qids_from_claims(wd_entity, "P57")
    countries = _extract_qids_from_claims(wd_entity, "P495")
    genres = _extract_qids_from_claims(wd_entity, "P136")

    if directors:
        wikidata_block["directors"] = directors
    if countries:
        wikidata_block["countries"] = countries
    if genres:
        wikidata_block["genres"] = genres

    # Opción A: QIDs en item + diccionario "entities" global
    qids_to_label = list({*directors, *countries, *genres})
    fetched = _fetch_wikidata_labels(qids_to_label, WIKI_LANGUAGE, WIKI_FALLBACK_LANGUAGE)

    for qid, ent in fetched.items():
        etype: str | None = None
        if qid in directors:
            etype = "person"
        elif qid in countries:
            etype = "country"
        elif qid in genres:
            etype = "genre"

        merged: WikidataEntity = dict(ent)
        if etype:
            merged["type"] = etype
        new_entities[qid] = merged

    item: WikiCacheItem = {
        "Title": norm_title,
        "Year": norm_year,
        "imdbID": imdb_id,
        "wiki": wiki_block,
        "wikidata": wikidata_block,
    }

    cache["items"].append(item)
    for qid, ent in new_entities.items():
        cache["entities"][qid] = ent

    year_label = norm_year if norm_year else "?"
    _log(f"[WIKI] cached ({source_language}): {norm_title} ({year_label})")
    return item


# -------------------------------------------------------------------
# API estilo “cliente”
# -------------------------------------------------------------------

class WikiClient:
    def get_wiki(self, *, title: str, year: int | None, imdb_id: str | None) -> WikiCacheItem | None:
        return get_wiki_entry(title=title, year=year, imdb_id=imdb_id)


_WIKI_CLIENT_SINGLETON: WikiClient | None = None


def get_wiki_client() -> WikiClient:
    global _WIKI_CLIENT_SINGLETON
    if _WIKI_CLIENT_SINGLETON is None:
        _WIKI_CLIENT_SINGLETON = WikiClient()
    return _WIKI_CLIENT_SINGLETON