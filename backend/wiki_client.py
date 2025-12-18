from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend import logger as _logger
from backend.config import DATA_DIR, OMDB_RETRY_EMPTY_CACHE, SILENT_MODE
from backend.omdb_client import (
    is_omdb_data_empty_for_ratings,
    omdb_cache,
    search_omdb_by_imdb_id,
    search_omdb_with_candidates,
)

# --------------------------------------------------------------------
# Tipos auxiliares
# --------------------------------------------------------------------


class WikiMeta(TypedDict, total=False):
    wikidata_id: str | None
    wikipedia_title: str | None
    source_lang: str | None
    imdb_id: str | None


class _AliasRecord(TypedDict):
    __alias: str


WikiRecord = dict[str, object]
WikiCache = dict[str, object]


# --------------------------------------------------------------------
# Fichero de caché maestro (wiki + omdb fusionado)
# --------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
WIKI_CACHE_PATH: Path = DATA_DIR / "wiki_cache.json"

_wiki_cache: WikiCache = {}
_wiki_cache_loaded: bool = False

# Contexto de progreso para prefijar logs (x/total, biblioteca, título)
_CURRENT_PROGRESS: dict[str, object | None] = {
    "idx": None,
    "total": None,
    "library": None,
    "title": None,
}


def set_wiki_progress(idx: int, total: int, library_title: str, movie_title: str) -> None:
    _CURRENT_PROGRESS["idx"] = idx
    _CURRENT_PROGRESS["total"] = total
    _CURRENT_PROGRESS["library"] = library_title
    _CURRENT_PROGRESS["title"] = movie_title


def _progress_prefix() -> str:
    idx = _CURRENT_PROGRESS.get("idx")
    total = _CURRENT_PROGRESS.get("total")
    library = _CURRENT_PROGRESS.get("library")
    title = _CURRENT_PROGRESS.get("title")

    if (
        not isinstance(idx, int)
        or not isinstance(total, int)
        or not isinstance(library, str)
        or not isinstance(title, str)
    ):
        return ""

    return f"({idx}/{total}) {library} · {title} | "


def _log_wiki(msg: str) -> None:
    prefix = _progress_prefix()
    text = f"{prefix}{msg}"
    try:
        _logger.info(text)
    except Exception:
        if not SILENT_MODE:
            print(text)


# --------------------------------------------------------------------
# HTTP session with retries
# --------------------------------------------------------------------
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _SESSION = session
    return _SESSION


# --------------------------------------------------------------------
# Carga / guardado de wiki_cache
# --------------------------------------------------------------------
def _load_wiki_cache() -> None:
    global _wiki_cache_loaded, _wiki_cache
    if _wiki_cache_loaded:
        return

    if WIKI_CACHE_PATH.exists():
        try:
            with WIKI_CACHE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _wiki_cache = dict(data) if isinstance(data, dict) else {}
            _log_wiki(f"[WIKI] wiki_cache cargada ({len(_wiki_cache)} entradas)")
        except Exception as exc:
            _logger.warning(f"[WIKI] Error cargando wiki_cache.json: {exc}")
            try:
                broken = WIKI_CACHE_PATH.with_suffix(".broken.json")
                WIKI_CACHE_PATH.replace(broken)
                _logger.warning(f"[WIKI] Archivo corrupto renombrado a {broken}")
            except Exception:
                pass
            _wiki_cache = {}
    else:
        _wiki_cache = {}

    _wiki_cache_loaded = True
    _upgrade_wiki_cache_to_alias_format()


def _save_wiki_cache() -> None:
    if not _wiki_cache_loaded:
        return
    try:
        dirpath = WIKI_CACHE_PATH.parent
        dirpath.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(dirpath),
        ) as tf:
            json.dump(_wiki_cache, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name
        os.replace(temp_name, str(WIKI_CACHE_PATH))
    except Exception as exc:
        _logger.error(f"[WIKI] Error guardando wiki_cache.json: {exc}")


def _normalize_title(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def _safe_str(value: object) -> str | None:
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return None


def _ratings_fields() -> tuple[str, ...]:
    return ("imdbRating", "imdbVotes", "Ratings", "Metascore")


def _is_meaningful_omdb_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().upper() == "N/A":
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _extract_wiki_part(record: WikiRecord | None) -> dict[str, object]:
    if not record:
        return {}
    w = record.get("__wiki")
    return dict(w) if isinstance(w, dict) else {}


def _extract_imdb_from_record(record: WikiRecord) -> str | None:
    imdb_obj = record.get("imdbID")
    imdb = _safe_str(imdb_obj)
    if imdb:
        return imdb

    w = record.get("__wiki")
    if isinstance(w, dict):
        imdb2 = _safe_str(w.get("imdb_id"))
        if imdb2:
            return imdb2
    return None


def _make_imdb_key(imdb_id: str) -> str:
    return f"imdb:{imdb_id.strip().lower()}"


def _is_alias_record(obj: object) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("__alias"), str)


def _resolve_cache_record(key: str, *, max_depth: int = 10) -> WikiRecord | None:
    current_key = key
    seen: set[str] = set()

    for _ in range(max_depth):
        if current_key in seen:
            return None
        seen.add(current_key)

        raw = _wiki_cache.get(current_key)
        if raw is None:
            return None
        if _is_alias_record(raw):
            target = raw.get("__alias")
            if not isinstance(target, str) or not target.strip():
                return None
            current_key = target
            continue
        return dict(raw) if isinstance(raw, dict) else None

    return None


def _set_alias(alias_key: str, target_key: str) -> None:
    _wiki_cache[alias_key] = _AliasRecord(__alias=target_key)


def _upgrade_wiki_cache_to_alias_format() -> None:
    """
    Migra en memoria:
      - Si existe un title:* con record completo que tiene imdbID → convertir a alias
        y asegurar canon imdb:*.
      - Si existe un imdb:* → siempre se considera canónico.
    """
    changed = False

    for key, value in list(_wiki_cache.items()):
        if not isinstance(key, str):
            continue
        if not key.startswith("title:"):
            continue
        if not isinstance(value, dict):
            continue
        if _is_alias_record(value):
            continue

        imdb_id = _extract_imdb_from_record(value)
        if not imdb_id:
            continue

        canon_key = _make_imdb_key(imdb_id)
        canon_obj = _wiki_cache.get(canon_key)

        if canon_obj is None or _is_alias_record(canon_obj):
            _wiki_cache[canon_key] = dict(value)
            changed = True

        _set_alias(key, canon_key)
        changed = True

    if changed:
        _save_wiki_cache()
        _log_wiki("[WIKI] wiki_cache migrada a formato alias (imdb único).")


def _apply_ratings_from_omdb_cache(record: WikiRecord, imdb_id: str) -> bool:
    cached_obj = omdb_cache.get(imdb_id)
    cached = cached_obj if isinstance(cached_obj, dict) else None
    if cached is None:
        return False

    changed = False
    for k in _ratings_fields():
        if k not in cached:
            continue
        val = cached.get(k)
        if not _is_meaningful_omdb_value(val):
            continue

        cur = record.get(k)
        if cur != val:
            record[k] = val
            changed = True

    return changed


# --------------------------------------------------------------------
# Consultas a Wikidata / Wikipedia
# --------------------------------------------------------------------
WIKIDATA_API: str = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL: str = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_TEMPLATE: str = "https://{lang}.wikipedia.org/w/api.php"


def _wikidata_get_entity(wikidata_id: str) -> dict[str, object] | None:
    try:
        session = _get_session()
        resp = session.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": wikidata_id,
                "format": "json",
                "props": "labels|claims|sitelinks",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities", {})
        if isinstance(entities, dict):
            entity = entities.get(wikidata_id)
            if isinstance(entity, dict):
                return entity
        return None
    except Exception as exc:
        _log_wiki(f"[WIKI] Error obteniendo entidad {wikidata_id} de Wikidata: {exc}")
        return None


def _wikidata_search_by_imdb(imdb_id: str) -> tuple[str, dict[str, object]] | None:
    query = f"""
    SELECT ?item WHERE {{
      ?item wdt:P345 "{imdb_id}" .
    }} LIMIT 1
    """

    try:
        session = _get_session()
        resp = session.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", {}).get("bindings", [])
        if not isinstance(results, list) or not results:
            return None

        first = results[0]
        item = first.get("item", {})
        if not isinstance(item, dict):
            return None

        item_uri = item.get("value")
        if not isinstance(item_uri, str):
            return None

        wikidata_id = item_uri.split("/")[-1]
        entity = _wikidata_get_entity(wikidata_id)
        if not entity:
            return None

        return wikidata_id, entity
    except Exception as exc:
        _log_wiki(f"[WIKI] Error en búsqueda SPARQL por IMDb ID {imdb_id}: {exc}")
        return None


def _wikidata_search_by_title(
    title: str,
    year: int | None,
    language: str = "en",
) -> tuple[str, dict[str, object]] | None:
    try:
        session = _get_session()
        resp = session.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": title,
                "language": language,
                "type": "item",
                "format": "json",
                "limit": 5,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("search", [])
        if not isinstance(search_results, list) or not search_results:
            return None

        for candidate in search_results:
            if not isinstance(candidate, dict):
                continue

            wikidata_id = candidate.get("id")
            if not isinstance(wikidata_id, str):
                continue

            entity = _wikidata_get_entity(wikidata_id)
            if not entity:
                continue

            claims = entity.get("claims", {})
            if not isinstance(claims, dict):
                continue

            if "P31" in claims:
                ok_type = False
                for inst in claims["P31"]:
                    if not isinstance(inst, dict):
                        continue
                    val = (
                        inst.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                    )
                    if isinstance(val, dict) and val.get("id") in {
                        "Q11424",
                        "Q24869",
                    }:
                        ok_type = True
                        break
                if not ok_type:
                    continue

            if year is not None and "P577" in claims:
                try:
                    first_p577 = claims["P577"][0]
                    if isinstance(first_p577, dict):
                        time_str = (
                            first_p577.get("mainsnak", {})
                            .get("datavalue", {})
                            .get("value", {})
                            .get("time")
                        )
                        if isinstance(time_str, str) and len(time_str) >= 5:
                            ent_year = int(time_str[1:5])
                            if abs(ent_year - int(year)) > 1:
                                continue
                except Exception:
                    pass

            return wikidata_id, entity

        return None
    except Exception as exc:
        _log_wiki(f"[WIKI] Error buscando por título '{title}' en Wikidata: {exc}")
        return None


def _extract_imdb_id_from_entity(entity: dict[str, object]) -> str | None:
    claims = entity.get("claims", {})
    if not isinstance(claims, dict) or "P345" not in claims:
        return None
    try:
        first = claims["P345"][0]
        if not isinstance(first, dict):
            return None
        snak = first["mainsnak"]
        if not isinstance(snak, dict):
            return None
        datavalue = snak["datavalue"]
        if not isinstance(datavalue, dict):
            return None
        value = datavalue["value"]
        if not isinstance(value, str):
            return None
        return value
    except Exception:
        return None


def _extract_wikipedia_title(entity: dict[str, object], language: str) -> str | None:
    sitelinks = entity.get("sitelinks", {})
    if not isinstance(sitelinks, dict):
        return None
    key = f"{language}wiki"
    site = sitelinks.get(key)
    if isinstance(site, dict):
        title = site.get("title")
        return str(title) if title is not None else None
    return None


def _wiki_meta_from_entity(
    *,
    wikidata_id: str | None,
    entity: dict[str, object] | None,
    imdb_id: str | None,
    language: str,
) -> WikiMeta:
    if entity is None:
        return {}

    wiki_title = _extract_wikipedia_title(entity, language) or _extract_wikipedia_title(entity, "es")
    return {
        "wikidata_id": wikidata_id,
        "wikipedia_title": wiki_title,
        "source_lang": language,
        "imdb_id": imdb_id,
    }


def _try_resolve_imdb_from_wiki_cache_by_title(norm_title: str) -> str | None:
    if not norm_title:
        return None

    imdb_ids: set[str] = set()
    for key in list(_wiki_cache.keys()):
        if not isinstance(key, str):
            continue
        if not key.startswith("title:"):
            continue

        parts = key.split(":", 2)
        if len(parts) != 3:
            continue

        key_norm_title = parts[2]
        if key_norm_title != norm_title:
            continue

        resolved = _resolve_cache_record(key)
        if resolved is None:
            continue

        imdb = _extract_imdb_from_record(resolved)
        imdb_norm = imdb.lower() if isinstance(imdb, str) and imdb.strip() else None
        if imdb_norm:
            imdb_ids.add(imdb_norm)
        if len(imdb_ids) > 1:
            return None

    if len(imdb_ids) == 1:
        return next(iter(imdb_ids))
    return None


# --------------------------------------------------------------------
# API pública: get_movie_record
# --------------------------------------------------------------------


def get_movie_record(
    title: str,
    year: int | None = None,
    imdb_id_hint: str | None = None,
    language: str = "en",
) -> WikiRecord | None:
    _load_wiki_cache()

    norm_title = _normalize_title(title)
    imdb_norm = _safe_str(imdb_id_hint)
    imdb_norm_l = imdb_norm.lower() if imdb_norm else None

    if imdb_norm_l is None:
        resolved = _try_resolve_imdb_from_wiki_cache_by_title(norm_title)
        if resolved is not None:
            imdb_norm_l = resolved
            _log_wiki(
                f"[WIKI] imdb_id_hint ausente: resuelto desde wiki_cache por título "
                f"norm='{norm_title}' → imdbID={imdb_norm_l}"
            )

    imdb_key = _make_imdb_key(imdb_norm_l) if imdb_norm_l else None
    title_key = f"title:{year}:{norm_title}"
    base_cache_key = imdb_key if imdb_key else title_key

    _log_wiki(
        f"[WIKI] get_movie_record(title='{title}', year={year}, imdb_hint={imdb_id_hint}) "
        f"→ base_cache_key='{base_cache_key}'"
    )

    cached: WikiRecord | None = None
    if imdb_key:
        cached = _resolve_cache_record(imdb_key)
    else:
        cached = _resolve_cache_record(title_key)

    if cached is not None:
        _log_wiki(f"[WIKI] cache HIT para {imdb_key or title_key}")

        if OMDB_RETRY_EMPTY_CACHE and is_omdb_data_empty_for_ratings(cached):
            imdb_cached = _extract_imdb_from_record(cached)
            if imdb_cached:
                _log_wiki(
                    f"[WIKI] Registro en wiki_cache sin ratings, reintentando OMDb "
                    f"con imdbID={imdb_cached}..."
                )
                refreshed = search_omdb_by_imdb_id(imdb_cached)
                if isinstance(refreshed, dict) and refreshed.get("Response") == "True":
                    merged: WikiRecord = dict(cached)
                    merged.update(refreshed)

                    wiki_part = cached.get("__wiki")
                    if isinstance(wiki_part, dict):
                        merged["__wiki"] = wiki_part

                    canon_key = _make_imdb_key(imdb_cached)
                    _wiki_cache[canon_key] = merged
                    _set_alias(title_key, canon_key)
                    if imdb_key:
                        _set_alias(imdb_key, canon_key)
                    _save_wiki_cache()
                    _log_wiki(
                        f"[WIKI] wiki_cache actualizada con datos OMDb para {canon_key} (alias por título)."
                    )
                    return merged

        return cached

    _log_wiki(f"[WIKI] cache MISS para {base_cache_key}, resolviendo...")

    wikidata_id: str | None = None
    entity: dict[str, object] | None = None
    imdb_id_from_wiki: str | None = None
    source_lang: str = language

    if imdb_norm_l:
        result = _wikidata_search_by_imdb(imdb_norm_l)
        if result is not None:
            wikidata_id, entity = result
            imdb_id_from_wiki = _extract_imdb_id_from_entity(entity)
            _log_wiki(
                f"[WIKI] Encontrado en Wikidata por IMDb ID {imdb_norm_l}: "
                f"wikidata_id={wikidata_id}, imdb_id_wiki={imdb_id_from_wiki}"
            )

        if entity is None:
            for lang in (language, "es"):
                result2 = _wikidata_search_by_title(title, year, language=lang)
                if result2 is None:
                    continue
                wikidata_id, entity = result2
                imdb_id_from_wiki = _extract_imdb_id_from_entity(entity)
                source_lang = lang
                _log_wiki(
                    f"[WIKI] Reintento por título (lang={lang}): "
                    f"wikidata_id={wikidata_id}, imdb_id_wiki={imdb_id_from_wiki}"
                )
                break
    else:
        for lang in (language, "es"):
            result3 = _wikidata_search_by_title(title, year, language=lang)
            if result3 is None:
                continue
            wikidata_id, entity = result3
            imdb_id_from_wiki = _extract_imdb_id_from_entity(entity)
            source_lang = lang
            _log_wiki(
                f"[WIKI] Encontrado en Wikidata por título (lang={lang}): "
                f"wikidata_id={wikidata_id}, imdb_id_wiki={imdb_id_from_wiki}"
            )
            break

    imdb_id_final = (imdb_id_from_wiki or imdb_norm_l) if (imdb_id_from_wiki or imdb_norm_l) else None

    if imdb_id_final:
        canon_key = _make_imdb_key(imdb_id_final)
        cached2 = _resolve_cache_record(canon_key)
        if cached2 is not None:
            _log_wiki(f"[WIKI] cache HIT (post-resolve) para {canon_key}")
            _set_alias(title_key, canon_key)
            if imdb_key:
                _set_alias(imdb_key, canon_key)
            _save_wiki_cache()
            return cached2

    omdb_data: dict[str, object] | None = None
    omdb_called_by_title = False

    if imdb_id_final is None:
        omdb_called_by_title = True
        omdb_data = search_omdb_with_candidates(title, year)
        if isinstance(omdb_data, dict) and omdb_data.get("Response") == "True":
            imdb_obj = omdb_data.get("imdbID")
            imdb_id_from_omdb = _safe_str(imdb_obj)
            if imdb_id_from_omdb:
                imdb_id_final = imdb_id_from_omdb.lower()

    if imdb_id_final is not None and not omdb_called_by_title:
        omdb_data = search_omdb_by_imdb_id(imdb_id_final)

    wiki_meta = _wiki_meta_from_entity(
        wikidata_id=wikidata_id,
        entity=entity,
        imdb_id=imdb_id_final,
        language=source_lang,
    )

    if isinstance(omdb_data, dict) and omdb_data:
        record_out: WikiRecord = dict(omdb_data)
        if imdb_id_final and not record_out.get("imdbID"):
            record_out["imdbID"] = imdb_id_final
    else:
        record_out = {
            "Title": title,
            "Year": str(year) if year is not None else None,
            "imdbID": imdb_id_final,
        }

    if wiki_meta:
        record_out["__wiki"] = wiki_meta

    if imdb_id_final:
        _apply_ratings_from_omdb_cache(record_out, imdb_id_final)

    existing: WikiRecord | None = None
    canon_key_final: str | None = _make_imdb_key(imdb_id_final) if imdb_id_final else None

    if canon_key_final:
        existing = _resolve_cache_record(canon_key_final)
    if existing is None:
        existing = _resolve_cache_record(title_key)
    if existing is None and imdb_key:
        existing = _resolve_cache_record(imdb_key)

    if existing is not None and not wiki_meta:
        existing_wiki = existing.get("__wiki")
        if isinstance(existing_wiki, dict):
            record_out["__wiki"] = existing_wiki

    if existing is not None:
        old_imdb = _safe_str(existing.get("imdbID"))
        new_imdb = _safe_str(record_out.get("imdbID"))

        old_wiki = _extract_wiki_part(existing)
        new_wiki = _extract_wiki_part(record_out)

        unchanged_ids = (old_imdb or None) == (new_imdb or None)
        unchanged_wiki = old_wiki == new_wiki

        unchanged_ratings = True
        for k in _ratings_fields():
            if existing.get(k) != record_out.get(k):
                unchanged_ratings = False
                break

        if unchanged_ids and unchanged_wiki and unchanged_ratings:
            return existing

    if canon_key_final:
        _wiki_cache[canon_key_final] = record_out
        _set_alias(title_key, canon_key_final)
        if imdb_key:
            _set_alias(imdb_key, canon_key_final)
    else:
        _wiki_cache[title_key] = record_out

    _save_wiki_cache()

    wikidata_id_logged: str | None = None
    wiki_part = record_out.get("__wiki")
    if isinstance(wiki_part, dict):
        wd_val = wiki_part.get("wikidata_id")
        if isinstance(wd_val, str):
            wikidata_id_logged = wd_val

    keys_logged: list[str] = [title_key]
    if canon_key_final:
        keys_logged.append(canon_key_final)
    if imdb_key:
        keys_logged.append(imdb_key)

    _log_wiki(
        f"[WIKI] Registro maestro guardado. "
        f"imdbID={record_out.get('imdbID')}, wikidata_id={wikidata_id_logged}, "
        f"keys={sorted(set(keys_logged))}"
    )

    return record_out