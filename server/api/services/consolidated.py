# _consolidate + get_record helpers
from __future__ import annotations

from typing import Any

from server.api.services.omdb import load_payload as load_omdb_payload
from server.api.services.wiki import load_payload as load_wiki_payload
from server.api.caching.file_cache import FileCache


def get_omdb_record(
    *,
    cache: FileCache,
    imdb_id: str | None,
    norm_title: str | None,
    norm_year: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    payload = load_omdb_payload(cache)
    records = payload.get("records") or {}
    idx_imdb = payload.get("index_imdb") or {}
    idx_ty = payload.get("index_ty") or {}

    rid: str | None = None
    if imdb_id:
        rid = idx_imdb.get(imdb_id.strip().lower()) or idx_imdb.get(imdb_id.strip())
    if not rid and norm_title is not None:
        t = norm_title.strip().lower()
        y = (norm_year or "").strip()
        key = f"{t}|{y}" if y else t
        rid = idx_ty.get(key) or idx_ty.get(t) or idx_ty.get(f"{t}|")

    if not rid:
        return None, None

    rec = records.get(str(rid))
    if not isinstance(rec, dict):
        return str(rid), None
    return str(rid), rec


def get_wiki_record(
    *,
    cache: FileCache,
    imdb_id: str | None,
    norm_title: str | None,
    norm_year: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    payload = load_wiki_payload(cache)
    records = payload.get("records") or {}
    idx_imdb = payload.get("index_imdb") or payload.get("index") or {}
    idx_ty = payload.get("index_ty") or {}

    rid: str | None = None
    if imdb_id:
        imdb_norm = imdb_id.strip().lower()
        rid = (
            idx_imdb.get(imdb_norm)
            or idx_imdb.get(f"imdb:{imdb_norm}")
            or idx_imdb.get(imdb_id.strip())
        )
    if not rid and norm_title is not None:
        t = norm_title.strip().lower()
        y = (norm_year or "").strip()
        key = f"{t}|{y}" if y else t
        rid = idx_ty.get(key) or idx_ty.get(t) or idx_ty.get(f"{t}|")

    if not rid:
        return None, None

    rec = records.get(str(rid))
    if not isinstance(rec, dict):
        return str(rid), None
    return str(rid), rec


def consolidate(
    *,
    cache: FileCache,
    imdb_id: str | None,
    title: str | None,
    year: str | None,
) -> dict[str, Any]:
    norm_title = title.strip().lower() if isinstance(title, str) else None
    norm_year = (year or "").strip() if isinstance(year, str) else None
    imdb_norm = imdb_id.strip().lower() if isinstance(imdb_id, str) else None

    omdb_rid, omdb_rec = get_omdb_record(
        cache=cache, imdb_id=imdb_norm, norm_title=norm_title, norm_year=norm_year
    )
    wiki_rid, wiki_rec = get_wiki_record(
        cache=cache, imdb_id=imdb_norm, norm_title=norm_title, norm_year=norm_year
    )

    omdb_payload: dict[str, Any] | None = None
    wiki_payload: dict[str, Any] | None = None
    wikidata_payload: dict[str, Any] | None = None
    wiki_from_omdb_cache: dict[str, Any] | None = None

    if isinstance(omdb_rec, dict):
        omdb_obj = omdb_rec.get("omdb")
        if isinstance(omdb_obj, dict):
            omdb_payload = omdb_obj
            w = omdb_obj.get("__wiki")
            if isinstance(w, dict):
                wiki_from_omdb_cache = w

    if isinstance(wiki_rec, dict):
        w = wiki_rec.get("wiki")
        wd = wiki_rec.get("wikidata")
        if isinstance(w, dict):
            wiki_payload = w
        if isinstance(wd, dict):
            wikidata_payload = wd

    resolved_imdb: str | None = imdb_norm
    if not resolved_imdb and isinstance(omdb_rec, dict):
        v = omdb_rec.get("imdbID")
        if isinstance(v, str) and v.strip():
            resolved_imdb = v.strip().lower()
    if not resolved_imdb and isinstance(wiki_rec, dict):
        v = wiki_rec.get("imdbID")
        if isinstance(v, str) and v.strip():
            resolved_imdb = v.strip().lower()

    merged: dict[str, Any] = {}
    if isinstance(omdb_payload, dict):
        merged["title"] = omdb_payload.get("Title")
        merged["year"] = omdb_payload.get("Year")
        merged["poster"] = omdb_payload.get("Poster")
        merged["plot"] = omdb_payload.get("Plot")
        merged["genre"] = omdb_payload.get("Genre")
        merged["director"] = omdb_payload.get("Director")
        merged["actors"] = omdb_payload.get("Actors")
        merged["imdbRating"] = omdb_payload.get("imdbRating")
        merged["imdbVotes"] = omdb_payload.get("imdbVotes")
        merged["ratings"] = omdb_payload.get("Ratings")

    if isinstance(wiki_payload, dict):
        merged["wikipedia_title"] = wiki_payload.get(
            "wikipedia_title"
        ) or wiki_payload.get("title")
        merged["source_language"] = wiki_payload.get(
            "source_language"
        ) or wiki_payload.get("language")

    if isinstance(wikidata_payload, dict):
        merged["wikidata_id"] = wikidata_payload.get(
            "wikidata_id"
        ) or wikidata_payload.get("qid")

    if not merged.get("wikipedia_title") and isinstance(wiki_from_omdb_cache, dict):
        merged["wikipedia_title"] = wiki_from_omdb_cache.get("wikipedia_title")
        merged["source_language"] = wiki_from_omdb_cache.get("source_language")
        merged["wikidata_id"] = merged.get("wikidata_id") or wiki_from_omdb_cache.get(
            "wikidata_id"
        )

    return {
        "key": {
            "imdb_id": resolved_imdb,
            "title_norm": norm_title,
            "year_norm": norm_year,
        },
        "sources": {
            "omdb": {
                "rid": omdb_rid,
                "status": (
                    (omdb_rec or {}).get("status")
                    if isinstance(omdb_rec, dict)
                    else None
                ),
                "fetched_at": (
                    (omdb_rec or {}).get("fetched_at")
                    if isinstance(omdb_rec, dict)
                    else None
                ),
                "ttl_s": (
                    (omdb_rec or {}).get("ttl_s")
                    if isinstance(omdb_rec, dict)
                    else None
                ),
            },
            "wiki": {
                "rid": wiki_rid,
                "status": (
                    (wiki_rec or {}).get("status")
                    if isinstance(wiki_rec, dict)
                    else None
                ),
                "fetched_at": (
                    (wiki_rec or {}).get("fetched_at")
                    if isinstance(wiki_rec, dict)
                    else None
                ),
                "ttl_s": (
                    (wiki_rec or {}).get("ttl_s")
                    if isinstance(wiki_rec, dict)
                    else None
                ),
            },
        },
        "merged": merged,
        "omdb": omdb_payload,
        "wiki": wiki_payload,
        "wikidata": wikidata_payload,
        "wiki_from_omdb_cache": wiki_from_omdb_cache,
    }
