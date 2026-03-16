from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from server.api.caching.file_cache import FileCache
from server.api.services.omdb import load_payload as load_omdb_payload
from server.api.services.wiki import load_payload as load_wiki_payload

_SEARCH_COLUMNS = [
    "title",
    "name",
    "file",
    "path",
    "imdb_id",
    "imdbID",
    "director",
    "actors",
    "genre",
    "plot",
    "wikipedia_title",
    "search_context",
]
_INTERNAL_SEARCH_COL = "__search_blob"
_CONTEXT_ENRICHED_COL = "__context_enriched"
_CONTEXT_TEXT_LIMIT = 4096
_OMDB_CONTEXT_COLUMNS = {
    "director": "Director",
    "actors": "Actors",
    "genre": "Genre",
    "plot": "Plot",
}


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text == "N/A":
        return None
    return text


def _payload_lookup_record(
    payload: Mapping[str, Any] | None,
    *,
    imdb_id: str | None,
    title: str | None,
    year: str | None,
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None

    records = payload.get("records")
    index_imdb = payload.get("index_imdb") or payload.get("index") or {}
    index_ty = payload.get("index_ty") or {}
    if not isinstance(records, Mapping):
        return None
    if not isinstance(index_imdb, Mapping):
        index_imdb = {}
    if not isinstance(index_ty, Mapping):
        index_ty = {}

    record_id: str | None = None
    clean_imdb = _clean_text(imdb_id)
    if clean_imdb:
        imdb_key = clean_imdb.lower()
        record_id = cast(
            str | None,
            index_imdb.get(imdb_key)
            or index_imdb.get(f"imdb:{imdb_key}")
            or index_imdb.get(clean_imdb),
        )

    clean_title = _clean_text(title)
    clean_year = _clean_text(year)
    if record_id is None and clean_title:
        title_key = clean_title.lower()
        record_id = cast(
            str | None,
            index_ty.get(f"{title_key}|{clean_year or ''}")
            or index_ty.get(title_key)
            or index_ty.get(f"{title_key}|"),
        )

    if record_id is None:
        return None

    record = records.get(str(record_id))
    return cast(dict[str, Any] | None, record if isinstance(record, dict) else None)


def _collect_context_terms(value: Any, parts: list[str]) -> None:
    if len(" | ".join(parts)) >= _CONTEXT_TEXT_LIMIT:
        return
    if value is None:
        return
    if isinstance(value, str):
        clean = _clean_text(value)
        if clean is not None:
            parts.append(clean)
        return
    if isinstance(value, (int, float, bool)):
        parts.append(str(value))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {"urls", "desktop", "mobile", "poster"}:
                continue
            _collect_context_terms(item, parts)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_context_terms(item, parts)


def _build_search_context(
    omdb_payload: Mapping[str, Any] | None,
    wiki_payload: Mapping[str, Any] | None,
    wikidata_payload: Mapping[str, Any] | None,
) -> str | None:
    parts: list[str] = []
    for payload in (omdb_payload, wiki_payload, wikidata_payload):
        _collect_context_terms(payload, parts)
    if not parts:
        return None

    unique_parts = list(dict.fromkeys(parts))
    context = " | ".join(unique_parts)
    if len(context) > _CONTEXT_TEXT_LIMIT:
        return context[:_CONTEXT_TEXT_LIMIT].rstrip()
    return context


def enrich_report_context(
    df: pd.DataFrame,
    *,
    cache: FileCache,
    profile_id: str | None,
) -> None:
    if df.empty or _CONTEXT_ENRICHED_COL in df.columns:
        return

    try:
        omdb_cache = load_omdb_payload(cache, profile_id)
    except Exception:
        omdb_cache = {}

    try:
        wiki_cache = load_wiki_payload(cache, profile_id)
    except Exception:
        wiki_cache = {}

    updates: dict[str, list[str | None]] = {
        "director": [],
        "actors": [],
        "genre": [],
        "plot": [],
        "wikipedia_title": [],
        "wikidata_id": [],
        "source_language": [],
        "search_context": [],
    }

    records = cast(list[dict[str, Any]], df.to_dict(orient="records"))
    for row in records:
        record_omdb = _payload_lookup_record(
            omdb_cache,
            imdb_id=_clean_text(row.get("imdb_id")),
            title=_clean_text(row.get("title")),
            year=_clean_text(row.get("year")),
        )
        record_wiki = _payload_lookup_record(
            wiki_cache,
            imdb_id=_clean_text(row.get("imdb_id")),
            title=_clean_text(row.get("title")),
            year=_clean_text(row.get("year")),
        )

        omdb_payload = (
            cast(dict[str, Any], record_omdb.get("omdb"))
            if isinstance(record_omdb, dict)
            and isinstance(record_omdb.get("omdb"), dict)
            else None
        )
        wiki_payload = (
            cast(dict[str, Any], record_wiki.get("wiki"))
            if isinstance(record_wiki, dict)
            and isinstance(record_wiki.get("wiki"), dict)
            else None
        )
        wikidata_payload = (
            cast(dict[str, Any], record_wiki.get("wikidata"))
            if isinstance(record_wiki, dict)
            and isinstance(record_wiki.get("wikidata"), dict)
            else None
        )

        for column, omdb_key in _OMDB_CONTEXT_COLUMNS.items():
            updates[column].append(
                _clean_text(row.get(column))
                or (
                    _clean_text(omdb_payload.get(omdb_key))
                    if isinstance(omdb_payload, Mapping)
                    else None
                )
            )

        updates["wikipedia_title"].append(
            _clean_text(row.get("wikipedia_title"))
            or (
                _clean_text(wiki_payload.get("wikipedia_title"))
                if isinstance(wiki_payload, Mapping)
                else None
            )
        )
        updates["wikidata_id"].append(
            _clean_text(row.get("wikidata_id"))
            or (
                _clean_text(wikidata_payload.get("qid"))
                if isinstance(wikidata_payload, Mapping)
                else None
            )
        )
        updates["source_language"].append(
            _clean_text(row.get("source_language"))
            or (
                _clean_text(wiki_payload.get("source_language"))
                if isinstance(wiki_payload, Mapping)
                else None
            )
        )
        updates["search_context"].append(
            _build_search_context(omdb_payload, wiki_payload, wikidata_payload)
        )

    for column, values in updates.items():
        series = pd.Series(values, index=df.index, dtype="string")
        if column not in df.columns:
            df[column] = series
            continue

        current = df[column].astype("string")
        missing = current.fillna("").str.strip() == ""
        if bool(missing.any()):
            df.loc[missing, column] = series.loc[missing]

    df[_CONTEXT_ENRICHED_COL] = pd.Series(
        ["1"] * len(df), index=df.index, dtype="string"
    )


def prepare_search_blob(df: pd.DataFrame) -> None:
    cols = [c for c in _SEARCH_COLUMNS if c in df.columns]
    if not cols:
        return

    try:
        parts: list[pd.Series] = []
        for c in cols:
            parts.append(df[c].astype("string").fillna("").str.lower())

        df[_INTERNAL_SEARCH_COL] = parts[0]
        for s in parts[1:]:
            df[_INTERNAL_SEARCH_COL] = df[_INTERNAL_SEARCH_COL] + " | " + s
    except Exception:
        return


def strip_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("__")]
    if not internal_cols:
        return df
    return cast(pd.DataFrame, df.drop(columns=internal_cols, errors="ignore"))


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        return value

    return value


def df_to_page(
    df: pd.DataFrame,
    *,
    offset: int,
    limit: int,
    query: str | None,
) -> dict[str, Any]:
    view = df

    if query:
        q = query.strip().lower()
        if q:
            if _INTERNAL_SEARCH_COL in view.columns:
                try:
                    blob_mask = (
                        view[_INTERNAL_SEARCH_COL]
                        .astype("string")
                        .fillna("")
                        .str.contains(q, regex=False, na=False)
                    )
                    view = view[blob_mask]
                except Exception:
                    pass

            if _INTERNAL_SEARCH_COL not in view.columns:
                candidate_cols = [c for c in _SEARCH_COLUMNS if c in view.columns]
                if candidate_cols:
                    cols_mask: pd.Series | None = None
                    for c in candidate_cols:
                        s = (
                            view[c]
                            .astype("string")
                            .fillna("")
                            .str.lower()
                            .str.contains(q, regex=False, na=False)
                        )
                        cols_mask = s if cols_mask is None else (cols_mask | s)

                    if cols_mask is not None:
                        view = view[cols_mask]

    total = int(len(view))

    page = cast(pd.DataFrame, view.iloc[offset : offset + limit])
    page = strip_internal_cols(page)

    raw_items = page.to_dict(orient="records")
    records = cast(list[dict[str, Any]], raw_items)

    items: list[dict[str, Any]] = [
        {k: _json_safe_value(v) for k, v in row.items()} for row in records
    ]

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
