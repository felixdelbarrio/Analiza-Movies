from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeAlias

from backend import logger as _logger
from backend.analyze_input_core import AnalysisRow, analyze_input_movie
from backend.metadata_fix import generate_metadata_suggestions_row
from backend.movie_input import MovieInput
from backend.omdb_client import omdb_query_with_cache
from backend.wiki_client import get_wiki_client


_OmdbCacheKey: TypeAlias = tuple[str, int | None, str | None]

# ✅ CACHÉS A NIVEL DE MÓDULO (persisten durante toda la ejecución)
#    Así evitamos llamar a OMDb/Wiki repetidamente para los mismos títulos.
_OMDB_LOCAL_CACHE: dict[_OmdbCacheKey, dict[str, object]] = {}
_WIKI_LOCAL_CACHE: dict[_OmdbCacheKey, dict[str, object]] = {}


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.upper() == "N/A":
            return None
        s = s.replace(",", "")
        return int(s)
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return None
    except Exception:
        return None


def _extract_wiki_meta(omdb_record: Mapping[str, object] | None) -> dict[str, object]:
    if not omdb_record:
        return {}
    wiki_raw = omdb_record.get("__wiki")
    if isinstance(wiki_raw, Mapping):
        return dict(wiki_raw)
    return {}


def analyze_movie(
    movie_input: MovieInput,
    *,
    source_movie: object | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[str]]:
    logs: list[str] = []

    # ------------------------------------------------------------------
    # 0) Precedencia de título/año para reporting (Plex manda si está)
    # ------------------------------------------------------------------
    display_title_raw = movie_input.extra.get("display_title")
    display_year_raw = movie_input.extra.get("display_year")

    display_title = (
        display_title_raw
        if isinstance(display_title_raw, str) and display_title_raw.strip()
        else movie_input.title
    )

    display_year: int | None
    if isinstance(display_year_raw, int):
        display_year = display_year_raw
    else:
        display_year = movie_input.year

    # ------------------------------------------------------------------
    # 1) Fetch OMDb + Wiki (con caché global)
    # ------------------------------------------------------------------
    omdb_data: Mapping[str, object] | None = None
    wiki_meta: dict[str, object] = {}

    def _norm_imdb_hint() -> str | None:
        hint = movie_input.imdb_id_hint
        if isinstance(hint, str):
            h = hint.strip()
            return h or None
        return None

    def fetch_omdb(title_for_fetch: str, year_for_fetch: int | None) -> Mapping[str, object]:
        nonlocal omdb_data, wiki_meta

        imdb_hint = _norm_imdb_hint()
        key: _OmdbCacheKey = (title_for_fetch, year_for_fetch, imdb_hint)

        # ✅ 1) Cache local (módulo)
        cached = _OMDB_LOCAL_CACHE.get(key)
        if cached is not None:
            omdb_data = cached
            wiki_meta = _WIKI_LOCAL_CACHE.get(key, {})
            return cached

        # ✅ 2) OMDb (cache v2 del propio cliente)
        omdb_record = omdb_query_with_cache(
            title=title_for_fetch,
            year=year_for_fetch,
            imdb_id=imdb_hint,
        )

        if omdb_record is None:
            empty: dict[str, object] = {}
            _OMDB_LOCAL_CACHE[key] = empty
            _WIKI_LOCAL_CACHE[key] = {}
            omdb_data = empty
            wiki_meta = {}
            return empty

        omdb_dict = dict(omdb_record)

        # ✅ 3) Si OMDb YA trae __wiki, no llamamos a Wiki
        if "__wiki" in omdb_dict and isinstance(omdb_dict.get("__wiki"), Mapping):
            wiki_dict = _extract_wiki_meta(omdb_dict)
            _OMDB_LOCAL_CACHE[key] = omdb_dict
            _WIKI_LOCAL_CACHE[key] = wiki_dict
            omdb_data = omdb_dict
            wiki_meta = wiki_dict
            return omdb_dict

        # ✅ 4) Si ya teníamos wiki_cache local, tampoco llamamos
        cached_wiki = _WIKI_LOCAL_CACHE.get(key)
        if cached_wiki is not None and cached_wiki:
            omdb_dict["__wiki"] = dict(cached_wiki)
            _OMDB_LOCAL_CACHE[key] = omdb_dict
            omdb_data = omdb_dict
            wiki_meta = dict(cached_wiki)
            return omdb_dict

        # --- Wiki: intentamos con título+año (y con imdb si lo hay) ---
        imdb_id_from_omdb: str | None = None
        imdb_raw = omdb_dict.get("imdbID")
        if isinstance(imdb_raw, str) and imdb_raw.strip():
            imdb_id_from_omdb = imdb_raw.strip()

        wiki_item = get_wiki_client().get_wiki(
            title=title_for_fetch,
            year=year_for_fetch,
            imdb_id=imdb_id_from_omdb or imdb_hint,
        )

        if wiki_item is not None:
            wiki_block = wiki_item.get("wiki")
            wikidata_block = wiki_item.get("wikidata")

            merged_wiki: dict[str, object] = {}
            if isinstance(wiki_block, Mapping):
                merged_wiki.update(dict(wiki_block))
            if isinstance(wikidata_block, Mapping):
                merged_wiki["wikidata"] = dict(wikidata_block)

            imdb_cached = wiki_item.get("imdbID")
            if isinstance(imdb_cached, str) or imdb_cached is None:
                merged_wiki["imdb_id"] = imdb_cached

            omdb_dict["__wiki"] = merged_wiki

        wiki_dict = _extract_wiki_meta(omdb_dict)

        _OMDB_LOCAL_CACHE[key] = omdb_dict
        _WIKI_LOCAL_CACHE[key] = wiki_dict

        omdb_data = omdb_dict
        wiki_meta = wiki_dict
        return omdb_dict

    # ------------------------------------------------------------------
    # 2) Metacritic final (OMDb -> Wiki)
    # ------------------------------------------------------------------
    record_for_meta = fetch_omdb(movie_input.title, movie_input.year)
    metacritic_score = _safe_int(record_for_meta.get("Metascore"))
    if metacritic_score is None and wiki_meta:
        metacritic_score = _safe_int(wiki_meta.get("metacritic_score"))

    # ------------------------------------------------------------------
    # 3) Plex rating (si aplica)
    # ------------------------------------------------------------------
    plex_rating: float | None = None
    if movie_input.source == "plex" and source_movie is not None:
        plex_user_rating = getattr(source_movie, "userRating", None)
        plex_rating_raw = getattr(source_movie, "rating", None)
        plex_rating = _safe_float(plex_user_rating) or _safe_float(plex_rating_raw)

    # ------------------------------------------------------------------
    # 4) Core genérico: ratings + decisión + misidentified_hint (único)
    # ------------------------------------------------------------------
    try:
        base_row: AnalysisRow = analyze_input_movie(
            movie_input,
            fetch_omdb,
            plex_title=display_title,
            plex_year=display_year,
            plex_rating=plex_rating,
            metacritic_score=metacritic_score,
        )
    except Exception as exc:  # pragma: no cover
        msg = (
            f"[ERROR] {movie_input.library} / {display_title} ({display_year}): "
            f"fallo en core de análisis: {exc}"
        )
        _logger.error(msg)
        logs.append(msg)
        return None, None, logs

    if not base_row:
        logs.append(
            f"[WARN] {movie_input.library} / {display_title} ({display_year}): "
            "core de análisis devolvió fila vacía."
        )
        return None, None, logs

    misidentified_hint = base_row.get("misidentified_hint")
    if isinstance(misidentified_hint, str) and misidentified_hint:
        logs.append(
            f"[MISIDENTIFIED] {movie_input.library} / {display_title} ({display_year}): "
            f"{misidentified_hint}"
        )

    # ------------------------------------------------------------------
    # 5) Sugerencias de metadata (solo Plex) -> usamos MovieInput
    # ------------------------------------------------------------------
    omdb_dict: dict[str, object] = dict(omdb_data) if omdb_data else {}
    meta_sugg: dict[str, object] | None = None
    if movie_input.source == "plex" and source_movie is not None:
        try:
            meta_candidate = generate_metadata_suggestions_row(
                movie_input,
                omdb_dict or None,
            )
            if isinstance(meta_candidate, dict):
                meta_sugg = meta_candidate
                logs.append(
                    "[METADATA_SUGG] "
                    f"{movie_input.library} / {display_title} ({display_year}): "
                    f"{meta_sugg.get('suggestions_json', '')}"
                )
        except Exception as exc:  # pragma: no cover
            _logger.warning(
                f"generate_metadata_suggestions_row falló para {display_title!r}: {exc}"
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
        if isinstance(imdb_id_raw, str):
            imdb_id = imdb_id_raw

    imdb_hint = _norm_imdb_hint()
    if imdb_id is None and imdb_hint is not None:
        imdb_id = imdb_hint

    if omdb_dict:
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

    # Tamaño en disco (bytes) -> export como file_size
    file_size_bytes = row.get("file_size_bytes")
    if isinstance(file_size_bytes, int):
        row["file_size"] = file_size_bytes
    else:
        row["file_size"] = movie_input.file_size_bytes

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

    row["wikidata_id"] = wiki_meta.get("wikibase_item") or wiki_meta.get("wikidata_id")
    row["wikipedia_title"] = wiki_meta.get("wikipedia_title")
    row["source_language"] = wiki_meta.get("source_language")

    return row, meta_sugg, logs