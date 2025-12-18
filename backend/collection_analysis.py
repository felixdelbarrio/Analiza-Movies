from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeAlias

from backend import logger as _logger
from backend.analyze_input_core import AnalysisRow, analyze_input_movie
from backend.metadata_fix import generate_metadata_suggestions_row
from backend.movie_input import MovieInput
from backend.wiki_client import get_movie_record


_OmdbCacheKey: TypeAlias = tuple[str, int | None, str | None]


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
    """Pipeline único de análisis para una película.

    Devuelve: (row_dict, metadata_suggestion_row_or_None, logs)

    - Homogeneiza el flujo para Plex y DLNA.
    - Precedencia de datos:
        1) Datos nativos de Plex (si existen) prevalecen en título/año, ids y rating.
        2) OMDb (Metascore, Poster, Website, imdbID).
        3) Wiki/Wikidata (fallback de metacritic_score y ids).

    Importante:
    - `misidentified_hint` se calcula UNA sola vez en `analyze_input_movie`,
      usando `plex_title/plex_year` (precedencia Plex cuando aplica).
      Aquí no se recalcula ni se sobrescribe.
    """
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
    # 1) Fetch OMDb/Wiki (unificado) + caché local por iteración
    # ------------------------------------------------------------------
    omdb_data: Mapping[str, object] | None = None
    wiki_meta: dict[str, object] = {}

    omdb_cache: dict[_OmdbCacheKey, dict[str, object]] = {}
    wiki_cache: dict[_OmdbCacheKey, dict[str, object]] = {}

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

        cached = omdb_cache.get(key)
        if cached is not None:
            omdb_data = cached
            wiki_meta = wiki_cache.get(key, {})
            return cached

        record = get_movie_record(
            title=title_for_fetch,
            year=year_for_fetch,
            imdb_id_hint=imdb_hint,
        )

        if record is None:
            empty: dict[str, object] = {}
            omdb_cache[key] = empty
            wiki_cache[key] = {}
            omdb_data = empty
            wiki_meta = {}
            return empty

        omdb_dict = dict(record) if isinstance(record, Mapping) else {}
        wiki_dict = _extract_wiki_meta(omdb_dict)

        omdb_cache[key] = omdb_dict
        wiki_cache[key] = wiki_dict

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
    # 5) Sugerencias de metadata (solo Plex)
    # ------------------------------------------------------------------
    omdb_dict: dict[str, object] = dict(omdb_data) if omdb_data else {}
    meta_sugg: dict[str, object] | None = None
    if movie_input.source == "plex" and source_movie is not None:
        try:
            meta_candidate = generate_metadata_suggestions_row(
                source_movie,
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

    # file_size_bytes -> file_size (compat dashboard)
    file_size_bytes = row.get("file_size_bytes")
    if isinstance(file_size_bytes, int):
        row["file_size"] = file_size_bytes
    else:
        row["file_size"] = movie_input.file_size_bytes

    row["file"] = movie_input.file_path or row.get("file", "")

    # NUEVO: URL real del origen (DLNA/UPnP, etc.)
    source_url_obj = movie_input.extra.get("source_url")
    row["source_url"] = source_url_obj if isinstance(source_url_obj, str) else ""

    # Campos Plex
    row["rating_key"] = movie_input.rating_key
    row["guid"] = movie_input.plex_guid
    row["thumb"] = movie_input.thumb_url

    # Campos externos
    row["imdb_id"] = imdb_id
    row["poster_url"] = poster_url
    row["trailer_url"] = trailer_url
    row["omdb_json"] = omdb_json_str

    row["wikidata_id"] = wiki_meta.get("wikidata_id")
    row["wikipedia_title"] = wiki_meta.get("wikipedia_title")

    # Importante: NO se recalcula ni se sobrescribe misidentified_hint aquí.

    return row, meta_sugg, logs