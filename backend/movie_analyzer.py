from __future__ import annotations

"""
Funciones para analizar una única película de Plex y construir la fila final
que será consumida por `reporting` y el dashboard.

Objetivo:
- Mantener la API pública: analyze_single_movie(movie).
- Delegar el núcleo de decisión (ratings → KEEP/DELETE/...) en el core genérico
  que usa scoring.decide_action + decision_logic.detect_misidentified.
- Enriquecer la fila para Plex con metadata extra (wiki, sugerencias, etc.).
"""

import json
from collections.abc import Mapping

from backend import logger as _logger
from backend.movie_input import MovieInput
from backend.analyze_input_core import AnalysisRow, analyze_input_movie
from backend.decision_logic import detect_misidentified
from backend.metadata_fix import generate_metadata_suggestions_row
from backend.plex_client import (
    get_best_search_title,
    get_imdb_id_from_movie,
    get_movie_file_info,
)
from backend.scoring import decide_action
from backend.wiki_client import get_movie_record


def _safe_int(value: object) -> int | None:
    """Conversión defensiva a int. Devuelve None si no es convertible."""
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


def analyze_single_movie(
    movie: object,
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[str]]:
    """Analiza un objeto `movie` devuelto por la API de Plex.

    Retorna una tupla: (row_dict, metadata_suggestion_row_or_None, logs).

    - `row_dict` es un dict con claves usadas por `reporting` y el dashboard.
      Si no hay datos o hay un error grave, puede ser None.
    - `metadata_suggestion_row_or_None` es el resultado de
      `generate_metadata_suggestions_row` (o None si no aplica).
    - `logs` es una lista de cadenas con anotaciones útiles para debugging.
    """
    logs: list[str] = []

    library = getattr(movie, "librarySectionTitle", "") or ""
    title = getattr(movie, "title", "") or ""

    year_value = getattr(movie, "year", None)
    year: int | None = year_value if isinstance(year_value, int) else None

    rating_key_raw = getattr(movie, "ratingKey", None)
    rating_key: str | None = (
        str(rating_key_raw) if rating_key_raw is not None else None
    )

    guid = getattr(movie, "guid", None)
    thumb = getattr(movie, "thumb", None)

    # 1) Info de archivo (ruta + tamaño)
    file_path, file_size = get_movie_file_info(movie)

    # 2) IMDb ID inicial y mejor título de búsqueda
    imdb_id_hint = get_imdb_id_from_movie(movie)
    search_title = get_best_search_title(movie) or title

    # 3) Preparar acceso a OMDb + Wikipedia/Wikidata mediante wiki_client
    omdb_data: Mapping[str, object] | None = None
    wiki_meta: dict[str, object] = {}

    def fetch_omdb(title_for_fetch: str, year_for_fetch: int | None) -> Mapping[str, object]:
        nonlocal omdb_data, wiki_meta

        record = get_movie_record(
            title=title_for_fetch,
            year=year_for_fetch,
            imdb_id_hint=imdb_id_hint,
        )

        if record is None:
            omdb_data = {}
            wiki_meta = {}
            _logger.info(
                f"[OMDb/WIKI] Sin datos para "
                f"{library} / {title_for_fetch} ({year_for_fetch})"
            )
            return {}

        if isinstance(record, Mapping):
            omdb_data = record
        else:
            omdb_data = dict(record)  # type: ignore[arg-type]

        wiki_raw = omdb_data.get("__wiki")
        if isinstance(wiki_raw, Mapping):
            wiki_meta = dict(wiki_raw)
        else:
            wiki_meta = {}

        return omdb_data

    # 4) Construcción de MovieInput para el core genérico
    try:
        dlna_input = MovieInput(
            source="plex",
            library=library,
            title=search_title,
            year=year,
            file_path=file_path or "",
            file_size_bytes=file_size,
            imdb_id_hint=imdb_id_hint,
            plex_guid=guid,
            rating_key=rating_key,
            thumb_url=thumb,
            extra={},
        )
    except Exception as exc:
        msg = (
            f"[ERROR] {library} / {title} ({year}): "
            f"fallo creando MovieInput: {exc}"
        )
        _logger.error(msg)
        logs.append(msg)
        return None, None, logs

    # 5) Núcleo de análisis (core genérico)
    try:
        base_row: AnalysisRow = analyze_input_movie(dlna_input, fetch_omdb)
    except Exception as exc:  # pragma: no cover (defensivo)
        msg = (
            f"[ERROR] {library} / {title} ({year}): "
            f"fallo en core de análisis: {exc}"
        )
        _logger.error(msg)
        logs.append(msg)
        return None, None, logs

    if not base_row:
        logs.append(
            f"[WARN] {library} / {title} ({year}): "
            "core de análisis devolvió fila vacía."
        )
        return None, None, logs

    # 6) Ratings + Metacritic + decisión final (incluyendo Metacritic)
    imdb_rating = base_row.get("imdb_rating")
    imdb_votes = base_row.get("imdb_votes")
    rt_score = base_row.get("rt_score")

    metacritic_score: int | None = None
    if omdb_data:
        metacritic_score = _safe_int(omdb_data.get("Metascore"))
    if metacritic_score is None and wiki_meta:
        metacritic_score = _safe_int(wiki_meta.get("metacritic_score"))

    decision, reason = decide_action(
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        year=year,
        metacritic_score=metacritic_score,
    )

    # 7) Rating Plex (userRating > rating)
    plex_user_rating = getattr(movie, "userRating", None)
    plex_rating_raw = getattr(movie, "rating", None)
    plex_rating: float | None = None
    if isinstance(plex_user_rating, (int, float)):
        plex_rating = float(plex_user_rating)
    elif isinstance(plex_rating_raw, (int, float)):
        plex_rating = float(plex_rating_raw)

    # 8) Misidentificación + sugerencias de metadata
    omdb_dict: dict[str, object] = dict(omdb_data) if omdb_data else {}

    misidentified_hint = detect_misidentified(
        plex_title=title,
        plex_year=year,
        omdb_data=omdb_dict or None,
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
    )

    if misidentified_hint:
        logs.append(
            f"[MISIDENTIFIED] {library} / {title} ({year}): "
            f"{misidentified_hint}"
        )

    meta_sugg: dict[str, object] | None = None
    try:
        meta_candidate = generate_metadata_suggestions_row(
            movie,
            omdb_dict or None,
        )
        if isinstance(meta_candidate, dict):
            meta_sugg = meta_candidate
            logs.append(
                "[METADATA_SUGG] "
                f"{library} / {title} ({year}): "
                f"{meta_sugg.get('suggestions_json', '')}"
            )
    except Exception as exc:  # pragma: no cover (defensivo)
        _logger.warning(
            f"generate_metadata_suggestions_row falló para {title!r}: {exc}"
        )

    # 9) Enriquecimiento con datos OMDb/Wiki para reporting
    poster_url: str | None = None
    trailer_url: str | None = None
    imdb_id: str | None = None

    if omdb_dict:
        poster_raw = omdb_dict.get("Poster")
        trailer_raw = omdb_dict.get("Website")
        imdb_id_raw = omdb_dict.get("imdbID")

        poster_url = poster_raw if isinstance(poster_raw, str) else None
        trailer_url = trailer_raw if isinstance(trailer_raw, str) else None
        if isinstance(imdb_id_raw, str):
            imdb_id = imdb_id_raw

    if imdb_id is None and isinstance(imdb_id_hint, str):
        imdb_id = imdb_id_hint

    omdb_json_str: str | None = None
    if omdb_dict:
        try:
            omdb_json_str = json.dumps(omdb_dict, ensure_ascii=False)
        except Exception:
            omdb_json_str = str(omdb_dict)

    if wiki_meta:
        logs.append(
            "[WIKI] Enriquecido desde Wikipedia/Wikidata: "
            f"wikidata_id={wiki_meta.get('wikidata_id')}, "
            f"wikipedia_title={wiki_meta.get('wikipedia_title')!r}"
        )

    # 10) Construcción de la fila final, partiendo de base_row
    row: dict[str, object] = dict(base_row)

    # Normalización de columnas y enriquecimiento
    row["source"] = "plex"
    row["library"] = library
    row["title"] = title
    row["year"] = year

    row["imdb_rating"] = imdb_rating
    row["imdb_votes"] = imdb_votes
    row["rt_score"] = rt_score
    row["plex_rating"] = plex_rating

    row["decision"] = decision
    row["reason"] = reason
    row["misidentified_hint"] = misidentified_hint

    # file / file_size: adaptamos file_size_bytes → file_size por compatibilidad
    file_size_bytes = row.get("file_size_bytes")
    if isinstance(file_size_bytes, int):
        row["file_size"] = file_size_bytes
    else:
        row["file_size"] = file_size

    row["file"] = file_path or row.get("file", "")

    row["rating_key"] = rating_key
    row["guid"] = guid
    row["imdb_id"] = imdb_id
    row["poster_url"] = poster_url
    row["trailer_url"] = trailer_url
    row["thumb"] = thumb
    row["omdb_json"] = omdb_json_str
    row["wikidata_id"] = wiki_meta.get("wikidata_id")
    row["wikipedia_title"] = wiki_meta.get("wikipedia_title")

    return row, meta_sugg, logs