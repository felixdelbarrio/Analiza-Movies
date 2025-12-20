from __future__ import annotations

import json
from collections.abc import Mapping

from backend import logger as _logger
from backend.config import METADATA_DRY_RUN, METADATA_APPLY_CHANGES, SILENT_MODE
from backend.movie_input import MovieInput, normalize_title_for_lookup
from backend.movie_input import guess_spanish_from_title_or_path, is_probably_english_title


def _log_info(msg: str) -> None:
    if SILENT_MODE:
        return
    try:
        _logger.info(msg)
    except Exception:
        print(msg)


def _log_debug(msg: str) -> None:
    if SILENT_MODE:
        return
    try:
        _logger.debug(msg)
    except Exception:
        pass


def _log_warning(msg: str) -> None:
    if SILENT_MODE:
        return
    try:
        _logger.warning(msg)
    except Exception:
        print(msg)


def _log_error(msg: str) -> None:
    if SILENT_MODE:
        return
    try:
        _logger.error(msg)
    except Exception:
        print(msg)


def _normalize_year(year: object | None) -> int | None:
    if year is None:
        return None
    try:
        return int(str(year))
    except (TypeError, ValueError):
        return None


def _is_spanish_context(movie_input: MovieInput) -> bool:
    """
    - Plex: usa movie_input.extra["library_language"] si está.
    - DLNA/local: heurística por título/path.
    """
    lang = movie_input.extra.get("library_language")
    if isinstance(lang, str) and lang.strip():
        l = lang.strip().lower()
        if l.startswith("es") or l.startswith("spa"):
            return True
        return False

    # Heurística (DLNA/UPnP/local/other)
    return guess_spanish_from_title_or_path(movie_input.title, movie_input.file_path)


def generate_metadata_suggestions_row(
    movie_input: MovieInput,
    omdb_data: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """
    Regla:
      - Si contexto ES -> solo bloqueamos new_title cuando el título actual
        (plex_title_str) parece español. Si el título actual ya parece inglés,
        permitimos sugerir new_title (para corregir normalizaciones raras).
      - new_year se sugiere igual si difiere.
    """
    if not omdb_data:
        return None

    plex_title = movie_input.extra.get("display_title") or movie_input.title
    plex_year = movie_input.extra.get("display_year") or movie_input.year

    plex_title_str = plex_title if isinstance(plex_title, str) else movie_input.title
    plex_year_val = plex_year if isinstance(plex_year, int) else movie_input.year

    library = movie_input.library
    plex_guid = movie_input.plex_guid

    omdb_title_obj = omdb_data.get("Title")
    omdb_year_obj = omdb_data.get("Year")
    omdb_title = omdb_title_obj if isinstance(omdb_title_obj, str) else None

    n_plex_title = normalize_title_for_lookup(plex_title_str) if plex_title_str else ""
    n_omdb_title = normalize_title_for_lookup(omdb_title) if omdb_title else ""

    n_plex_year = _normalize_year(plex_year_val)
    n_omdb_year = _normalize_year(omdb_year_obj)

    title_diff = bool(n_plex_title and n_omdb_title and n_plex_title != n_omdb_title)
    year_diff = (
        n_plex_year is not None
        and n_omdb_year is not None
        and n_plex_year != n_omdb_year
    )

    if not title_diff and not year_diff:
        return None

    # ✅ Regla idioma (fina):
    # En contexto ES, solo bloqueamos cambio de título si el título ACTUAL parece español.
    # Si el título ya parece inglés, permitimos sugerir new_title.
    if _is_spanish_context(movie_input) and title_diff and omdb_title:
        title_current = plex_title_str or ""
        if guess_spanish_from_title_or_path(title_current, movie_input.file_path):
            _log_debug(
                f"Skip new_title suggestion (ES context + ES title) for {library} / {plex_title_str!r} -> {omdb_title!r}"
            )
            title_diff = False

    suggestions: dict[str, object] = {}
    if title_diff and omdb_title is not None:
        suggestions["new_title"] = omdb_title
    if year_diff:
        suggestions["new_year"] = n_omdb_year

    if not suggestions:
        return None

    if "new_title" in suggestions and "new_year" in suggestions:
        action = "Fix title & year"
    elif "new_title" in suggestions:
        action = "Fix title"
    else:
        action = "Fix year"

    row: dict[str, object] = {
        "plex_guid": plex_guid,
        "library": library,
        "plex_title": plex_title_str,
        "plex_year": plex_year_val,
        "omdb_title": omdb_title_obj,
        "omdb_year": omdb_year_obj,
        "action": action,
        "suggestions_json": json.dumps(
            suggestions, ensure_ascii=False, separators=(",", ":")
        ),
    }

    _log_debug(
        f"Generated metadata suggestion for {library} / {plex_title_str}: {suggestions}"
    )
    return row