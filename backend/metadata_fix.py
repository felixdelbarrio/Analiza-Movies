from __future__ import annotations

"""
backend/metadata_fix.py

Generación de sugerencias de corrección de metadata (Plex) basadas en OMDb.

Este módulo NO aplica cambios en Plex; solo construye una fila (row) que luego
se exporta a CSV (metadata_fix.csv) para revisión/aplicación posterior.

Regla de idioma (importante):
- Si el contexto es Español (ES), evitamos sugerir cambio de título cuando:
    * el título ACTUAL parece español, y
    * OMDb propone un título distinto (normalmente en inglés).
  Motivo: muchos catálogos en ES prefieren mantener el título localizado.
- Si el título actual ya parece inglés (aunque el contexto sea ES),
  permitimos sugerir new_title para corregir normalizaciones raras / inconsistencias.
- El cambio de año (new_year) se sugiere siempre que difiera (si ambos son parseables).

Filosofía de logs (alineada con backend/logger.py):
- SILENT_MODE=True:
    - No se emite ruido normal (info/warn).
    - En DEBUG_MODE=True se permiten trazas mínimas y útiles (progress).
- SILENT_MODE=False:
    - info/warn/error visibles de forma normal.
    - DEBUG_MODE=True añade mensajes de diagnóstico (info con prefijo).
"""

import json
from collections.abc import Mapping

from backend import logger as _logger
from backend.movie_input import MovieInput, normalize_title_for_lookup
from backend.movie_input import guess_spanish_from_title_or_path


# ============================================================================
# Logging controlado por modos (sin forzar imports circulares)
# ============================================================================

def _safe_get_cfg():
    """Devuelve backend.config si ya está importado (evita dependencias circulares)."""
    import sys
    return sys.modules.get("backend.config")


def _is_silent_mode() -> bool:
    cfg = _safe_get_cfg()
    if cfg is None:
        return False
    try:
        return bool(getattr(cfg, "SILENT_MODE", False))
    except Exception:
        return False


def _is_debug_mode() -> bool:
    cfg = _safe_get_cfg()
    if cfg is None:
        return False
    try:
        return bool(getattr(cfg, "DEBUG_MODE", False))
    except Exception:
        return False


def _log_info(msg: str) -> None:
    """Info normal: respetando SILENT_MODE."""
    if _is_silent_mode():
        return
    try:
        _logger.info(msg)
    except Exception:
        print(msg)


def _log_warning(msg: str) -> None:
    """Warning normal: respetando SILENT_MODE."""
    if _is_silent_mode():
        return
    try:
        _logger.warning(msg)
    except Exception:
        print(msg)


def _log_error(msg: str) -> None:
    """Error: en tu logger suele ser siempre visible, pero aquí no forzamos always."""
    try:
        _logger.error(msg)
    except Exception:
        print(msg)


def _log_debug(msg: str) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False → no hace nada.
    - DEBUG_MODE=True:
        * SILENT_MODE=True: progress (señales mínimas).
        * SILENT_MODE=False: info normal.
    """
    if not _is_debug_mode():
        return

    text = str(msg)
    try:
        if _is_silent_mode():
            _logger.progress(f"[METADATA][DEBUG] {text}")
        else:
            _logger.info(f"[METADATA][DEBUG] {text}")
    except Exception:
        if not _is_silent_mode():
            print(text)


# ============================================================================
# Helpers
# ============================================================================

def _normalize_year(year: object | None) -> int | None:
    """
    Normaliza un año posible (int/str) a int o None.
    """
    if year is None:
        return None
    try:
        return int(str(year).strip())
    except (TypeError, ValueError):
        return None


def _is_spanish_context(movie_input: MovieInput) -> bool:
    """
    Determina si el contexto del item sugiere "catálogo en Español".

    Prioridad:
    1) Plex: movie_input.extra["library_language"] (si existe).
    2) Heurística por título/path (DLNA/local/otros).

    Returns:
        True si el contexto parece ES, False en caso contrario.
    """
    lang = movie_input.extra.get("library_language")
    if isinstance(lang, str) and lang.strip():
        l = lang.strip().lower()
        return bool(l.startswith("es") or l.startswith("spa"))

    return guess_spanish_from_title_or_path(movie_input.title, movie_input.file_path)


# ============================================================================
# API pública
# ============================================================================

def generate_metadata_suggestions_row(
    movie_input: MovieInput,
    omdb_data: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """
    Genera una fila de sugerencias de metadata para Plex.

    Regla:
      - En contexto ES:
          * si el título actual parece español, NO sugerimos new_title
            aunque OMDb difiera.
          * si el título actual ya parece inglés, sí permitimos sugerir new_title.
      - new_year se sugiere si difiere y ambos años son válidos.

    Args:
        movie_input: entrada unificada (Plex/DLNA/etc.). En Plex se usa extra display_*.
        omdb_data: payload OMDb (Mapping) o None.

    Returns:
        dict listo para CSV (metadata_fix.csv) o None si no hay cambios sugeribles.
    """
    if not omdb_data:
        return None

    # En Plex queremos comparar contra lo que ve el usuario:
    plex_title = movie_input.extra.get("display_title") or movie_input.title
    plex_year = movie_input.extra.get("display_year") or movie_input.year

    plex_title_str = plex_title if isinstance(plex_title, str) else movie_input.title
    plex_year_val = plex_year if isinstance(plex_year, int) else movie_input.year

    library = movie_input.library
    plex_guid = movie_input.plex_guid

    omdb_title_obj = omdb_data.get("Title")
    omdb_year_obj = omdb_data.get("Year")

    omdb_title = omdb_title_obj if isinstance(omdb_title_obj, str) else None

    # Normalización para comparar "equivalencia" (sin tildes/puntuación/espacios raros)
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

    # Regla idioma (fina)
    # En contexto ES, bloqueamos new_title solo si el título actual parece español.
    if _is_spanish_context(movie_input) and title_diff and omdb_title:
        title_current = plex_title_str or ""
        if guess_spanish_from_title_or_path(title_current, movie_input.file_path):
            _log_debug(
                "Skip new_title suggestion (ES context + ES title) | "
                f"{library} / {plex_title_str!r} -> {omdb_title!r}"
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
        "suggestions_json": json.dumps(suggestions, ensure_ascii=False, separators=(",", ":")),
    }

    _log_debug(f"Generated metadata suggestion | {library} / {plex_title_str}: {suggestions}")
    return row