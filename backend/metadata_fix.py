from __future__ import annotations

"""
backend/metadata_fix.py

Generación de sugerencias de corrección de metadata (Plex) basadas en OMDb.

Este módulo **NO aplica cambios en Plex**. Solo construye una fila (row) que luego
se exporta a CSV (metadata_fix.csv) para revisión / aplicación posterior.

✅ Mejora aplicada: “consumiendo helpers centralizados”
------------------------------------------------------
Este módulo NO implementa heurísticas de idioma por su cuenta.
Consume helpers centralizados en movie_input.py:

- normalize_title_for_lookup(...)
- detect_context_language_code(...)
- should_skip_new_title_suggestion(...)

Con esto:
- Evitamos duplicación.
- Mantenemos una política consistente en todo el proyecto.
- La regla multi-idioma está en un único sitio.

🪵 Logs 100% alineados con backend/logger.py
-------------------------------------------
- No hacemos “policy propia” ni introspección de backend.config.
- Usamos el logger central:
    - _logger.debug_ctx("METADATA", ...) para debug contextual.
    - _logger.info / _logger.warning / _logger.error cuando procede.
- Si DEBUG_MODE es False, debug_ctx ya será no-op (según vuestra implementación).
- En SILENT_MODE el logger central ya decide qué imprimir.

Tipado
------
- Sin Any explícito.
- Salida: dict[str, object] listo para CSV, o None si no hay sugerencias.
"""

import json
from collections.abc import Mapping

from backend import logger as _logger
from backend.movie_input import (
    MovieInput,
    detect_context_language_code,
    normalize_title_for_lookup,
    should_skip_new_title_suggestion,
)

# ============================================================================
# Helpers defensivos
# ============================================================================


def _normalize_year(year: object | None) -> int | None:
    """
    Normaliza un año posible (int/str) a int o None.

    OMDb Year puede venir como:
      - "1994"
      - "1994–1998"
      - "N/A"

    Aceptamos rango razonable para cine (1800..2200) por robustez.
    """
    if year is None:
        return None

    try:
        s = str(year).strip()
    except Exception:
        return None

    if not s or s.upper() == "N/A":
        return None

    # "1994–1998" / "1994-1998" -> 1994
    if len(s) >= 4 and s[:4].isdigit():
        y = int(s[:4])
        return y if 1800 <= y <= 2200 else None

    # fallback: entero completo
    try:
        y2 = int(s)
        return y2 if 1800 <= y2 <= 2200 else None
    except Exception:
        return None


def _get_display_title_year(movie_input: MovieInput) -> tuple[str, object | None]:
    """
    En Plex queremos comparar contra lo que “ve el usuario”:
    - extra.display_title / extra.display_year si están
    - si no, movie_input.title / movie_input.year
    """
    dt = movie_input.extra.get("display_title")
    dy = movie_input.extra.get("display_year")

    title = dt if isinstance(dt, str) and dt.strip() else (movie_input.title or "")
    year_obj: object | None = dy if dy is not None else movie_input.year
    return title, year_obj


# ============================================================================
# API pública
# ============================================================================


def generate_metadata_suggestions_row(
    movie_input: MovieInput,
    omdb_data: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """
    Genera una fila de sugerencias de metadata para Plex.

    Reglas:
      - new_title se sugiere si difiere, salvo bloqueo por política multi-idioma:
          should_skip_new_title_suggestion(...)
      - new_year se sugiere si difiere y ambos años son válidos.

    Args:
        movie_input: entrada unificada (Plex/DLNA/etc.). En Plex se usa extra display_*.
        omdb_data: payload OMDb (Mapping) o None.

    Returns:
        dict listo para CSV (metadata_fix.csv) o None si no hay cambios sugeribles.
    """
    if not omdb_data:
        return None

    plex_title, plex_year_obj = _get_display_title_year(movie_input)
    library = movie_input.library
    plex_guid = movie_input.plex_guid

    omdb_title_obj = omdb_data.get("Title")
    omdb_year_obj = omdb_data.get("Year")

    omdb_title = omdb_title_obj if isinstance(omdb_title_obj, str) else ""

    # Normalización para comparar equivalencia (misma convención del proyecto)
    n_plex_title = normalize_title_for_lookup(plex_title) if plex_title else ""
    n_omdb_title = normalize_title_for_lookup(omdb_title) if omdb_title else ""

    n_plex_year = _normalize_year(plex_year_obj)
    n_omdb_year = _normalize_year(omdb_year_obj)

    title_diff = bool(n_plex_title and n_omdb_title and n_plex_title != n_omdb_title)
    year_diff = bool(n_plex_year is not None and n_omdb_year is not None and n_plex_year != n_omdb_year)

    if not title_diff and not year_diff:
        return None

    # Política multi-idioma: NO sugerimos "des-localizar" títulos por accidente.
    if title_diff and omdb_title:
        ctx_lang = detect_context_language_code(movie_input)
        if should_skip_new_title_suggestion(
            context_lang=ctx_lang,
            current_title=plex_title,
            omdb_title=omdb_title,
        ):
            _logger.debug_ctx(
                "METADATA",
                "Skip new_title suggestion (localized context/title vs OMDb) | "
                f"ctx_lang={ctx_lang} | {library} / {plex_title!r} -> {omdb_title!r}",
            )
            title_diff = False

    suggestions: dict[str, object] = {}
    if title_diff and omdb_title:
        suggestions["new_title"] = omdb_title
    if year_diff and n_omdb_year is not None:
        suggestions["new_year"] = n_omdb_year

    if not suggestions:
        return None

    # Acción humana para CSV
    if "new_title" in suggestions and "new_year" in suggestions:
        action = "Fix title & year"
    elif "new_title" in suggestions:
        action = "Fix title"
    else:
        action = "Fix year"

    row: dict[str, object] = {
        "plex_guid": plex_guid,
        "library": library,
        "plex_title": plex_title,
        "plex_year": plex_year_obj,  # valor original (humano)
        "omdb_title": omdb_title_obj,
        "omdb_year": omdb_year_obj,
        "action": action,
        "suggestions_json": json.dumps(suggestions, ensure_ascii=False, separators=(",", ":")),
    }

    _logger.debug_ctx("METADATA", f"Generated metadata suggestion | {library} / {plex_title}: {suggestions}")
    return row