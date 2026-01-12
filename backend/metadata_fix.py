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
- coalesce_movie_identity(...)  ✅ (refactor: mejor guardrail imdb_id)

Con esto:
- Evitamos duplicación.
- Mantenemos una política consistente en todo el proyecto.
- La regla multi-idioma está en un único sitio.

🪵 Logs 100% alineados con backend/logger.py
-------------------------------------------
- Usamos el logger central:
    - logger.debug_ctx("METADATA", ...) para debug contextual.
    - logger.warning / logger.error solo cuando procede.
- No imprimimos nada “por item” en modo normal.
- En SILENT_MODE/DEBUG_MODE la política la impone backend/logger.py.

Tipado
------
- Sin Any explícito.
- Salida: dict[str, object] listo para CSV, o None si no hay sugerencias.

──────────────────────────────────────────────────────────────────────────────
CAMBIOS / MEJORAS EN ESTA REVISION
──────────────────────────────────────────────────────────────────────────────
- FIX: comparación de títulos más robusta:
    - si normalize_title_for_lookup devuelve "", caemos a comparación básica
      (casefold/strip) para evitar falsos negativos.
- FIX: normalize_year soporta mejor guiones unicode (“1994–1998”, “1994—1998”)
  y cadenas con ruido.
- Robustez: si faltan plex_guid/library, no crashea; devuelve strings vacías.
- Logging: debug_ctx más informativo (incluye títulos/years normalizados),
  sin introducir logs ruidosos en modos no-debug.

- NUEVO (guardrail anti-MISIDENTIFIED):
    - Si existe imdb_id_hint en Plex y OMDb devuelve imdbID distinto => NO sugerimos cambios
      destructivos (título/año). Emitimos fila "Skip (IMDb mismatch)" para revisión.

- NUEVO (utilidad multi-idioma):
    - Si el contexto NO es EN y el título OMDb parece estar “en otro idioma” (típicamente EN),
      NO sugerimos new_title. En su lugar registramos alt_title (informativo) para revisión.
    - Añadimos un fallback "loose" para detectar títulos ingleses sin function-words.
    - Si plex_original_title coincide con OMDb Title, no sugerimos cambio de título.

- NUEVO (CSV más útil, alineado con reporting.py opción A):
    - Añade columnas: context_lang, plex_original_title, plex_imdb_id, omdb_imdb_id,
      imdb_rating, imdb_votes y action.

- FIX (coherencia con reporting.py revisado):
    - Aseguramos que SIEMPRE se emite la key "action" cuando se devuelve una fila.

- REFAC (alineado con collection_analysis/analyze_input_core):
    - Usa coalesce_movie_identity(...) para derivar plex_imdb_id más fiable
      (apoyándose en file_path + extra["source_url"]).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping

from backend import logger
from backend.movie_input import (
    MovieInput,
    coalesce_movie_identity,
    detect_context_language_code,
    normalize_title_for_lookup,
    should_skip_new_title_suggestion,
)

# ============================================================================
# Helpers defensivos
# ============================================================================

_YEAR_PREFIX_RE = re.compile(r"^\s*(\d{4})")

# “ASCII title” loose: letras/dígitos/espacios y puntuación típica, sin diacríticos.
_ASCII_TITLE_SAFE_RE = re.compile(r"^[0-9A-Za-z\s\-\:\'\!\?\.\,\&\(\)\/]+$")


def _normalize_year(year: object | None) -> int | None:
    """
    Normaliza un año posible (int/str) a int o None.

    OMDb Year puede venir como:
      - "1994"
      - "1994–1998"
      - "1994-1998"
      - "N/A"

    Aceptamos rango razonable para cine (1800..2200) por robustez.
    """
    if year is None:
        return None

    if isinstance(year, int):
        return year if 1800 <= year <= 2200 else None

    try:
        s = str(year).strip()
    except Exception:
        return None

    if not s or s.upper() == "N/A":
        return None

    s_norm = unicodedata.normalize("NFKC", s)
    m = _YEAR_PREFIX_RE.match(s_norm)
    if m:
        try:
            y = int(m.group(1))
            return y if 1800 <= y <= 2200 else None
        except Exception:
            return None

    try:
        y2 = int(s_norm)
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


def _fallback_norm_title_basic(text: str) -> str:
    try:
        return unicodedata.normalize("NFKC", (text or "")).strip().casefold()
    except Exception:
        return (text or "").strip().lower()


def _titles_differ(plex_title: str, omdb_title: str) -> bool:
    n_plex = normalize_title_for_lookup(plex_title) if plex_title else ""
    n_omdb = normalize_title_for_lookup(omdb_title) if omdb_title else ""
    if n_plex and n_omdb:
        return n_plex != n_omdb

    b_plex = _fallback_norm_title_basic(plex_title)
    b_omdb = _fallback_norm_title_basic(omdb_title)
    return bool(b_plex and b_omdb and b_plex != b_omdb)


def _norm_lookup(text: str) -> str:
    try:
        n = normalize_title_for_lookup(text or "")
        return n or _fallback_norm_title_basic(text)
    except Exception:
        return _fallback_norm_title_basic(text)


def _is_ascii_like_title(text: str) -> bool:
    """
    Heurística “loose”:
    True si el título usa solo ASCII típico (sin diacríticos/CJK).
    Útil para bibliotecas ES/IT/FR donde OMDb suele devolver el título canónico (EN)
    aunque no contenga function-words.
    """
    t = (text or "").strip()
    if not t:
        return False
    t2 = unicodedata.normalize("NFKC", t)
    return bool(_ASCII_TITLE_SAFE_RE.match(t2))


def _coerce_str(value: object | None) -> str:
    """
    Helper defensivo para evitar AttributeError en .strip() si value no es str.
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_source_url(movie_input: MovieInput) -> str:
    extra = getattr(movie_input, "extra", {}) or {}
    if isinstance(extra, dict):
        v = extra.get("source_url")
        if isinstance(v, str):
            return v
    return ""


def _coalesce_plex_imdb_id(movie_input: MovieInput) -> str:
    """
    Devuelve un imdb id más robusto (si se puede inferir desde path/source_url),
    y si no, cae al imdb_id_hint tal cual.
    """
    try:
        source_url = _extract_source_url(movie_input)
        hint = f"{movie_input.file_path or ''} {source_url}".strip()

        _t2, _y2, imdb2 = coalesce_movie_identity(
            title=movie_input.title or "",
            year=movie_input.year,
            file_path=hint,
            imdb_id_hint=getattr(movie_input, "imdb_id_hint", None),
        )
        imdb2s = _coerce_str(imdb2)
        if imdb2s:
            return imdb2s
    except Exception:
        pass

    return _coerce_str(getattr(movie_input, "imdb_id_hint", None))


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
      - Si hay imdb_id_hint en Plex y OMDb devuelve imdbID distinto => NO sugerimos cambios
        (posible MISIDENTIFIED). Emitimos fila "Skip (IMDb mismatch)" para revisión.
      - new_title se sugiere si difiere, salvo bloqueo por política multi-idioma.
        Si se bloquea (contexto no EN), añadimos alt_title (informativo) en vez de new_title.
      - new_year se sugiere si difiere y ambos años son válidos.

    Returns:
        dict listo para CSV (metadata_fix.csv) o None si no hay cambios sugeribles.
    """
    if not omdb_data:
        return None

    plex_title, plex_year_obj = _get_display_title_year(movie_input)

    library_obj = getattr(movie_input, "library", "")
    library = library_obj if isinstance(library_obj, str) else ""

    plex_guid_obj = getattr(movie_input, "plex_guid", "")
    plex_guid = plex_guid_obj if isinstance(plex_guid_obj, str) else ""

    ctx_lang = detect_context_language_code(movie_input)

    plex_original_title_raw = movie_input.extra.get("plex_original_title")
    plex_original_title = _coerce_str(plex_original_title_raw)

    # IMDb ids (para guardrail)  ✅ ahora coalesce también
    plex_imdb_id = _coalesce_plex_imdb_id(movie_input)
    omdb_imdb_id_obj = omdb_data.get("imdbID")
    omdb_imdb_id = _coerce_str(omdb_imdb_id_obj)

    # Extras OMDb útiles para auditar confianza
    imdb_rating_obj = omdb_data.get("imdbRating")
    imdb_votes_obj = omdb_data.get("imdbVotes")

    omdb_title_obj = omdb_data.get("Title")
    omdb_year_obj = omdb_data.get("Year")
    omdb_title = omdb_title_obj if isinstance(omdb_title_obj, str) else ""

    # ------------------------------------------------------------------
    # Guardrail: IMDb mismatch => no sugerir cambios destructivos
    # ------------------------------------------------------------------
    if plex_imdb_id and omdb_imdb_id and plex_imdb_id != omdb_imdb_id:
        logger.warning(
            "METADATA: Skip suggestions due to IMDb mismatch | "
            f"{library} / plex_imdb={plex_imdb_id} omdb_imdb={omdb_imdb_id} "
            f"| plex={plex_title!r} omdb={omdb_title!r}"
        )
        return {
            "plex_guid": plex_guid,
            "library": library,
            "context_lang": ctx_lang,
            "plex_original_title": plex_original_title,
            "plex_imdb_id": plex_imdb_id,
            "omdb_imdb_id": omdb_imdb_id_obj,
            "imdb_rating": imdb_rating_obj,
            "imdb_votes": imdb_votes_obj,
            "plex_title": plex_title,
            "plex_year": plex_year_obj,  # valor original (humano)
            "omdb_title": omdb_title_obj,
            "omdb_year": omdb_year_obj,
            "action": "Skip (IMDb mismatch)",
            "suggestions_json": json.dumps(
                {
                    "skip_reason": "imdb_mismatch",
                    "plex_imdb_id": plex_imdb_id,
                    "omdb_imdb_id": omdb_imdb_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }

    n_plex_year = _normalize_year(plex_year_obj)
    n_omdb_year = _normalize_year(omdb_year_obj)

    title_diff = bool(plex_title and omdb_title and _titles_differ(plex_title, omdb_title))
    year_diff = bool(n_plex_year is not None and n_omdb_year is not None and n_plex_year != n_omdb_year)

    if not title_diff and not year_diff:
        return None

    # ✅ Si Plex ya tiene originalTitle y coincide con OMDb Title, no sugerimos título.
    if title_diff and plex_original_title and omdb_title:
        if _norm_lookup(plex_original_title) == _norm_lookup(omdb_title):
            logger.debug_ctx(
                "METADATA",
                "Skip title suggestion: plex_original_title matches OMDb Title | "
                f"{library} / original={plex_original_title!r} omdb={omdb_title!r}",
            )
            title_diff = False

    suggestions: dict[str, object] = {}

    if title_diff and omdb_title:
        blocked = should_skip_new_title_suggestion(
            context_lang=ctx_lang,
            current_title=plex_title,
            omdb_title=omdb_title,
        )

        # Fallback “loose” (títulos EN sin function-words).
        if (not blocked) and (ctx_lang in ("es", "it", "fr")) and _is_ascii_like_title(omdb_title):
            blocked = True

        if blocked:
            logger.debug_ctx(
                "METADATA",
                "Blocked new_title (localized library) -> emitting alt_title | "
                f"ctx_lang={ctx_lang} | {library} / plex={plex_title!r} omdb={omdb_title!r}",
            )
            suggestions["alt_title"] = omdb_title
            suggestions["alt_title_lang"] = "en"
            title_diff = False

    if title_diff and omdb_title:
        suggestions["new_title"] = omdb_title
    if year_diff and n_omdb_year is not None:
        suggestions["new_year"] = n_omdb_year

    if not suggestions:
        return None

    if "new_title" in suggestions and "new_year" in suggestions:
        action = "Fix title & year"
    elif "new_title" in suggestions:
        action = "Fix title"
    elif "new_year" in suggestions and ("alt_title" in suggestions):
        action = "Fix year (alt title info)"
    elif "alt_title" in suggestions:
        action = "Alt title info"
    else:
        action = "Fix year"

    row: dict[str, object] = {
        "plex_guid": plex_guid,
        "library": library,
        "context_lang": ctx_lang,
        "plex_original_title": plex_original_title,
        "plex_imdb_id": plex_imdb_id,
        "omdb_imdb_id": omdb_imdb_id_obj,
        "imdb_rating": imdb_rating_obj,
        "imdb_votes": imdb_votes_obj,
        "plex_title": plex_title,
        "plex_year": plex_year_obj,  # valor original (humano)
        "omdb_title": omdb_title_obj,
        "omdb_year": omdb_year_obj,
        "action": action,  # ✅ coherente con reporting.py
        "suggestions_json": json.dumps(suggestions, ensure_ascii=False, separators=(",", ":")),
    }

    logger.debug_ctx(
        "METADATA",
        f"Generated metadata suggestion | {library} / {plex_title!r} | suggestions={suggestions}",
    )
    return row
