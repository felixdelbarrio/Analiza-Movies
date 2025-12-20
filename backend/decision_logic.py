"""
Lógica de heurística para detectar posibles películas mal identificadas.

Funciones públicas:
- detect_misidentified(...): devuelve una cadena con pistas ('' si no hay).
- sort_filtered_rows(rows): ordena filas según reglas de prioridad.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from typing import Final

from backend import logger as _logger
from backend.config import (
    IMDB_MIN_VOTES_FOR_KNOWN,
    IMDB_RATING_LOW_THRESHOLD,
    RT_RATING_LOW_THRESHOLD,
)

TITLE_SIMILARITY_THRESHOLD: Final[float] = 0.60


def _normalize_title(s: str | None) -> str:
    """Normaliza un título para comparación: minúsculas, sin puntuación, espacios colapsados."""
    if not s:
        return ""
    s2 = s.lower()
    s2 = re.sub(r"[^a-z0-9\s]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _safe_imdb_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def detect_misidentified(
    plex_title: str | None,
    plex_year: int | None,
    plex_imdb_id: str | None,
    omdb_data: Mapping[str, object] | None,
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
) -> str:
    """
    Devuelve un texto con pistas de posible identificación errónea,
    o cadena vacía si no hay sospechas.

    REGLA DE ORO:
      - Si Plex IMDb ID coincide con OMDb imdbID -> NO es misidentified (return "").

    Heurísticas (solo si no aplica la regla de oro):
      - IMDb ID mismatch (si ambos existen y difieren).
      - Título Plex vs Título OMDb muy distintos.
      - Año Plex vs Año OMDb separados > 1 año.
      - IMDb muy baja con bastantes votos (posible "otra" peli).
      - Rotten Tomatoes muy bajo con bastantes votos.
    """
    if not omdb_data:
        return ""

    hints: list[str] = []

    plex_imdb = _safe_imdb_id(plex_imdb_id)

    omdb_imdb = _safe_imdb_id(omdb_data.get("imdbID"))
    if omdb_data.get("Response") != "True":
        # Si OMDb no trae ficha válida, no forzamos misidentified aquí.
        return ""

    # -----------------------------
    # 0) REGLA DE ORO: si coincide imdb -> NO misidentified
    # -----------------------------
    if plex_imdb and omdb_imdb and plex_imdb == omdb_imdb:
        return ""

    # -----------------------------
    # 1) IMDb ID mismatch (señal fuerte)
    # -----------------------------
    if plex_imdb and omdb_imdb and plex_imdb != omdb_imdb:
        hints.append(f"IMDb mismatch: Plex={plex_imdb} vs OMDb={omdb_imdb}")

    # -----------------------------
    # 2) Datos básicos de OMDb
    # -----------------------------
    omdb_title_raw = omdb_data.get("Title")
    omdb_title = omdb_title_raw if isinstance(omdb_title_raw, str) else ""
    omdb_year_raw = omdb_data.get("Year")

    pt = _normalize_title(plex_title)
    ot = _normalize_title(omdb_title)

    # -----------------------------
    # 3) Títulos claramente distintos
    # -----------------------------
    if pt and ot:
        if pt != ot and pt not in ot and ot not in pt:
            sim = difflib.SequenceMatcher(a=pt, b=ot).ratio()
            _logger.debug(f"Title similarity for '{plex_title}' vs '{omdb_title}': {sim:.2f}")
            if sim < TITLE_SIMILARITY_THRESHOLD:
                hints.append(
                    f"Title mismatch: Plex='{plex_title}' vs OMDb='{omdb_title}' (sim={sim:.2f})"
                )

    # -----------------------------
    # 4) Años muy diferentes (> 1)
    # -----------------------------
    try:
        if plex_year is not None and omdb_year_raw is not None:
            plex_year_int = int(plex_year)
            omdb_year_int = int(str(omdb_year_raw)[:4])  # "1994–1998"
            if abs(plex_year_int - omdb_year_int) > 1:
                hints.append(f"Year mismatch: Plex={plex_year_int}, OMDb={omdb_year_int}")
    except Exception:
        _logger.debug(
            f"Could not compare years: plex_year={plex_year!r}, omdb_year={omdb_year_raw!r}"
        )

    # -----------------------------
    # 5) IMDb muy baja con suficientes votos
    # -----------------------------
    votes: int = imdb_votes if isinstance(imdb_votes, int) else 0
    if (
        imdb_rating is not None
        and imdb_rating <= IMDB_RATING_LOW_THRESHOLD
        and votes >= IMDB_MIN_VOTES_FOR_KNOWN
    ):
        hints.append(
            (
                f"IMDb muy baja ({imdb_rating:.1f} ≤ {IMDB_RATING_LOW_THRESHOLD}) "
                f"con bastantes votos ({votes} ≥ {IMDB_MIN_VOTES_FOR_KNOWN}). "
                "Revisar identificación."
            )
        )

    # -----------------------------
    # 6) RT muy bajo con suficientes votos
    # -----------------------------
    if (
        rt_score is not None
        and rt_score <= RT_RATING_LOW_THRESHOLD
        and votes >= IMDB_MIN_VOTES_FOR_KNOWN
    ):
        hints.append(
            (
                f"RT muy bajo ({rt_score}% ≤ {RT_RATING_LOW_THRESHOLD}%) "
                f"para una peli aparentemente conocida ({votes} votos IMDb)."
            )
        )

    return " | ".join(hints)


def sort_filtered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Ordena las filas filtradas para el CSV final, priorizando:
      1) DELETE primero, luego MAYBE, luego KEEP, luego UNKNOWN.
      2) Más votos IMDb (las más relevantes/seguras antes).
      3) Mayor rating IMDb.
      4) Mayor tamaño de fichero (más espacio a liberar primero).
    """

    def key_func(r: dict[str, object]) -> tuple[int, int, float, int]:
        decision_raw = r.get("decision")
        decision = decision_raw if isinstance(decision_raw, str) else "UNKNOWN"

        imdb_votes_raw = r.get("imdb_votes")
        imdb_votes = imdb_votes_raw if isinstance(imdb_votes_raw, int) else 0

        imdb_rating_raw = r.get("imdb_rating")
        imdb_rating = float(imdb_rating_raw) if isinstance(imdb_rating_raw, (int, float)) else 0.0

        file_size_raw = r.get("file_size")
        file_size = file_size_raw if isinstance(file_size_raw, int) else 0

        decision_rank = {"DELETE": 0, "MAYBE": 1, "KEEP": 2, "UNKNOWN": 3}.get(decision, 3)
        return decision_rank, -imdb_votes, -imdb_rating, -file_size

    return sorted(rows, key=key_func)