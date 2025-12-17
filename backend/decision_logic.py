"""Lógica de heurística para detectar posibles películas mal identificadas.

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


def detect_misidentified(
    plex_title: str | None,
    plex_year: int | None,
    omdb_data: Mapping[str, object] | None,
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
) -> str:
    """
    Devuelve un texto con pistas de posible identificación errónea,
    o cadena vacía si no hay sospechas.

    Criterios:
      - Título Plex vs Título OMDb muy distintos.
      - Año Plex vs Año OMDb separados > 1 año.
      - IMDb muy baja con bastantes votos (posible "otra" peli").
      - Rotten Tomatoes muy bajo con bastantes votos.
    """
    if not omdb_data:
        return ""

    hints: list[str] = []

    # -----------------------------
    # 0) Datos básicos de OMDb
    # -----------------------------
    omdb_title_raw = omdb_data.get("Title")
    omdb_title = omdb_title_raw if isinstance(omdb_title_raw, str) else ""

    # Usaremos el campo Year crudo para cumplir con tests que esperan
    # que los errores de parseo se capturen y logueen.
    omdb_year_raw = omdb_data.get("Year")

    pt = _normalize_title(plex_title)
    ot = _normalize_title(omdb_title)

    # -----------------------------
    # 1) Títulos claramente distintos
    # -----------------------------
    if pt and ot:
        # Si uno contiene al otro, probablemente están relacionados
        if pt != ot and pt not in ot and ot not in pt:
            sim = difflib.SequenceMatcher(a=pt, b=ot).ratio()
            if sim < TITLE_SIMILARITY_THRESHOLD:
                hints.append(
                    f"Title mismatch: Plex='{plex_title}' vs OMDb='{omdb_title}' "
                    f"(sim={sim:.2f})"
                )
                # Nuestro logger no soporta formato posicional, así que formateamos aquí
                _logger.debug(
                    f"Title similarity for '{plex_title}' vs '{omdb_title}': {sim:.2f}"
                )

    # -----------------------------
    # 2) Años muy diferentes (> 1)
    #    Importante: si el año de OMDb no es parseable, se captura y se loguea.
    # -----------------------------
    try:
        if plex_year is not None and omdb_year_raw is not None:
            plex_year_int = int(plex_year)
            # Tomamos los 4 primeros caracteres del Year de OMDb, como "1994–1998"
            omdb_year_int = int(str(omdb_year_raw)[:4])

            if abs(plex_year_int - omdb_year_int) > 1:
                hints.append(f"Year mismatch: Plex={plex_year_int}, OMDb={omdb_year_int}")
    except Exception:
        _logger.debug(
            f"Could not compare years: plex_year={plex_year!r}, omdb_year={omdb_year_raw!r}"
        )

    # -----------------------------
    # 3) IMDb muy baja con suficientes votos
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
    # 4) RT muy bajo con suficientes votos
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


def sort_filtered_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
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
        imdb_rating = (
            float(imdb_rating_raw)
            if isinstance(imdb_rating_raw, (int, float))
            else 0.0
        )

        file_size_raw = r.get("file_size")
        file_size = file_size_raw if isinstance(file_size_raw, int) else 0

        decision_rank = {"DELETE": 0, "MAYBE": 1, "KEEP": 2, "UNKNOWN": 3}.get(
            decision,
            3,
        )
        # Negativos para ordenar de mayor a menor en votes/rating/size
        return decision_rank, -imdb_votes, -imdb_rating, -file_size

    return sorted(rows, key=key_func)