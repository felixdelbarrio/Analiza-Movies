from __future__ import annotations

"""
Core genérico de análisis para una película, independiente del origen
(Plex, DLNA, fichero local, etc.).

Este módulo recibe un MovieInput (modelo de entrada unificado),
obtiene datos de OMDb mediante una función inyectada y delega la
decisión final a la lógica bayesiana de `scoring.py` y a los
umbrales configurados en `config.py`, a través de `decide_action`.

Además, utiliza `decision_logic.detect_misidentified` para producir
la pista `misidentified_hint` cuando hay sospechas de identificación
incorrecta.
"""

from collections.abc import Callable, Mapping
from typing import TypedDict

from backend.movie_input import MovieInput
from backend.decision_logic import detect_misidentified
from backend.omdb_client import extract_ratings_from_omdb
from backend.scoring import decide_action


class AnalysisRow(TypedDict, total=False):
    """
    Contrato de salida mínimo del core genérico.

    Esta fila es luego enriquecida por capas superiores (por ejemplo,
    el analizador específico de Plex) antes de volcarse a CSV.
    """

    source: str
    library: str
    title: str
    year: int | None

    imdb_rating: float | None
    rt_score: int | None
    imdb_votes: int | None
    plex_rating: float | None

    decision: str
    reason: str
    misidentified_hint: str

    file: str
    file_size_bytes: int | None

    imdb_id_hint: str


FetchOmdbCallable = Callable[[str, int | None], Mapping[str, object]]


def analyze_input_movie(
    movie: MovieInput,
    fetch_omdb: FetchOmdbCallable,
) -> AnalysisRow:
    """
    Analiza una película genérica (`MovieInput`) usando OMDb.

    Pasos:
      1. Llama a `fetch_omdb(title, year)` para obtener un dict tipo OMDb.
      2. Usa `extract_ratings_from_omdb` para sacar imdb_rating, imdb_votes, rt_score.
      3. Llama a `scoring.decide_action` (Bayes + thresholds del .env vía config.py).
      4. Usa `decision_logic.detect_misidentified` para construir `misidentified_hint`.
      5. Devuelve una fila `AnalysisRow` mínima, lista para ser enriquecida
         por capas superiores (Plex, DLNA concreto, etc.).

    No realiza I/O de ficheros ni logging directamente.
    """
    # ------------------------------------------------------------------
    # 1) Consultar OMDb mediante la función inyectada
    # ------------------------------------------------------------------
    omdb_data: dict[str, object] = {}
    try:
        raw = fetch_omdb(movie.title, movie.year)
        omdb_data = dict(raw) if isinstance(raw, Mapping) else {}
    except Exception:
        # Defensivo: si falla la llamada, trabajamos sin datos OMDb
        omdb_data = {}

    # ------------------------------------------------------------------
    # 2) Extraer ratings desde OMDb
    # ------------------------------------------------------------------
    imdb_rating, imdb_votes, rt_score = extract_ratings_from_omdb(omdb_data)

    # ------------------------------------------------------------------
    # 3) Decisión KEEP / MAYBE / DELETE / UNKNOWN vía scoring.decide_action
    # ------------------------------------------------------------------
    decision, reason = decide_action(
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        year=movie.year,
        metacritic_score=None,
    )

    # ------------------------------------------------------------------
    # 4) Detección de posibles películas mal identificadas
    # ------------------------------------------------------------------
    misidentified_hint = detect_misidentified(
        plex_title=movie.title,
        plex_year=movie.year,
        omdb_data=omdb_data,
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
    )

    # ------------------------------------------------------------------
    # 5) Construir fila base
    # ------------------------------------------------------------------
    row: AnalysisRow = {
        "source": movie.source,
        "library": movie.library,
        "title": movie.title,
        "year": movie.year,
        "imdb_rating": imdb_rating,
        "rt_score": rt_score,
        "imdb_votes": imdb_votes,
        "plex_rating": None,  # dlna/local no tienen rating Plex aquí
        "decision": decision,
        "reason": reason,
        "misidentified_hint": misidentified_hint,
        "file": movie.file_path,
        "file_size_bytes": movie.file_size_bytes,
    }

    if movie.imdb_id_hint:
        row["imdb_id_hint"] = movie.imdb_id_hint

    return row