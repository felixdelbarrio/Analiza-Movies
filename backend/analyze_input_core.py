from __future__ import annotations

"""
Core genérico de análisis para una película, independiente del origen
(Plex, DLNA, fichero local, etc).

Este módulo recibe un MovieInput (modelo de entrada unificado),
obtiene datos de OMDb mediante una función inyectada y delega la
decisión final a la lógica bayesiana de `scoring.py` (compute_scoring).

Además, utiliza `decision_logic.detect_misidentified` para producir
la pista `misidentified_hint` cuando hay sospechas de identificación
incorrecta.
"""

from collections.abc import Callable, Mapping
from typing import TypedDict

from backend.decision_logic import detect_misidentified
from backend.movie_input import MovieInput
from backend.omdb_client import extract_ratings_from_omdb
from backend.scoring import compute_scoring


class AnalysisRow(TypedDict, total=False):
    """Contrato de salida mínimo del core genérico.

    Esta fila es luego enriquecida por capas superiores (por ejemplo,
    el pipeline de colección) antes de volcarse a CSV.
    """

    source: str
    library: str
    title: str
    year: int | None

    imdb_rating: float | None
    imdb_bayes: float | None  # ✅ NUEVO: puntuación bayesiana final exportable
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
    *,
    plex_title: str | None = None,
    plex_year: int | None = None,
    plex_rating: float | None = None,
    metacritic_score: int | None = None,
) -> AnalysisRow:
    """Analiza una película genérica (`MovieInput`) usando OMDb.

    Pasos:
      1. Llama a `fetch_omdb(title, year)` para obtener un dict tipo OMDb.
      2. Usa `extract_ratings_from_omdb` para sacar imdb_rating, imdb_votes,
         rt_score.
      3. Llama a `scoring.compute_scoring` (Bayes + thresholds del .env vía
         config.py) para producir decision/reason y el score bayesiano.
      4. Usa `decision_logic.detect_misidentified` para construir
         `misidentified_hint`.
      5. Devuelve una fila `AnalysisRow` mínima, lista para ser enriquecida por
         capas superiores (Plex, DLNA concreto, etc).

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
    # 3) Decisión KEEP / MAYBE / DELETE / UNKNOWN vía scoring.compute_scoring
    #    + capturamos el score bayesiano en imdb_bayes
    # ------------------------------------------------------------------
    decision: str = "UNKNOWN"
    reason: str = ""
    imdb_bayes: float | None = None

    try:
        scoring = compute_scoring(
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rt_score=rt_score,
            year=movie.year,
            metacritic_score=metacritic_score,
        )

        d = scoring.get("decision")
        r = scoring.get("reason")
        decision = str(d) if d is not None else "UNKNOWN"
        reason = str(r) if r is not None else ""

        inputs = scoring.get("inputs")
        if isinstance(inputs, Mapping):
            sb = inputs.get("score_bayes")
            if isinstance(sb, (int, float)):
                imdb_bayes = float(sb)

    except Exception:
        # Defensivo: si algo raro pasa, no rompemos el análisis
        decision = "UNKNOWN"
        reason = "compute_scoring failed"
        imdb_bayes = None

    # ------------------------------------------------------------------
    # 4) Detección de posibles películas mal identificadas
    # ------------------------------------------------------------------
    detect_title = plex_title if plex_title is not None else movie.title
    detect_year = plex_year if plex_year is not None else movie.year

    # IMPORTANTE: plex_imdb_id sale del MovieInput unificado (hint)
    misidentified_hint = detect_misidentified(
        plex_title=detect_title,
        plex_year=detect_year,
        plex_imdb_id=movie.imdb_id_hint,
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
        "imdb_bayes": imdb_bayes,  # ✅ NUEVO
        "rt_score": rt_score,
        "imdb_votes": imdb_votes,
        "plex_rating": plex_rating,
        "decision": decision,
        "reason": reason,
        "misidentified_hint": misidentified_hint,
        "file": movie.file_path,
        "file_size_bytes": movie.file_size_bytes,
    }

    if movie.imdb_id_hint:
        row["imdb_id_hint"] = movie.imdb_id_hint

    return row