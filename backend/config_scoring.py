from __future__ import annotations

from backend.config_base import (
    _cap_float_min,
    _cap_int,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    _get_env_str,
)

# ============================================================
# Scoring bayesiano / heurística
# ============================================================

BAYES_GLOBAL_MEAN_DEFAULT: float = _get_env_float("BAYES_GLOBAL_MEAN_DEFAULT", 6.5)
BAYES_DELETE_MAX_SCORE: float = _get_env_float("BAYES_DELETE_MAX_SCORE", 5.6)

BAYES_MIN_TITLES_FOR_GLOBAL_MEAN: int = _cap_int(
    "BAYES_MIN_TITLES_FOR_GLOBAL_MEAN",
    _get_env_int("BAYES_MIN_TITLES_FOR_GLOBAL_MEAN", 200),
    min_v=0,
    max_v=1_000_000,
)

RATING_MIN_TITLES_FOR_AUTO: int = _cap_int(
    "RATING_MIN_TITLES_FOR_AUTO",
    _get_env_int("RATING_MIN_TITLES_FOR_AUTO", 300),
    min_v=0,
    max_v=1_000_000,
)


def _parse_votes_by_year(raw: str) -> list[tuple[int, int]]:
    if not raw:
        return []
    cleaned = raw.strip().strip('"').strip("'")
    table: list[tuple[int, int]] = []
    for part in cleaned.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        try:
            year_limit_str, votes_min_str = chunk.split(":")
            year_limit = int(year_limit_str.strip())
            votes_min = int(votes_min_str.strip())
            table.append((year_limit, votes_min))
        except Exception:
            continue
    return sorted(table, key=lambda x: x[0])


_IMDB_VOTES_BY_YEAR_RAW: str = (
    _get_env_str("IMDB_VOTES_BY_YEAR", "1980:500,2000:2000,2010:5000,9999:10000")
    or "1980:500,2000:2000,2010:5000,9999:10000"
)
IMDB_VOTES_BY_YEAR: list[tuple[int, int]] = _parse_votes_by_year(_IMDB_VOTES_BY_YEAR_RAW)

IMDB_KEEP_MIN_RATING: float = _get_env_float("IMDB_KEEP_MIN_RATING", 5.7)
IMDB_DELETE_MAX_RATING: float = _get_env_float("IMDB_DELETE_MAX_RATING", 5.5)

IMDB_KEEP_MIN_VOTES: int = _cap_int(
    "IMDB_KEEP_MIN_VOTES",
    _get_env_int("IMDB_KEEP_MIN_VOTES", 30000),
    min_v=0,
    max_v=2_000_000_000,
)


def get_votes_threshold_for_year(year: int | None) -> int:
    if not IMDB_VOTES_BY_YEAR:
        return IMDB_KEEP_MIN_VOTES
    try:
        y = int(year) if year is not None else None
    except (TypeError, ValueError):
        y = None
    if y is None:
        return IMDB_VOTES_BY_YEAR[-1][1]
    for year_limit, votes_min in IMDB_VOTES_BY_YEAR:
        if y <= year_limit:
            return votes_min
    return IMDB_VOTES_BY_YEAR[-1][1]


# ============================================================
# Misidentificación / títulos sospechosos (decision_logic.py)
# ============================================================

# Umbral de similitud (difflib ratio) por debajo del cual consideramos mismatch.
DECISION_TITLE_SIMILARITY_THRESHOLD: float = _cap_float_min(
    "DECISION_TITLE_SIMILARITY_THRESHOLD",
    _get_env_float("DECISION_TITLE_SIMILARITY_THRESHOLD", 0.60),
    min_v=0.0,
)

# Límite razonable para comparar títulos
# (protege contra metadata corrupta o títulos patológicos).
DECISION_MAX_TITLE_LEN_FOR_COMPARE: int = _cap_int(
    "DECISION_MAX_TITLE_LEN_FOR_COMPARE",
    _get_env_int("DECISION_MAX_TITLE_LEN_FOR_COMPARE", 180),
    min_v=32,
    max_v=2000,
)

# Diferencia de años tolerable antes de considerar mismatch (Plex vs OMDb).
DECISION_YEAR_MISMATCH_MAX_DELTA: int = _cap_int(
    "DECISION_YEAR_MISMATCH_MAX_DELTA",
    _get_env_int("DECISION_YEAR_MISMATCH_MAX_DELTA", 1),
    min_v=0,
    max_v=20,
)

# Longitud mínima de título para ejecutar difflib.SequenceMatcher.
# Evita ruido y coste innecesario con títulos muy cortos ("Up", "It", etc.).
DECISION_MIN_TITLE_LEN_FOR_DIFFLIB: int = _cap_int(
    "DECISION_MIN_TITLE_LEN_FOR_DIFFLIB",
    _get_env_int("DECISION_MIN_TITLE_LEN_FOR_DIFFLIB", 4),
    min_v=1,
    max_v=50,
)

# Control de “usabilidad” de OMDb en decision_logic:
# - True  => solo se consideran reglas si Response == "True"
# - False => basta con que OMDb traiga imdbID o Title
#
# Default True = comportamiento conservador (menos falsos positivos).
DECISION_OMDB_REQUIRE_RESPONSE_TRUE: bool = _get_env_bool(
    "DECISION_OMDB_REQUIRE_RESPONSE_TRUE",
    True,
)

# Umbrales “peli conocida” + muy baja puntuación (heurística suave).
IMDB_RATING_LOW_THRESHOLD: float = _get_env_float("IMDB_RATING_LOW_THRESHOLD", 3.0)
RT_RATING_LOW_THRESHOLD: int = _cap_int(
    "RT_RATING_LOW_THRESHOLD",
    _get_env_int("RT_RATING_LOW_THRESHOLD", 20),
    min_v=0,
    max_v=100,
)
IMDB_MIN_VOTES_FOR_KNOWN: int = _cap_int(
    "IMDB_MIN_VOTES_FOR_KNOWN",
    _get_env_int("IMDB_MIN_VOTES_FOR_KNOWN", 100),
    min_v=0,
    max_v=2_000_000_000,
)

# ============================================================
# Rotten Tomatoes
# ============================================================

RT_KEEP_MIN_SCORE: int = _cap_int(
    "RT_KEEP_MIN_SCORE",
    _get_env_int("RT_KEEP_MIN_SCORE", 55),
    min_v=0,
    max_v=100,
)
IMDB_KEEP_MIN_RATING_WITH_RT: float = _get_env_float(
    "IMDB_KEEP_MIN_RATING_WITH_RT",
    6.0,
)
RT_DELETE_MAX_SCORE: int = _cap_int(
    "RT_DELETE_MAX_SCORE",
    _get_env_int("RT_DELETE_MAX_SCORE", 50),
    min_v=0,
    max_v=100,
)

# ============================================================
# Metacritic
# ============================================================

METACRITIC_KEEP_MIN_SCORE: int = _cap_int(
    "METACRITIC_KEEP_MIN_SCORE",
    _get_env_int("METACRITIC_KEEP_MIN_SCORE", 70),
    min_v=0,
    max_v=100,
)
METACRITIC_DELETE_MAX_SCORE: int = _cap_int(
    "METACRITIC_DELETE_MAX_SCORE",
    _get_env_int("METACRITIC_DELETE_MAX_SCORE", 40),
    min_v=0,
    max_v=100,
)

# ============================================================
# Percentiles automáticos
# ============================================================

AUTO_KEEP_RATING_PERCENTILE: float = _get_env_float("AUTO_KEEP_RATING_PERCENTILE", 0.60)
AUTO_DELETE_RATING_PERCENTILE: float = _get_env_float("AUTO_DELETE_RATING_PERCENTILE", 0.40)

# ============================================================
# Metadata fix
# ============================================================

METADATA_DRY_RUN: bool = _get_env_bool("METADATA_DRY_RUN", True)
METADATA_APPLY_CHANGES: bool = _get_env_bool("METADATA_APPLY_CHANGES", False)

