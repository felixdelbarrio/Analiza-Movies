from __future__ import annotations

"""
backend/analyze_input_core.py

Core genérico de análisis para una película (MovieInput), independiente del origen
(Plex, DLNA, fichero local, etc.).

Principios de diseño
--------------------
Este módulo está diseñado para ser:
- Puro y testeable:
    * No hace IO de disco.
    * No hace red directamente (inyecta fetch_omdb).
    * No escribe caches.
    * No hace logging por defecto (solo trazas si se inyecta callback).
- Defensivo:
    * Nunca debe lanzar por un problema de instrumentación, parseo o scoring.
    * Normaliza decisiones y reason.
- Performante:
    * “Lazy OMDb” en 2 fases para evitar llamadas innecesarias.

Lazy OMDb (2 fases)
-------------------
Fase A (barata):
- Calcula scoring SIN ratings (IMDB/RT/votes None).
- Si la decisión es fuerte (KEEP/DELETE) se evita OMDb.

Fase B (cara):
- Solo si la decisión es MAYBE/UNKNOWN:
    * llama a fetch_omdb(title, year)
    * extrae ratings
    * recalcula scoring con señales reales
    * si hay OMDb, ejecuta detect_misidentified()

Instrumentación / trazas
------------------------
- El core no conoce DEBUG_MODE/SILENT_MODE.
- Si se pasa analysis_trace(line), emitimos trazas cortas (truncadas).
- Las trazas nunca rompen el análisis (try/except).

Salida
------
Devuelve un AnalysisRow (dict minimalista). Capas superiores enriquecen:
- poster_url, trailer_url, omdb_json, wiki ids, etc.
"""

from collections.abc import Callable, Mapping
from typing import Final, TypedDict

from backend.decision_logic import detect_misidentified
from backend.movie_input import MovieInput
from backend.omdb_client import extract_ratings_from_omdb
from backend.scoring import compute_scoring

# ============================================================================
# Tipos públicos
# ============================================================================


class AnalysisRow(TypedDict, total=False):
    """
    Contrato de salida mínimo del core.

    NOTA:
    - Es “base row”. Capas superiores pueden sobrescribir title/year/file
      para representar mejor el “display” (p.ej. título real de Plex).
    """

    # Identidad básica
    source: str
    library: str
    title: str
    year: int | None

    # Señales (ratings)
    imdb_rating: float | None
    imdb_bayes: float | None  # puntuación bayesiana final (exportable)
    rt_score: int | None
    imdb_votes: int | None
    plex_rating: float | None

    # Resultado
    decision: str
    reason: str
    misidentified_hint: str

    # Archivo y tamaño
    file: str
    file_size_bytes: int | None

    # Pista de imdb_id (si el origen lo trae)
    imdb_id_hint: str

    # Meta opcional (útil para debugging aguas arriba; no rompe compatibilidad)
    used_omdb: bool
    omdb_keys_count: int


# Callable inyectada para consultar OMDb (normalmente cacheada + throttling).
FetchOmdbCallable = Callable[[str, int | None], Mapping[str, object]]

# Callback opcional de trazas (core -> orquestador -> logger.debug_ctx/progress).
TraceCallable = Callable[[str], None]

# ============================================================================
# Constantes internas (truncado y normalización)
# ============================================================================

_VALID_DECISIONS: Final[set[str]] = {"KEEP", "MAYBE", "DELETE", "UNKNOWN"}

_TRACE_LINE_MAX_CHARS: Final[int] = 220
_TRACE_REASON_MAX_CHARS: Final[int] = 140

# Para no “ensuciar” reason con strings enormes o payloads inesperados.
_REASON_FALLBACK: Final[str] = "scoring did not provide a usable reason"


# ============================================================================
# Helpers defensivos
# ============================================================================

def _clip(text: str, *, max_len: int) -> str:
    """Trunca un string a un tamaño máximo (con sentinel)."""
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 12)] + " …(truncated)"


def _safe_str(value: object, *, max_len: int) -> str:
    """Convierte a string de forma defensiva y truncada."""
    try:
        return _clip(str(value), max_len=max_len)
    except Exception:
        return "<unprintable>"


def _normalize_decision(decision_raw: object) -> str:
    """Normaliza la decisión a {KEEP,MAYBE,DELETE,UNKNOWN}."""
    if decision_raw is None:
        return "UNKNOWN"
    cand = str(decision_raw).strip().upper()
    return cand if cand in _VALID_DECISIONS else "UNKNOWN"


def _extract_bayes_from_scoring(scoring: Mapping[str, object]) -> float | None:
    """
    Extrae el score bayesiano desde el dict de scoring.

    Convención actual:
        scoring["inputs"]["score_bayes"]
    """
    inputs = scoring.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    sb = inputs.get("score_bayes")
    if isinstance(sb, (int, float)):
        return float(sb)
    return None


def _compute_scoring_safe(
    *,
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
    year: int | None,
    metacritic_score: int | None,
    trace: Callable[[str], None],
) -> tuple[str, str, float | None]:
    """
    Ejecuta compute_scoring de forma defensiva.

    Returns:
        (decision, reason, imdb_bayes)
    """
    try:
        scoring = compute_scoring(
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rt_score=rt_score,
            year=year,
            metacritic_score=metacritic_score,
        )

        if not isinstance(scoring, Mapping):
            trace("scoring fail | compute_scoring returned non-mapping -> UNKNOWN")
            return "UNKNOWN", "compute_scoring returned non-mapping", None

        decision = _normalize_decision(scoring.get("decision"))

        reason_raw = scoring.get("reason")
        if isinstance(reason_raw, str):
            reason = reason_raw
        elif reason_raw is None:
            reason = _REASON_FALLBACK
        else:
            # Evitamos estructuras raras: lo convertimos a str defensivo.
            reason = str(reason_raw)

        imdb_bayes = _extract_bayes_from_scoring(scoring)

        trace(
            "scoring ok | "
            f"decision={decision} "
            f"bayes={_safe_str(imdb_bayes, max_len=32)} "
            f"reason={_safe_str(reason, max_len=_TRACE_REASON_MAX_CHARS)}"
        )
        return decision, reason, imdb_bayes

    except Exception as exc:
        trace(f"scoring fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")
        return "UNKNOWN", "compute_scoring failed", None


# ============================================================================
# API principal
# ============================================================================

def analyze_input_movie(
    movie: MovieInput,
    fetch_omdb: FetchOmdbCallable,
    *,
    plex_title: str | None = None,
    plex_year: int | None = None,
    plex_rating: float | None = None,
    metacritic_score: int | None = None,
    analysis_trace: TraceCallable | None = None,
) -> AnalysisRow:
    """
    Analiza una película genérica (MovieInput) usando señales externas.

    Lazy OMDb:
    - Fase A: scoring sin OMDb.
    - Fase B: solo si decisión no concluyente (MAYBE/UNKNOWN).

    Args:
        movie:
            Entrada unificada.
        fetch_omdb:
            Callable inyectada para resolver OMDb (idealmente cacheada y throttled).
            Firma: fetch_omdb(title, year) -> Mapping[str, object]
        plex_title / plex_year:
            Display title/año preferibles para detect_misidentified (si aplica).
        plex_rating:
            Rating del origen Plex (si aplica). No se usa para scoring aquí,
            pero se exporta a la fila para reporting/diagnóstico.
        metacritic_score:
            Señal opcional ya computada externamente (si existe).
        analysis_trace:
            Callback opcional de trazas (corto, truncado, nunca rompe).

    Returns:
        AnalysisRow: fila base y robusta para reporting.
    """

    def _trace(msg: str) -> None:
        """Trazas seguras y truncadas."""
        if analysis_trace is None:
            return
        try:
            analysis_trace(_clip(msg, max_len=_TRACE_LINE_MAX_CHARS))
        except Exception:
            # La instrumentación no debe romper el pipeline.
            return

    # Normalización de campos “mínimos” para trazabilidad
    lib = movie.library or ""
    title = movie.title or ""
    year = movie.year

    _trace(
        "start | "
        f"src={_safe_str(movie.source, max_len=32)} "
        f"lib={_safe_str(lib, max_len=80)} "
        f"title={_safe_str(title, max_len=120)} "
        f"year={_safe_str(year, max_len=16)}"
    )

    # ------------------------------------------------------------------
    # Fase A: scoring sin OMDb (sin red, sin cache lookup)
    # ------------------------------------------------------------------
    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None
    imdb_bayes: float | None = None

    _trace("phaseA | scoring without omdb (ratings=None)")
    decision, reason, imdb_bayes = _compute_scoring_safe(
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
        year=year,
        metacritic_score=metacritic_score,
        trace=_trace,
    )

    # Decisión de “ir a OMDb”:
    # - KEEP/DELETE suelen ser suficientemente fuertes con heurística local.
    # - MAYBE/UNKNOWN requieren señales externas para desempatar.
    should_fetch_omdb = decision in {"MAYBE", "UNKNOWN"}

    # ------------------------------------------------------------------
    # Fase B: OMDb solo si hace falta
    # ------------------------------------------------------------------
    used_omdb = False
    omdb_data: dict[str, object] = {}

    if should_fetch_omdb:
        _trace("phaseB | fetching omdb (needed)")
        try:
            raw = fetch_omdb(title, year)
            omdb_data = dict(raw) if isinstance(raw, Mapping) else {}
            used_omdb = bool(omdb_data)
            _trace(f"omdb ok | keys={len(omdb_data)}")
        except Exception as exc:
            omdb_data = {}
            used_omdb = False
            _trace(f"omdb fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

        # Ratings desde OMDb (defensivo)
        if omdb_data:
            try:
                imdb_rating, imdb_votes, rt_score = extract_ratings_from_omdb(omdb_data)
                _trace(
                    "ratings ok | "
                    f"imdb_rating={_safe_str(imdb_rating, max_len=32)} "
                    f"votes={_safe_str(imdb_votes, max_len=32)} "
                    f"rt={_safe_str(rt_score, max_len=32)}"
                )
            except Exception as exc:
                imdb_rating, imdb_votes, rt_score = None, None, None
                _trace(f"ratings fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

        # Re-score con señales reales (aunque ratings sigan None, la lógica es consistente)
        _trace("phaseB | scoring with omdb-derived signals")
        decision, reason, imdb_bayes = _compute_scoring_safe(
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rt_score=rt_score,
            year=year,
            metacritic_score=metacritic_score,
            trace=_trace,
        )
    else:
        _trace("phaseB | skipped omdb (decision strong)")

    # ------------------------------------------------------------------
    # Misidentified: solo si OMDb devolvió algo usable
    # ------------------------------------------------------------------
    misidentified_hint = ""

    if omdb_data:
        detect_title = plex_title if isinstance(plex_title, str) and plex_title.strip() else title
        detect_year = plex_year if isinstance(plex_year, int) else year

        _trace("misidentified | running")
        try:
            misidentified_hint = detect_misidentified(
                plex_title=detect_title,
                plex_year=detect_year,
                plex_imdb_id=movie.imdb_id_hint,
                omdb_data=omdb_data,
                imdb_rating=imdb_rating,
                imdb_votes=imdb_votes,
                rt_score=rt_score,
            )
        except Exception as exc:
            misidentified_hint = ""
            _trace(f"misidentified fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

        if not isinstance(misidentified_hint, str):
            misidentified_hint = str(misidentified_hint) if misidentified_hint is not None else ""

        if misidentified_hint.strip():
            _trace("misidentified | YES")
        else:
            _trace("misidentified | no")
    else:
        _trace("misidentified | skipped (no omdb data)")

    # ------------------------------------------------------------------
    # Construcción de fila base
    # ------------------------------------------------------------------
    row: AnalysisRow = {
        "source": movie.source,
        "library": movie.library,
        "title": movie.title,
        "year": year,
        "imdb_rating": imdb_rating,
        "imdb_bayes": imdb_bayes,
        "rt_score": rt_score,
        "imdb_votes": imdb_votes,
        "plex_rating": plex_rating,
        "decision": _normalize_decision(decision),
        "reason": reason if isinstance(reason, str) and reason.strip() else _REASON_FALLBACK,
        "misidentified_hint": misidentified_hint.strip(),
        "file": movie.file_path,
        "file_size_bytes": movie.file_size_bytes,
        "used_omdb": used_omdb,
        "omdb_keys_count": len(omdb_data),
    }

    if isinstance(movie.imdb_id_hint, str) and movie.imdb_id_hint.strip():
        row["imdb_id_hint"] = movie.imdb_id_hint.strip()

    _trace("done")
    return row