from __future__ import annotations

"""
backend/analyze_input_core.py

Core genérico de análisis para una película, independiente del origen
(Plex, DLNA, fichero local, etc.).

Qué hace (flow):
1) Obtiene datos OMDb mediante una función inyectada (`fetch_omdb`).
2) Extrae señales normalizadas (IMDb rating/votes, RottenTomatoes score) desde el
   payload OMDb con `extract_ratings_from_omdb`.
3) Calcula la decisión KEEP/MAYBE/DELETE/UNKNOWN mediante `scoring.compute_scoring`
   y captura el score bayesiano (`imdb_bayes`) si está disponible.
4) Ejecuta `decision_logic.detect_misidentified` para producir un
   `misidentified_hint` cuando hay sospechas de identificación incorrecta.

Principios de diseño:
- Pureza / testabilidad:
  - No hace I/O de ficheros.
  - No hace llamadas de red directamente.
  - No escribe en caches ni en disco.
  - No hace logging “por defecto”.
- Robustez:
  - Cualquier fallo en OMDb / ratings / scoring / misidentified se degrada a un
    resultado razonable (UNKNOWN) y el core NO rompe el pipeline.
  - La instrumentación nunca puede romper el análisis.
- Instrumentación opcional:
  - Permite inyectar `analysis_trace` (callback) para que capas superiores registren
    trazas en modo DEBUG, sin acoplar este core a backend.logger.

Filosofía de logs (alineada con lo trabajado):
- El core no conoce SILENT_MODE/DEBUG_MODE: solo emite trazas si le inyectan
  un callback.
- Las trazas se mantienen pequeñas:
  - truncadas
  - sin dumps de JSON
  - sin “reason” completo si es largo

Este módulo devuelve un `AnalysisRow` (TypedDict) con campos mínimos.
Capas superiores (p.ej. collection_analysis.py) enriquecen esa fila con:
- file_url, poster_url, trailer_url, omdb_json, wikidata_id, etc.
"""

from collections.abc import Callable, Mapping
from typing import TypedDict

from backend.decision_logic import detect_misidentified
from backend.movie_input import MovieInput
from backend.omdb_client import extract_ratings_from_omdb
from backend.scoring import compute_scoring


class AnalysisRow(TypedDict, total=False):
    """
    Contrato de salida mínimo del core genérico.

    NOTA:
    - Este dict se considera “base row”.
    - Los orquestadores (Plex/DLNA/colección) pueden sobrescribir campos como
      title/year/file si quieren mostrar el “display_title” del origen, etc.
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
    plex_rating: float | None  # puede venir inyectado desde capas Plex

    # Resultado
    decision: str
    reason: str
    misidentified_hint: str

    # Archivo (friendly path) y tamaño (si se conoce)
    file: str
    file_size_bytes: int | None

    # Pista de imdb_id (si el origen lo trae)
    imdb_id_hint: str


# Función inyectada para consultar OMDb (normalmente cacheada en omdb_client)
FetchOmdbCallable = Callable[[str, int | None], Mapping[str, object]]

# Callback opcional de trazas (instrumentación).
# Diseñado para que collection_analysis lo conecte a su _append_log/_log_debug con caps.
TraceCallable = Callable[[str], None]

# Decisiones válidas esperadas por el resto del pipeline
_VALID_DECISIONS: set[str] = {"KEEP", "MAYBE", "DELETE", "UNKNOWN"}


# ---------------------------------------------------------------------------
# Helpers internos (sin dependencias externas, defensivos)
# ---------------------------------------------------------------------------

_TRACE_LINE_MAX_CHARS: int = 220
_TRACE_REASON_MAX_CHARS: int = 120


def _clip(s: str, max_len: int) -> str:
    """Trunca strings para evitar trazas enormes (títulos raros, errores gigantes, etc.)."""
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 12)] + " …(truncated)"


def _safe_short(value: object, *, max_len: int = _TRACE_LINE_MAX_CHARS) -> str:
    """Convierte a string corto y seguro para trazas (sin JSONs grandes)."""
    try:
        return _clip(str(value), max_len=max_len)
    except Exception:
        return "<unprintable>"


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
    Analiza una película genérica (`MovieInput`) usando OMDb.

    Args:
        movie:
            Entrada unificada (source/library/title/year/file_path, etc.).
        fetch_omdb:
            Callable inyectada que devuelve un Mapping tipo OMDb.
            Se asume que puede usar cache interna y throttling global.
        plex_title / plex_year:
            Título/año “display” del origen Plex, si se quiere priorizar para
            detectar misidentifications (Plex a veces muestra título distinto al buscado).
        plex_rating:
            Rating del usuario o rating del item en Plex (si aplica).
        metacritic_score:
            Score opcional (puede venir de OMDb o Wiki minimal) para enriquecer scoring.
        analysis_trace:
            Callback opcional para trazas debug. Si se pasa, este core emitirá mensajes
            cortos (sin payloads grandes). Las capas superiores deben decidir si
            imprimirlas, guardarlas o caparlas.

    Returns:
        AnalysisRow: fila mínima, lista para enriquecer y exportar.
    """

    def _trace(msg: str) -> None:
        """
        Emite trazas cortas y seguras si hay callback.
        Nunca debe romper el análisis.
        """
        if analysis_trace is None:
            return
        try:
            analysis_trace(_clip(msg, max_len=_TRACE_LINE_MAX_CHARS))
        except Exception:
            return

    # Identificadores compactos para traza (evitar ruido)
    lib = movie.library or ""
    title = movie.title or ""
    year = movie.year

    _trace(
        "start | "
        f"lib={_safe_short(lib)} title={_safe_short(title)} year={_safe_short(year)} "
        f"src={_safe_short(movie.source)}"
    )

    # ------------------------------------------------------------------
    # 1) Consultar OMDb mediante la función inyectada (defensivo)
    # ------------------------------------------------------------------
    omdb_data: dict[str, object] = {}
    try:
        raw = fetch_omdb(movie.title, movie.year)
        omdb_data = dict(raw) if isinstance(raw, Mapping) else {}
        _trace(f"omdb ok | keys={len(omdb_data)}")
    except Exception as exc:
        omdb_data = {}
        _trace(f"omdb fail | err={_safe_short(exc)}")

    # ------------------------------------------------------------------
    # 2) Extraer ratings desde OMDb
    # ------------------------------------------------------------------
    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None

    try:
        imdb_rating, imdb_votes, rt_score = extract_ratings_from_omdb(omdb_data)
        _trace(
            "ratings ok | "
            f"imdb_rating={_safe_short(imdb_rating)} votes={_safe_short(imdb_votes)} rt={_safe_short(rt_score)}"
        )
    except Exception as exc:
        imdb_rating, imdb_votes, rt_score = None, None, None
        _trace(f"ratings fail | err={_safe_short(exc)}")

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

        decision_candidate = str(d) if d is not None else "UNKNOWN"
        if decision_candidate not in _VALID_DECISIONS:
            _trace(f"scoring unexpected decision={_safe_short(decision_candidate)} -> UNKNOWN")
            decision_candidate = "UNKNOWN"

        decision = decision_candidate
        reason = str(r) if r is not None else ""

        inputs = scoring.get("inputs")
        if isinstance(inputs, Mapping):
            sb = inputs.get("score_bayes")
            if isinstance(sb, (int, float)):
                imdb_bayes = float(sb)

        # “reason” puede ser largo: solo mostramos un clip para depurar.
        reason_clip = _safe_short(reason, max_len=_TRACE_REASON_MAX_CHARS)
        _trace(f"scoring ok | decision={decision} bayes={_safe_short(imdb_bayes)} reason={reason_clip}")

    except Exception as exc:
        decision = "UNKNOWN"
        reason = "compute_scoring failed"
        imdb_bayes = None
        _trace(f"scoring fail | err={_safe_short(exc)}")

    # ------------------------------------------------------------------
    # 4) Detección de posibles películas mal identificadas
    # ------------------------------------------------------------------
    detect_title = plex_title if plex_title is not None else movie.title
    detect_year = plex_year if plex_year is not None else movie.year

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
        _trace(f"misidentified fail | err={_safe_short(exc)}")

    if not isinstance(misidentified_hint, str):
        misidentified_hint = str(misidentified_hint) if misidentified_hint is not None else ""

    if misidentified_hint.strip():
        # Señal binaria para correlación (texto completo ya viaja en row).
        _trace("misidentified yes")

    # ------------------------------------------------------------------
    # 5) Construir fila base
    # ------------------------------------------------------------------
    row: AnalysisRow = {
        "source": movie.source,
        "library": movie.library,
        "title": movie.title,
        "year": movie.year,
        "imdb_rating": imdb_rating,
        "imdb_bayes": imdb_bayes,
        "rt_score": rt_score,
        "imdb_votes": imdb_votes,
        "plex_rating": plex_rating,
        "decision": decision,
        "reason": reason,
        "misidentified_hint": misidentified_hint,
        "file": movie.file_path,
        "file_size_bytes": movie.file_size_bytes,
    }

    # Solo añadimos imdb_id_hint si existe (evita strings/columnas inútiles)
    if isinstance(movie.imdb_id_hint, str) and movie.imdb_id_hint.strip():
        row["imdb_id_hint"] = movie.imdb_id_hint.strip()

    _trace("done")

    return row