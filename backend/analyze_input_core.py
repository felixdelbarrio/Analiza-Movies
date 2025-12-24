from __future__ import annotations

"""
backend/analyze_input_core.py

Core genérico de análisis para una película (MovieInput), independiente del origen
(Plex, DLNA, fichero local, etc.).

🎯 Principios de diseño (se mantienen)
-------------------------------------
- Puro y testeable:
    * No hace IO de disco.
    * No hace red directamente (inyecta fetch_omdb).
    * No escribe caches.
    * No hace logging por defecto (solo trazas si se inyecta callback).
- Defensivo:
    * Nunca debe lanzar por instrumentación, parseo o scoring.
    * Normaliza decisión y reason.
- Performante:
    * Lazy OMDb en 2 fases para evitar llamadas innecesarias.

✅ Knobs centralizables (config.py)
----------------------------------
- COLLECTION_TRACE_LINE_MAX_CHARS:
    * Longitud máxima de una línea de traza (core -> orquestador).
- ANALYZE_TRACE_REASON_MAX_CHARS:
    * Truncado adicional para el "reason" dentro de mensajes de traza
      (para evitar contaminar trazas con textos enormes).
- ANALYZE_CORE_METRICS_ENABLED:
    * Feature flag para desactivar métricas del core (run_metrics) sin tocar código.

Importante:
- Para no romper compatibilidad: si config.py no tiene alguna variable,
  usamos defaults razonables.

📈 Métricas (run_metrics.py) - best-effort
------------------------------------------
Este módulo intenta emitir contadores y timings si:
1) run_metrics está importable, y
2) ANALYZE_CORE_METRICS_ENABLED=True

Nunca rompe el análisis si run_metrics no existe o su API difiere.

Lazy OMDb (2 fases)
-------------------
Fase A (barata):
- Scoring SIN ratings (IMDB/RT/votes None).
- Si la decisión es fuerte (KEEP/DELETE) se evita OMDb.

Fase B (cara):
- Solo si decisión es MAYBE/UNKNOWN:
    * llama a fetch_omdb(title, year)
    * extrae ratings
    * recalcula scoring con señales reales
    * si hay OMDb, ejecuta detect_misidentified()

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

# -----------------------------------------------------------------------------
# Config (best-effort): centralizamos knobs si existen
# -----------------------------------------------------------------------------
try:
    from backend import config as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None  # type: ignore

# -----------------------------------------------------------------------------
# run_metrics (best-effort / no-op si no existe)
# -----------------------------------------------------------------------------
try:
    import backend.run_metrics as _rm  # type: ignore
except Exception:  # pragma: no cover
    _rm = None  # type: ignore


# =============================================================================
# Helpers de configuración (NO lanzan)
# =============================================================================

def _cfg_get_int(name: str, default: int) -> int:
    """
    Lee un int desde config si existe; fallback seguro.

    Nota:
    - No loguea (config ya se encarga de warnings).
    - Nunca lanza.
    """
    if _cfg is None:
        return int(default)
    try:
        return int(getattr(_cfg, name, default))
    except Exception:
        return int(default)


def _cfg_get_bool(name: str, default: bool) -> bool:
    """
    Lee un bool desde config si existe; fallback seguro.

    Nota:
    - No loguea.
    - Nunca lanza.
    """
    if _cfg is None:
        return bool(default)
    try:
        return bool(getattr(_cfg, name, default))
    except Exception:
        return bool(default)


# =============================================================================
# Knobs del core (desde config.py)
# =============================================================================

# Feature flag global del módulo: métricas del core ON/OFF.
_METRICS_ENABLED: Final[bool] = _cfg_get_bool("ANALYZE_CORE_METRICS_ENABLED", True)

# Truncados (para trazas) – no afecta la lógica de scoring, solo verbosidad.
_TRACE_LINE_MAX_CHARS: Final[int] = _cfg_get_int("COLLECTION_TRACE_LINE_MAX_CHARS", 220)
_TRACE_REASON_MAX_CHARS: Final[int] = _cfg_get_int("ANALYZE_TRACE_REASON_MAX_CHARS", 140)

# Normalización del "reason" (fallback si scoring no aporta un reason usable).
_REASON_FALLBACK: Final[str] = "scoring did not provide a usable reason"

# Set de decisiones válidas para normalizar.
_VALID_DECISIONS: Final[set[str]] = {"KEEP", "MAYBE", "DELETE", "UNKNOWN"}


# =============================================================================
# Métricas (best-effort / nunca rompen)
# =============================================================================

def _rm_inc(name: str, value: int = 1) -> None:
    """
    Incremento best-effort. Nunca rompe el análisis.

    Compatibilidad:
    - Soporta APIs típicas: inc(name, value=...) o counter_inc(name, value=...)
    """
    if not _METRICS_ENABLED or _rm is None:
        return
    try:
        fn = getattr(_rm, "inc", None) or getattr(_rm, "counter_inc", None)
        if callable(fn):
            fn(name, value=value)  # type: ignore[misc]
    except Exception:
        return


def _rm_observe_seconds(name: str, seconds: float) -> None:
    """
    Timing best-effort. Nunca rompe el análisis.

    Compatibilidad:
    - observe_seconds(name, seconds=...) o timing(name, seconds=...)
    """
    if not _METRICS_ENABLED or _rm is None:
        return
    try:
        fn = getattr(_rm, "observe_seconds", None) or getattr(_rm, "timing", None)
        if callable(fn):
            fn(name, seconds=seconds)  # type: ignore[misc]
    except Exception:
        return


class _RM_Timer:
    """
    Context manager de timing best-effort.

    Importante:
    - No importa time a nivel de módulo para minimizar carga.
    - Nunca lanza; el análisis no debe depender de métricas.
    """
    def __init__(self, name: str) -> None:
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_RM_Timer":
        from time import monotonic
        self._t0 = monotonic()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        from time import monotonic
        _rm_observe_seconds(self._name, monotonic() - self._t0)


# =============================================================================
# Tipos públicos
# =============================================================================

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


# =============================================================================
# Helpers defensivos (strings / normalización)
# =============================================================================

def _clip(text: str, *, max_len: int) -> str:
    """
    Trunca un string a un tamaño máximo (con sentinel).

    Nota: mantiene el final "...(truncated)" para que sea evidente en debug.
    """
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 12)] + " …(truncated)"


def _safe_str(value: object, *, max_len: int) -> str:
    """
    Convierte a string de forma defensiva y truncada.

    - Nunca lanza.
    - Útil para trazas y para no arrastrar payloads enormes.
    """
    try:
        return _clip(str(value), max_len=max_len)
    except Exception:
        return "<unprintable>"


def _normalize_decision(decision_raw: object) -> str:
    """
    Normaliza la decisión a {KEEP,MAYBE,DELETE,UNKNOWN}.

    Importante:
    - Si llega algo inesperado: UNKNOWN.
    """
    if decision_raw is None:
        return "UNKNOWN"
    cand = str(decision_raw).strip().upper()
    return cand if cand in _VALID_DECISIONS else "UNKNOWN"


def _normalize_reason(reason_raw: object) -> str:
    """
    Normaliza reason a string “usable”.

    Política:
    - str no vacío -> se usa tal cual
    - None -> fallback
    - otro tipo -> str(...)
    """
    if isinstance(reason_raw, str):
        r = reason_raw.strip()
        return r if r else _REASON_FALLBACK
    if reason_raw is None:
        return _REASON_FALLBACK
    try:
        r = str(reason_raw).strip()
        return r if r else _REASON_FALLBACK
    except Exception:
        return _REASON_FALLBACK


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


# =============================================================================
# Scoring defensivo (wrapper)
# =============================================================================

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
        (decision_normalizada, reason_normalizado, imdb_bayes_opt)

    Garantías:
    - Nunca lanza.
    - Siempre devuelve una decisión válida.
    - El reason siempre es string no vacío (fallback si hace falta).
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
            _rm_inc("analyze_core.scoring.fail", 1)
            trace("scoring fail | compute_scoring returned non-mapping -> UNKNOWN")
            return "UNKNOWN", "compute_scoring returned non-mapping", None

        decision = _normalize_decision(scoring.get("decision"))
        reason = _normalize_reason(scoring.get("reason"))
        imdb_bayes = _extract_bayes_from_scoring(scoring)

        _rm_inc("analyze_core.scoring.ok", 1)
        trace(
            "scoring ok | "
            f"decision={decision} "
            f"bayes={_safe_str(imdb_bayes, max_len=32)} "
            f"reason={_safe_str(reason, max_len=_TRACE_REASON_MAX_CHARS)}"
        )
        return decision, reason, imdb_bayes

    except Exception as exc:
        _rm_inc("analyze_core.scoring.fail", 1)
        trace(f"scoring fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")
        return "UNKNOWN", "compute_scoring failed", None


# =============================================================================
# API principal
# =============================================================================

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
            NO es logging; el orquestador decide cómo imprimirlo.

    Returns:
        AnalysisRow: fila base y robusta para reporting.
    """
    _rm_inc("analyze_core.calls", 1)
    with _RM_Timer("analyze_core.seconds"):

        def _trace(msg: str) -> None:
            """
            Trazas seguras y truncadas.

            Importante:
            - No hace logging directamente.
            - Nunca lanza.
            - Trunca a COLLECTION_TRACE_LINE_MAX_CHARS (config).
            """
            if analysis_trace is None:
                return
            try:
                analysis_trace(_clip(msg, max_len=_TRACE_LINE_MAX_CHARS))
            except Exception:
                # La instrumentación no debe romper el pipeline.
                return

        # Normalización mínima para evitar None/objetos raros en trazas
        lib = movie.library or ""
        title = movie.title or ""
        year = movie.year if isinstance(movie.year, int) else movie.year  # mantiene compatibilidad

        _trace(
            "start | "
            f"src={_safe_str(movie.source, max_len=32)} "
            f"lib={_safe_str(lib, max_len=80)} "
            f"title={_safe_str(title, max_len=120)} "
            f"year={_safe_str(year, max_len=16)}"
        )

        # ------------------------------------------------------------------
        # Fase A: scoring sin OMDb
        # ------------------------------------------------------------------
        _rm_inc("analyze_core.phaseA.calls", 1)

        imdb_rating: float | None = None
        imdb_votes: int | None = None
        rt_score: int | None = None
        imdb_bayes: float | None = None

        _trace("phaseA | scoring without omdb (ratings=None)")
        decision, reason, imdb_bayes = _compute_scoring_safe(
            imdb_rating=None,
            imdb_votes=None,
            rt_score=None,
            year=movie.year,
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
            _rm_inc("analyze_core.phaseB.fetch_attempted", 1)
            _trace("phaseB | fetching omdb (needed)")
            try:
                raw = fetch_omdb(title, movie.year)
                omdb_data = dict(raw) if isinstance(raw, Mapping) else {}
                used_omdb = bool(omdb_data)

                _rm_inc("analyze_core.phaseB.fetch_ok", 1 if used_omdb else 0)
                _rm_inc("analyze_core.phaseB.fetch_empty", 1 if not used_omdb else 0)
                _trace(f"omdb ok | keys={len(omdb_data)}")
            except Exception as exc:
                omdb_data = {}
                used_omdb = False
                _rm_inc("analyze_core.phaseB.fetch_fail", 1)
                _trace(f"omdb fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

            # Ratings desde OMDb (defensivo)
            if omdb_data:
                try:
                    imdb_rating, imdb_votes, rt_score = extract_ratings_from_omdb(omdb_data)
                    _rm_inc("analyze_core.ratings.extract_ok", 1)
                    _trace(
                        "ratings ok | "
                        f"imdb_rating={_safe_str(imdb_rating, max_len=32)} "
                        f"votes={_safe_str(imdb_votes, max_len=32)} "
                        f"rt={_safe_str(rt_score, max_len=32)}"
                    )
                except Exception as exc:
                    imdb_rating, imdb_votes, rt_score = None, None, None
                    _rm_inc("analyze_core.ratings.extract_fail", 1)
                    _trace(f"ratings fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

            # Re-score con señales reales
            _trace("phaseB | scoring with omdb-derived signals")
            decision, reason, imdb_bayes = _compute_scoring_safe(
                imdb_rating=imdb_rating,
                imdb_votes=imdb_votes,
                rt_score=rt_score,
                year=movie.year,
                metacritic_score=metacritic_score,
                trace=_trace,
            )
        else:
            _trace("phaseB | skipped omdb (decision strong)")

        _rm_inc("analyze_core.used_omdb.true", 1 if used_omdb else 0)
        _rm_inc("analyze_core.used_omdb.false", 1 if not used_omdb else 0)

        # ------------------------------------------------------------------
        # Misidentified: solo si OMDb devolvió algo usable
        # ------------------------------------------------------------------
        misidentified_hint = ""

        if omdb_data:
            _rm_inc("analyze_core.misidentified.ran", 1)

            detect_title = plex_title if isinstance(plex_title, str) and plex_title.strip() else title
            detect_year = plex_year if isinstance(plex_year, int) else movie.year

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
                _rm_inc("analyze_core.misidentified.fail", 1)
                _trace(f"misidentified fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

            if not isinstance(misidentified_hint, str):
                try:
                    misidentified_hint = str(misidentified_hint) if misidentified_hint is not None else ""
                except Exception:
                    misidentified_hint = ""

            if misidentified_hint.strip():
                _rm_inc("analyze_core.misidentified.yes", 1)
                _trace("misidentified | YES")
            else:
                _rm_inc("analyze_core.misidentified.no", 1)
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
            "year": movie.year,
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

        # Exportamos imdb_id_hint solo si es usable (mantiene compatibilidad)
        if isinstance(movie.imdb_id_hint, str) and movie.imdb_id_hint.strip():
            row["imdb_id_hint"] = movie.imdb_id_hint.strip()

        _trace("done")
        return row