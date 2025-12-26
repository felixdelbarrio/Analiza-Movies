from __future__ import annotations

"""
backend/analyze_input_core.py

Core genérico de análisis para una película (MovieInput), independiente del origen
(Plex, DLNA, fichero local, etc.).

Principios de diseño
--------------------
- Puro y testeable:
    * No hace IO de disco.
    * No hace red directamente (inyecta fetch_omdb).
    * No escribe caches.
    * No hace logging imperativo:
        - Solo emite trazas si se inyecta callback (analysis_trace)
        - O, en modo debug, usa logger central debug_ctx (si existe)
- Defensivo:
    * Nunca debe lanzar por instrumentación, parseo o scoring.
    * Un fallo en OMDb / parseo de ratings NO debe cambiar una decisión fuerte.
    * Normaliza decision/reason y añade reason_code estable.
- Performante:
    * Lazy OMDb en 2 fases: evita llamadas externas cuando no aportan valor.
    * Minimiza branching (fase B extraída a helper).
    * Optimiza integración con run_metrics (lazy bind opcional).

Normalización
-------------
- decision: {KEEP, DELETE, MAYBE, UNKNOWN}
- reason: texto legible para UI/logs humanos
- reason_code: código estable para métricas/analytics (evita texto libre)

Lazy OMDb (2 fases)
-------------------
Fase A (barata):
- scoring sin ratings (IMDB/RT/votes None)
- si decisión es fuerte (KEEP/DELETE) => NO se llama a OMDb

Fase B (cara):
- solo si decisión es MAYBE/UNKNOWN:
    * fetch_omdb(title, year)
    * valida payload “usable”
    * extrae ratings
    * recalcula scoring con señales reales
    * si OMDb usable, ejecuta detect_misidentified()

Guardrails
----------
- Si OMDb falla / no usable / ratings no parsean:
    * No altera decisiones fuertes (no aplicable porque fuertes no llaman OMDb)
    * Para MAYBE/UNKNOWN: la decisión final queda como la de fase A
      (o equivalente si compute_scoring es estable a señales None).

Observabilidad: micro-métricas (best-effort)
--------------------------------------------
- analyze_core.decision_strong_without_omdb
- analyze_core.decision_changed_after_omdb
- analyze_core.decision_changed_after_omdb.to_keep / to_delete / to_maybe / to_unknown
- analyze_core.inconsistency.delete_with_high_imdb
- analyze_core.inconsistency.keep_with_low_imdb
- analyze_core.delete_but_misidentified
- analyze_core.strong_potential_contradiction_without_omdb (heurística; no hace red)

Knobs centralizables (backend/config o backend/config_core)
----------------------------------------------------------
Este módulo hace best-effort: si una variable no existe, usa defaults razonables.

- COLLECTION_TRACE_LINE_MAX_CHARS
- ANALYZE_TRACE_REASON_MAX_CHARS
- ANALYZE_CORE_METRICS_ENABLED
- ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_RATING
- ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_VOTES
- ANALYZE_INCONSISTENCY_KEEP_IMDB_MAX_RATING
- ANALYZE_INCONSISTENCY_KEEP_IMDB_MIN_VOTES
- ANALYZE_LOOKUP_TITLE_FALLBACK_ENABLED
- ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS
- ANALYZE_METRICS_STRONG_POTENTIAL_CONTRADICTION_ENABLED
- ANALYZE_METRICS_LAZY_BIND_ENABLED

Salida
------
Devuelve un AnalysisRow (fila base). Capas superiores pueden enriquecer con:
- poster_url, trailer_url, omdb_json, wiki ids, etc.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypedDict, cast

from backend.decision_logic import detect_misidentified
from backend.movie_input import MovieInput
from backend.omdb_client import extract_ratings_from_omdb
from backend.scoring import compute_scoring

# -----------------------------------------------------------------------------
# Config (best-effort). Preferimos backend.config (agregador) y caemos a config_core.
# -----------------------------------------------------------------------------
try:
    from backend import config as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None

try:
    from backend import config_core as _cfg_core  # type: ignore
except Exception:  # pragma: no cover
    _cfg_core = None

# -----------------------------------------------------------------------------
# run_metrics (best-effort / no-op si no existe)
# -----------------------------------------------------------------------------
try:
    import backend.run_metrics as _rm  # type: ignore
except Exception:  # pragma: no cover
    _rm = None

# -----------------------------------------------------------------------------
# Logger central (best-effort)
# -----------------------------------------------------------------------------
try:
    from backend import logger as _log  # type: ignore
except Exception:  # pragma: no cover
    _log = None


# =============================================================================
# Helpers de configuración (NO lanzan)
# =============================================================================

def _cfg_get_attr(name: str) -> object:
    """
    Busca un atributo en:
      1) backend.config (si existe)
      2) backend.config_core (si existe)
    y devuelve None si no se encuentra.
    """
    if _cfg is not None and hasattr(_cfg, name):
        try:
            return getattr(_cfg, name)
        except Exception:
            return None
    if _cfg_core is not None and hasattr(_cfg_core, name):
        try:
            return getattr(_cfg_core, name)
        except Exception:
            return None
    return None


def _cfg_get_int(name: str, default: int) -> int:
    v = _cfg_get_attr(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _cfg_get_float(name: str, default: float) -> float:
    v = _cfg_get_attr(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _cfg_get_bool(name: str, default: bool) -> bool:
    v = _cfg_get_attr(name)
    if v is None:
        return bool(default)
    try:
        return bool(v)
    except Exception:
        return bool(default)


# =============================================================================
# Knobs del core (desde config/config_core)
# =============================================================================

_METRICS_ENABLED: Final[bool] = _cfg_get_bool("ANALYZE_CORE_METRICS_ENABLED", True)
_METRICS_LAZY_BIND_ENABLED: Final[bool] = _cfg_get_bool("ANALYZE_METRICS_LAZY_BIND_ENABLED", True)

_TRACE_LINE_MAX_CHARS: Final[int] = _cfg_get_int("COLLECTION_TRACE_LINE_MAX_CHARS", 220)
_TRACE_REASON_MAX_CHARS: Final[int] = _cfg_get_int("ANALYZE_TRACE_REASON_MAX_CHARS", 140)

# Inconsistencias (solo observación; NO cambia decisiones)
_INCONS_DELETE_MIN_RATING: Final[float] = _cfg_get_float("ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_RATING", 7.5)
_INCONS_DELETE_MIN_VOTES: Final[int] = _cfg_get_int("ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_VOTES", 10_000)

_INCONS_KEEP_MAX_RATING: Final[float] = _cfg_get_float("ANALYZE_INCONSISTENCY_KEEP_IMDB_MAX_RATING", 4.5)
_INCONS_KEEP_MIN_VOTES: Final[int] = _cfg_get_int("ANALYZE_INCONSISTENCY_KEEP_IMDB_MIN_VOTES", 25_000)

# Fallback de lookup_title (si normalized_title_for_lookup() devuelve vacío)
_LOOKUP_TITLE_FALLBACK_ENABLED: Final[bool] = _cfg_get_bool("ANALYZE_LOOKUP_TITLE_FALLBACK_ENABLED", True)
_LOOKUP_TITLE_FALLBACK_MAX_CHARS: Final[int] = _cfg_get_int("ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS", 180)

# Métrica heurística para “contradicción” sin OMDb (no hace red)
_STRONG_POTENTIAL_CONTRADICTION_ENABLED: Final[bool] = _cfg_get_bool(
    "ANALYZE_METRICS_STRONG_POTENTIAL_CONTRADICTION_ENABLED",
    True,
)

_REASON_FALLBACK: Final[str] = "scoring did not provide a usable reason"
_VALID_DECISIONS: Final[set[str]] = {"KEEP", "MAYBE", "DELETE", "UNKNOWN"}
_LOG_TAG: Final[str] = "analyze_core"

# Prefijo centralizado de métricas (evita typos)
_METRIC_PREFIX: Final[str] = "analyze_core."


# =============================================================================
# Métricas (best-effort / nunca rompen)
# =============================================================================

class _IncFn(Protocol):
    def __call__(self, name: str, *, value: int = 1) -> None: ...


class _ObserveSecondsFn(Protocol):
    def __call__(self, name: str, *, seconds: float) -> None: ...


# Cache de binding (opcional) para reducir getattr en hot path
_METRICS_BIND_LOCKED: bool = False
_METRICS_INC_FN: _IncFn | None = None
_METRICS_OBS_FN: _ObserveSecondsFn | None = None


def _metrics_bind_once() -> None:
    global _METRICS_BIND_LOCKED, _METRICS_INC_FN, _METRICS_OBS_FN
    if _METRICS_BIND_LOCKED:
        return
    _METRICS_BIND_LOCKED = True

    if _rm is None:
        _METRICS_INC_FN = None
        _METRICS_OBS_FN = None
        return

    try:
        inc_obj = getattr(_rm, "inc", None) or getattr(_rm, "counter_inc", None)
        obs_obj = getattr(_rm, "observe_seconds", None) or getattr(_rm, "timing", None)
        _METRICS_INC_FN = cast(_IncFn, inc_obj) if callable(inc_obj) else None
        _METRICS_OBS_FN = cast(_ObserveSecondsFn, obs_obj) if callable(obs_obj) else None
    except Exception:
        _METRICS_INC_FN = None
        _METRICS_OBS_FN = None


def _m(name: str, value: int = 1) -> None:
    """
    Incrementa métrica con prefijo unificado.
    Nunca rompe el análisis.
    """
    if not _METRICS_ENABLED or _rm is None:
        return
    full = f"{_METRIC_PREFIX}{name}"
    try:
        if _METRICS_LAZY_BIND_ENABLED:
            _metrics_bind_once()
            if _METRICS_INC_FN is not None:
                _METRICS_INC_FN(full, value=value)
                return
        # fallback sin binding
        fn_obj = getattr(_rm, "inc", None) or getattr(_rm, "counter_inc", None)
        if callable(fn_obj):
            cast(_IncFn, fn_obj)(full, value=value)
    except Exception:
        return


def _m_obs(name: str, seconds: float) -> None:
    """
    Observa timing con prefijo unificado.
    Nunca rompe el análisis.
    """
    if not _METRICS_ENABLED or _rm is None:
        return
    full = f"{_METRIC_PREFIX}{name}"
    try:
        if _METRICS_LAZY_BIND_ENABLED:
            _metrics_bind_once()
            if _METRICS_OBS_FN is not None:
                _METRICS_OBS_FN(full, seconds=seconds)
                return
        fn_obj = getattr(_rm, "observe_seconds", None) or getattr(_rm, "timing", None)
        if callable(fn_obj):
            cast(_ObserveSecondsFn, fn_obj)(full, seconds=seconds)
    except Exception:
        return


class _RM_Timer:
    def __init__(self, name: str) -> None:
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_RM_Timer":
        from time import monotonic
        self._t0 = monotonic()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        from time import monotonic
        _m_obs(self._name, monotonic() - self._t0)


# =============================================================================
# Tipos públicos
# =============================================================================

class AnalysisRow(TypedDict, total=False):
    """
    Contrato de salida mínimo del core.

    Compatibilidad:
    - Mantiene: decision, reason.
    - Añade: reason_code (estable) y metadatos (phaseA/phaseB).
    """

    source: str
    library: str
    title: str
    year: int | None

    imdb_rating: float | None
    imdb_bayes: float | None
    rt_score: int | None
    imdb_votes: int | None
    plex_rating: float | None

    decision: str
    reason: str
    reason_code: str
    misidentified_hint: str

    file: str
    file_size_bytes: int | None

    imdb_id_hint: str

    used_omdb: bool
    omdb_keys_count: int
    decision_phaseA: str
    decision_phaseB: str


FetchOmdbCallable = Callable[[str, int | None], Mapping[str, object]]
TraceCallable = Callable[[str], None]


# =============================================================================
# Helpers defensivos (strings / normalización)
# =============================================================================

def _clip(text: str, *, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 12)] + " …(truncated)"


def _safe_str(value: object, *, max_len: int) -> str:
    try:
        return _clip(str(value), max_len=max_len)
    except Exception:
        return "<unprintable>"


def _normalize_decision(decision_raw: object) -> str:
    if decision_raw is None:
        return "UNKNOWN"
    cand = str(decision_raw).strip().upper()
    return cand if cand in _VALID_DECISIONS else "UNKNOWN"


def _normalize_reason(reason_raw: object) -> str:
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
    inputs = scoring.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    sb = inputs.get("score_bayes")
    if isinstance(sb, (int, float)):
        return float(sb)
    return None


def _is_omdb_usable(omdb_data: Mapping[str, object]) -> bool:
    """
    Determina si el payload OMDb es usable.

    Evita falsos positivos típicos:
    - {"Response":"False", "Error":"Movie not found!"}
    """
    resp = omdb_data.get("Response")
    if isinstance(resp, str) and resp.strip().lower() == "true":
        return True

    imdb_id = omdb_data.get("imdbID")
    if isinstance(imdb_id, str) and imdb_id.strip():
        return True

    title = omdb_data.get("Title")
    if isinstance(title, str) and title.strip():
        return True

    return False


def _collapse_spaces(s: str) -> str:
    return " ".join(s.strip().split())


def _get_lookup_title(movie: MovieInput, *, trace: Callable[[str], None]) -> str:
    """
    Obtiene el título “lookup” para OMDb.

    Política:
    1) movie.normalized_title_for_lookup().strip()
    2) Si queda vacío y fallback habilitado:
        - usa movie.title colapsando espacios
        - recorta a ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS

    Nota:
    - Solo mejora cobertura; no cambia decisiones fuertes (porque fuertes no llaman OMDb).
    """
    try:
        primary = (movie.normalized_title_for_lookup() or "").strip()
    except Exception:
        primary = ""

    if primary:
        return primary

    if not _LOOKUP_TITLE_FALLBACK_ENABLED:
        return ""

    raw = movie.title or ""
    fb = _collapse_spaces(raw)
    if not fb:
        return ""

    if len(fb) > _LOOKUP_TITLE_FALLBACK_MAX_CHARS:
        fb = fb[:_LOOKUP_TITLE_FALLBACK_MAX_CHARS].rstrip()

    trace(f"lookup_title fallback | used movie.title (len={len(fb)})")
    return fb


# =============================================================================
# Logging/tracing centralizado (sin romper)
# =============================================================================

def _make_tracer(analysis_trace: TraceCallable | None) -> Callable[[str], None]:
    """
    Devuelve una función trace(msg) segura y truncada.

    Prioridad:
    - Si analysis_trace está inyectado: se usa ese callback
    - Si no: logger.debug_ctx(tag=analyze_core) si existe
    - Si no hay logger: no hace nada
    """
    def _trace(msg: str) -> None:
        clipped = _clip(msg, max_len=_TRACE_LINE_MAX_CHARS)

        if analysis_trace is not None:
            try:
                analysis_trace(clipped)
            except Exception:
                return
            return

        if _log is None:
            return
        try:
            if hasattr(_log, "debug_ctx"):
                _log.debug_ctx(_LOG_TAG, clipped)  # type: ignore[attr-defined]
        except Exception:
            return

    return _trace


# =============================================================================
# reason_code (estable)
# =============================================================================

def _derive_reason_code(
    *,
    scoring: Mapping[str, object] | None,
    decision: str,
    used_omdb: bool,
    omdb_usable: bool,
    has_signals: bool,
) -> str:
    """
    Construye un reason_code estable.

    Prioridad:
    1) scoring["reason_code"] si existe y es str no vacío
    2) Heurística local conservadora:
        - decision final
        - si se usó OMDb
        - si OMDb era usable
        - si existen señales reales

    Nota:
    - reason_code es para telemetría estable (no string-matching del reason humano).
    """
    if scoring is not None:
        rc = scoring.get("reason_code")
        if isinstance(rc, str) and rc.strip():
            return rc.strip()

    if decision in {"KEEP", "DELETE"} and not used_omdb:
        return "strong_without_external"
    if decision in {"MAYBE", "UNKNOWN"} and not used_omdb:
        return "insufficient_signals_no_external"
    if used_omdb and not omdb_usable:
        return "external_failed_or_unusable"
    if used_omdb and omdb_usable and not has_signals:
        return "external_unusable_no_signals"
    if used_omdb and omdb_usable and has_signals:
        return "external_signals_applied"
    return "unknown"


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
) -> tuple[str, str, float | None, Mapping[str, object] | None]:
    """
    Ejecuta compute_scoring de forma defensiva.

    Returns:
        (decision_norm, reason_norm, imdb_bayes_opt, scoring_or_none)

    Garantías:
    - No lanza.
    - decision siempre pertenece a _VALID_DECISIONS.
    - reason siempre es string no vacío.
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
            _m("scoring.fail", 1)
            trace("scoring fail | compute_scoring returned non-mapping -> UNKNOWN")
            return "UNKNOWN", "compute_scoring returned non-mapping", None, None

        decision = _normalize_decision(scoring.get("decision"))
        reason = _normalize_reason(scoring.get("reason"))
        imdb_bayes = _extract_bayes_from_scoring(scoring)

        _m("scoring.ok", 1)
        trace(
            "scoring ok | "
            f"decision={decision} "
            f"bayes={_safe_str(imdb_bayes, max_len=32)} "
            f"reason={_safe_str(reason, max_len=_TRACE_REASON_MAX_CHARS)}"
        )
        return decision, reason, imdb_bayes, cast(Mapping[str, object], scoring)

    except Exception as exc:
        _m("scoring.fail", 1)
        trace(f"scoring fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")
        return "UNKNOWN", "compute_scoring failed", None, None


# =============================================================================
# Phase B (OMDb) – extraído para reducir branching
# =============================================================================

@dataclass(frozen=True)
class _PhaseBResult:
    used_omdb: bool
    omdb_usable: bool
    omdb_data: dict[str, object]
    imdb_rating: float | None
    imdb_votes: int | None
    rt_score: int | None
    decision: str
    reason: str
    imdb_bayes: float | None
    scoring: Mapping[str, object] | None


def _run_phaseB_omdb(
    *,
    movie: MovieInput,
    lookup_title: str,
    fetch_omdb: FetchOmdbCallable,
    metacritic_score: int | None,
    trace: Callable[[str], None],
) -> _PhaseBResult:
    """
    Ejecuta:
      - fetch_omdb
      - validación de payload
      - extracción de ratings
      - re-scoring

    Garantías:
    - No lanza.
    - Si OMDb falla/no usable: ratings quedan None y la decisión saldrá estable.
    """
    used_omdb = False
    omdb_usable = False
    omdb_data: dict[str, object] = {}
    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None

    if not lookup_title:
        _m("phaseB.skipped_empty_title", 1)
        trace("phaseB | skipped omdb (empty lookup_title)")
        decisionB, reasonB, imdb_bayesB, scoringB = _compute_scoring_safe(
            imdb_rating=None,
            imdb_votes=None,
            rt_score=None,
            year=movie.year,
            metacritic_score=metacritic_score,
            trace=trace,
        )
        return _PhaseBResult(
            used_omdb=False,
            omdb_usable=False,
            omdb_data={},
            imdb_rating=None,
            imdb_votes=None,
            rt_score=None,
            decision=decisionB,
            reason=reasonB,
            imdb_bayes=imdb_bayesB,
            scoring=scoringB,
        )

    _m("phaseB.fetch_attempted", 1)
    trace("phaseB | fetching omdb (needed)")

    try:
        raw = fetch_omdb(lookup_title, movie.year)
        raw_map: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
        used_omdb = True

        if _is_omdb_usable(raw_map):
            omdb_data = dict(raw_map)
            omdb_usable = True
            _m("phaseB.fetch_ok", 1)
            trace(f"omdb ok | usable=yes keys={len(omdb_data)}")
        else:
            _m("phaseB.fetch_unusable", 1)
            trace(
                "omdb ok | usable=no "
                f"keys={len(raw_map)} "
                f"response={_safe_str(raw_map.get('Response'), max_len=32)}"
            )
    except Exception as exc:
        used_omdb = True
        omdb_usable = False
        omdb_data = {}
        _m("phaseB.fetch_fail", 1)
        trace(f"omdb fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

    if omdb_data:
        try:
            imdb_rating, imdb_votes, rt_score = extract_ratings_from_omdb(omdb_data)
            _m("ratings.extract_ok", 1)
            trace(
                "ratings ok | "
                f"imdb_rating={_safe_str(imdb_rating, max_len=32)} "
                f"votes={_safe_str(imdb_votes, max_len=32)} "
                f"rt={_safe_str(rt_score, max_len=32)}"
            )
        except Exception as exc:
            imdb_rating, imdb_votes, rt_score = None, None, None
            _m("ratings.extract_fail", 1)
            trace(f"ratings fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

    trace("phaseB | scoring with omdb-derived signals")
    decisionB, reasonB, imdb_bayesB, scoringB = _compute_scoring_safe(
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        year=movie.year,
        metacritic_score=metacritic_score,
        trace=trace,
    )

    return _PhaseBResult(
        used_omdb=used_omdb,
        omdb_usable=omdb_usable,
        omdb_data=omdb_data,
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        decision=decisionB,
        reason=reasonB,
        imdb_bayes=imdb_bayesB,
        scoring=scoringB,
    )


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
    Analiza una película genérica (MovieInput) usando señales externas (OMDb).

    - Fase A: scoring sin OMDb
    - Fase B: OMDb solo si fase A es MAYBE/UNKNOWN
    - Misidentified: solo si OMDb usable

    Devuelve una fila base robusta y consistente.
    """
    _m("calls", 1)

    with _RM_Timer("seconds"):
        trace = _make_tracer(analysis_trace)

        lib = movie.library or ""
        title_raw = movie.title or ""

        # lookup_title: normalizado + fallback defensivo (para no perder cobertura)
        lookup_title = _get_lookup_title(movie, trace=trace)

        trace(
            "start | "
            f"src={_safe_str(movie.source, max_len=32)} "
            f"lib={_safe_str(lib, max_len=80)} "
            f"title={_safe_str(title_raw, max_len=120)} "
            f"year={_safe_str(movie.year, max_len=16)} "
            f"lookup_title={_safe_str(lookup_title, max_len=120)}"
        )

        # ------------------------------------------------------------------
        # Fase A: scoring sin OMDb
        # ------------------------------------------------------------------
        _m("phaseA.calls", 1)
        decisionA, reasonA, imdb_bayesA, scoringA = _compute_scoring_safe(
            imdb_rating=None,
            imdb_votes=None,
            rt_score=None,
            year=movie.year,
            metacritic_score=metacritic_score,
            trace=trace,
        )

        decision_phaseA = decisionA
        strongA = decisionA in {"KEEP", "DELETE"}

        decision_final = decisionA
        reason_final = reasonA
        imdb_bayes = imdb_bayesA
        scoring_final: Mapping[str, object] | None = scoringA

        used_omdb = False
        omdb_usable = False
        omdb_data: dict[str, object] = {}
        imdb_rating: float | None = None
        imdb_votes: int | None = None
        rt_score: int | None = None

        if strongA:
            _m("decision_strong_without_omdb", 1)
            trace(f"phaseB | skipped omdb (decision strong: {decisionA})")

            # (2) Heurística: “strong” con posible contradicción sin OMDb (NO hace red)
            if _STRONG_POTENTIAL_CONTRADICTION_ENABLED:
                # Ejemplos conservadores:
                # - DELETE pero plex_rating alto (si está disponible)
                # - KEEP pero plex_rating extremadamente bajo
                try:
                    if decisionA == "DELETE" and isinstance(plex_rating, (int, float)) and float(plex_rating) >= 8.0:
                        _m("strong_potential_contradiction_without_omdb", 1)
                        trace(f"heuristic | strong contradiction (DELETE with high plex_rating={plex_rating})")
                    if decisionA == "KEEP" and isinstance(plex_rating, (int, float)) and float(plex_rating) <= 2.0:
                        _m("strong_potential_contradiction_without_omdb", 1)
                        trace(f"heuristic | strong contradiction (KEEP with low plex_rating={plex_rating})")
                except Exception:
                    pass
        else:
            _m("phaseB.fetch_needed", 1)
            phaseB = _run_phaseB_omdb(
                movie=movie,
                lookup_title=lookup_title,
                fetch_omdb=fetch_omdb,
                metacritic_score=metacritic_score,
                trace=trace,
            )

            used_omdb = phaseB.used_omdb
            omdb_usable = phaseB.omdb_usable
            omdb_data = phaseB.omdb_data
            imdb_rating = phaseB.imdb_rating
            imdb_votes = phaseB.imdb_votes
            rt_score = phaseB.rt_score

            decision_final = phaseB.decision
            reason_final = phaseB.reason
            imdb_bayes = phaseB.imdb_bayes
            scoring_final = phaseB.scoring

        decision_phaseB = decision_final

        # Micro-métrica: decisión cambió tras OMDb
        if used_omdb and decision_phaseA != decision_phaseB:
            _m("decision_changed_after_omdb", 1)
            _m(f"decision_changed_after_omdb.to_{decision_phaseB.lower()}", 1)
            trace(f"decision changed | {decision_phaseA} -> {decision_phaseB}")

        _m("used_omdb.true", 1 if used_omdb else 0)
        _m("used_omdb.false", 1 if not used_omdb else 0)

        # ------------------------------------------------------------------
        # Misidentified: solo si OMDb usable (no solo “used_omdb”)
        # ------------------------------------------------------------------
        misidentified_hint = ""
        if omdb_data:
            _m("misidentified.ran", 1)

            detect_title = plex_title if isinstance(plex_title, str) and plex_title.strip() else title_raw
            detect_year = plex_year if isinstance(plex_year, int) else movie.year

            trace("misidentified | running")
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
                _m("misidentified.fail", 1)
                trace(f"misidentified fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

            if not isinstance(misidentified_hint, str):
                try:
                    misidentified_hint = str(misidentified_hint) if misidentified_hint is not None else ""
                except Exception:
                    misidentified_hint = ""

            if misidentified_hint.strip():
                _m("misidentified.yes", 1)
                trace("misidentified | YES")
            else:
                _m("misidentified.no", 1)
                trace("misidentified | no")
        else:
            trace("misidentified | skipped (no usable omdb data)")

        # (4) Métrica: DELETE pero misidentified sugiere mismatch (solo observación)
        if decision_phaseB == "DELETE" and misidentified_hint.strip():
            _m("delete_but_misidentified", 1)
            trace("inconsistency | DELETE but misidentified_hint is present")

        # ------------------------------------------------------------------
        # Inconsistencias (observación; NO cambia decisión)
        # ------------------------------------------------------------------
        # DELETE con IMDb muy alto + votos altos
        if (
            decision_phaseB == "DELETE"
            and isinstance(imdb_rating, (int, float))
            and float(imdb_rating) >= float(_INCONS_DELETE_MIN_RATING)
            and isinstance(imdb_votes, int)
            and int(imdb_votes) >= int(_INCONS_DELETE_MIN_VOTES)
        ):
            _m("inconsistency.delete_with_high_imdb", 1)
            trace(
                "inconsistency | DELETE with high imdb "
                f"(rating={imdb_rating} votes={imdb_votes}) "
                f"min_rating={_INCONS_DELETE_MIN_RATING} min_votes={_INCONS_DELETE_MIN_VOTES}"
            )

        # KEEP con IMDb muy bajo + votos altos
        if (
            decision_phaseB == "KEEP"
            and isinstance(imdb_rating, (int, float))
            and float(imdb_rating) <= float(_INCONS_KEEP_MAX_RATING)
            and isinstance(imdb_votes, int)
            and int(imdb_votes) >= int(_INCONS_KEEP_MIN_VOTES)
        ):
            _m("inconsistency.keep_with_low_imdb", 1)
            trace(
                "inconsistency | KEEP with low imdb "
                f"(rating={imdb_rating} votes={imdb_votes}) "
                f"max_rating={_INCONS_KEEP_MAX_RATING} min_votes={_INCONS_KEEP_MIN_VOTES}"
            )

        # ------------------------------------------------------------------
        # reason_code estable (5)
        # ------------------------------------------------------------------
        has_signals = any(v is not None for v in (imdb_rating, imdb_votes, rt_score))
        reason_code = _derive_reason_code(
            scoring=scoring_final,
            decision=decision_phaseB,
            used_omdb=used_omdb,
            omdb_usable=omdb_usable,
            has_signals=has_signals,
        )

        # ------------------------------------------------------------------
        # Fila base
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
            "decision": _normalize_decision(decision_phaseB),
            "reason": reason_final if isinstance(reason_final, str) and reason_final.strip() else _REASON_FALLBACK,
            "reason_code": reason_code,
            "misidentified_hint": misidentified_hint.strip(),
            "file": movie.file_path,
            "file_size_bytes": movie.file_size_bytes,
            "used_omdb": used_omdb,
            "omdb_keys_count": len(omdb_data),
            "decision_phaseA": _normalize_decision(decision_phaseA),
            "decision_phaseB": _normalize_decision(decision_phaseB),
        }

        if isinstance(movie.imdb_id_hint, str) and movie.imdb_id_hint.strip():
            row["imdb_id_hint"] = movie.imdb_id_hint.strip()

        trace("done")
        return row