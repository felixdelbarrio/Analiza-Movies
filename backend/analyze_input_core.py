"""
backend/analyze_input_core.py

Core genérico de análisis para una película (MovieInput), independiente del origen
(Plex, DLNA, fichero local, etc.).

✅ Soporta extract_ratings_from_omdb con tupla legacy (3) o nueva (4) incluyendo metacritic_score.
✅ metacritic_score cuenta como señal externa y se propaga a scoring + report.
✅ Best-effort con config, run_metrics y logger (no rompe si faltan).

REFactor (title identity / lookup):
- ✅ Usa coalesce_movie_identity (best-effort) para derivar lookup_title + lookup_year
  apoyándose en file_path + extra["source_url"] cuando existe.
- ✅ lookup_year SOLO se usa para fetch OMDb (mejor precisión). movie.year se conserva en el row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Final, Protocol, TypedDict, cast
import threading

from backend.decision_logic import detect_misidentified
from backend.movie_input import MovieInput, coalesce_movie_identity
from backend.omdb_client import extract_ratings_from_omdb
from backend.scoring import compute_scoring

# -----------------------------------------------------------------------------
# Config (best-effort). Preferimos backend.config (agregador) y caemos a config_core.
# -----------------------------------------------------------------------------
_cfg: ModuleType | None = None
try:
    from backend import config as _cfg_mod  # type: ignore

    _cfg = cast(ModuleType, _cfg_mod)
except Exception:  # pragma: no cover
    _cfg = None

_cfg_core: ModuleType | None = None
try:
    from backend import config_core as _cfg_core_mod  # type: ignore

    _cfg_core = cast(ModuleType, _cfg_core_mod)
except Exception:  # pragma: no cover
    _cfg_core = None

# -----------------------------------------------------------------------------
# run_metrics (best-effort / no-op si no existe)
# -----------------------------------------------------------------------------
_rm: ModuleType | None = None
try:
    import backend.run_metrics as _rm_mod  # type: ignore

    _rm = cast(ModuleType, _rm_mod)
except Exception:  # pragma: no cover
    _rm = None

# -----------------------------------------------------------------------------
# Logger central (best-effort)
# -----------------------------------------------------------------------------
_log: ModuleType | None = None
try:
    from backend import logger as _log_mod  # type: ignore

    _log = cast(ModuleType, _log_mod)
except Exception:  # pragma: no cover
    _log = None


# =============================================================================
# Helpers de configuración (NO lanzan)
# =============================================================================
def _safe_getattr_obj(mod: ModuleType, name: str) -> object | None:
    """getattr() tipado estable (corta Any a object)."""
    try:
        return cast(object, getattr(mod, name))
    except Exception:
        return None


def _cfg_get_attr(name: str) -> object | None:
    """Busca un atributo en config/config_core y devuelve None si no se encuentra."""
    if _cfg is not None:
        try:
            if hasattr(_cfg, name):
                return _safe_getattr_obj(_cfg, name)
        except Exception:
            pass

    if _cfg_core is not None:
        try:
            if hasattr(_cfg_core, name):
                return _safe_getattr_obj(_cfg_core, name)
        except Exception:
            pass

    return None


def _int_or_none(value: object | None) -> int | None:
    """Convierte a int si el tipo es seguro (int/float/bool/str numérico)."""
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            return int(float(s))
        return None
    except Exception:
        return None


def _to_int(value: object | None, default: int) -> int:
    out = _int_or_none(value)
    return out if out is not None else default


def _to_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return default
            return float(s)
        return default
    except Exception:
        return default


def _to_bool(value: object | None, default: bool) -> bool:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "f", "no", "n", "off"}:
                return False
            return default
        return bool(value)
    except Exception:
        return default


def _cfg_get_int(name: str, default: int) -> int:
    return _to_int(_cfg_get_attr(name), default)


def _cfg_get_float(name: str, default: float) -> float:
    return _to_float(_cfg_get_attr(name), default)


def _cfg_get_bool(name: str, default: bool) -> bool:
    return _to_bool(_cfg_get_attr(name), default)


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

# Fallback de lookup_title
_LOOKUP_TITLE_FALLBACK_ENABLED: Final[bool] = _cfg_get_bool("ANALYZE_LOOKUP_TITLE_FALLBACK_ENABLED", True)
_LOOKUP_TITLE_FALLBACK_MAX_CHARS: Final[int] = _cfg_get_int("ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS", 180)

# Métrica heurística
_STRONG_POTENTIAL_CONTRADICTION_ENABLED: Final[bool] = _cfg_get_bool(
    "ANALYZE_METRICS_STRONG_POTENTIAL_CONTRADICTION_ENABLED",
    True,
)

_REASON_FALLBACK: Final[str] = "scoring did not provide a usable reason"
_VALID_DECISIONS: Final[set[str]] = {"KEEP", "MAYBE", "DELETE", "UNKNOWN"}
_LOG_TAG: Final[str] = "analyze_core"
_METRIC_PREFIX: Final[str] = "analyze_core."


# =============================================================================
# Métricas (best-effort / nunca rompen)
# =============================================================================
class _IncFn(Protocol):
    def __call__(self, name: str, *, value: int = 1) -> None: ...


class _ObserveSecondsFn(Protocol):
    def __call__(self, name: str, *, seconds: float) -> None: ...


# Bind thread-safe (evita races)
_METRICS_BIND_ONCE_DONE: bool = False
_METRICS_BIND_LOCK = threading.Lock()
_METRICS_INC_FN: _IncFn | None = None
_METRICS_OBS_FN: _ObserveSecondsFn | None = None


def _metrics_bind_once() -> None:
    global _METRICS_BIND_ONCE_DONE, _METRICS_INC_FN, _METRICS_OBS_FN

    acquired = False
    try:
        acquired = _METRICS_BIND_LOCK.acquire()

        if _METRICS_BIND_ONCE_DONE:
            return
        _METRICS_BIND_ONCE_DONE = True

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
    finally:
        if acquired:
            try:
                _METRICS_BIND_LOCK.release()
            except Exception:
                pass


def _m(name: str, value: int = 1) -> None:
    # No emitir métricas con value=0 (evita ruido y edge cases)
    if value == 0:
        return
    if not _METRICS_ENABLED or _rm is None:
        return

    full = f"{_METRIC_PREFIX}{name}"

    try:
        did_call = False

        if _METRICS_LAZY_BIND_ENABLED:
            _metrics_bind_once()
            fn = _METRICS_INC_FN
            if fn is not None:
                fn(full, value=value)
                did_call = True

        if did_call:
            return

        fn_obj = getattr(_rm, "inc", None) or getattr(_rm, "counter_inc", None)
        if callable(fn_obj):
            cast(_IncFn, fn_obj)(full, value=value)
    except Exception:
        return


def _m_obs(name: str, seconds: float) -> None:
    if not _METRICS_ENABLED or _rm is None:
        return

    full = f"{_METRIC_PREFIX}{name}"

    try:
        did_call = False

        if _METRICS_LAZY_BIND_ENABLED:
            _metrics_bind_once()
            fn = _METRICS_OBS_FN
            if fn is not None:
                fn(full, seconds=seconds)
                did_call = True

        if did_call:
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        from time import monotonic

        _m_obs(self._name, monotonic() - self._t0)


# =============================================================================
# Tipos públicos
# =============================================================================
class AnalysisRow(TypedDict, total=False):
    source: str
    library: str
    title: str
    year: int | None

    imdb_rating: float | None
    imdb_bayes: float | None
    rt_score: int | None
    metacritic_score: int | None
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


# =============================================================================
# Lookup identity (NEW) + backwards-compatible lookup_title
# =============================================================================
def _get_lookup_identity(movie: MovieInput, *, trace: Callable[[str], None]) -> tuple[str, int | None, str | None]:
    """
    Devuelve (lookup_title, lookup_year, imdb_id_hint_coalesced).

    - lookup_title: string normalizado para lookup
    - lookup_year: año coalesced (puede venir del filename/path), SOLO para OMDb fetch
    - imdb_id: si coalesce lo detecta (útil para el futuro; no cambia decisión aquí)
    """
    # 1) Intento: normalizado “oficial” del MovieInput
    try:
        primary = (movie.normalized_title_for_lookup() or "").strip()
    except Exception:
        primary = ""

    # 2) Coalesce identidad (título/año/imdb) usando hints del path/url (best-effort)
    try:
        extra = getattr(movie, "extra", {}) or {}
        source_url = ""
        if isinstance(extra, dict):
            v = extra.get("source_url")
            if isinstance(v, str):
                source_url = v

        hint = f"{movie.file_path or ''} {source_url}".strip()

        title2, year2, imdb2 = coalesce_movie_identity(
            title=movie.title or "",
            year=movie.year,
            file_path=hint,
            imdb_id_hint=movie.imdb_id_hint,
        )
    except Exception:
        title2, year2, imdb2 = (movie.title or ""), movie.year, movie.imdb_id_hint

    # Si el primary existe, lo usamos como lookup_title, pero devolvemos year2/imdb2 igualmente
    if primary:
        return primary, year2, imdb2

    if not _LOOKUP_TITLE_FALLBACK_ENABLED:
        return "", year2, imdb2

    fb = _collapse_spaces(title2 or "")
    if not fb:
        return "", year2, imdb2

    if len(fb) > _LOOKUP_TITLE_FALLBACK_MAX_CHARS:
        fb = fb[:_LOOKUP_TITLE_FALLBACK_MAX_CHARS].rstrip()

    trace(f"lookup_title fallback | used coalesced title (len={len(fb)})")
    return fb, year2, imdb2


def _get_lookup_title(movie: MovieInput, *, trace: Callable[[str], None]) -> str:
    """
    Backwards-compatible: devuelve SOLO el lookup_title (str).
    """
    t, _y, _imdb = _get_lookup_identity(movie, trace=trace)
    return t


# =============================================================================
# Logging/tracing centralizado (sin romper)
# =============================================================================
def _make_tracer(analysis_trace: TraceCallable | None) -> Callable[[str], None]:
    def _trace(msg: str) -> None:
        clipped = _clip(msg, max_len=_TRACE_LINE_MAX_CHARS)

        if analysis_trace is not None:
            try:
                analysis_trace(clipped)
            except Exception:
                pass
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
    try:
        scoring_obj: object = compute_scoring(
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rt_score=rt_score,
            year=year,
            metacritic_score=metacritic_score,
        )

        if not isinstance(scoring_obj, Mapping):
            _m("scoring.fail", 1)
            trace("scoring fail | compute_scoring returned non-mapping -> UNKNOWN")
            return "UNKNOWN", "compute_scoring returned non-mapping", None, None

        scoring = cast(Mapping[str, object], scoring_obj)

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
        return decision, reason, imdb_bayes, scoring

    except Exception as exc:
        _m("scoring.fail", 1)
        trace(f"scoring fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")
        return "UNKNOWN", "compute_scoring failed", None, None


def _safe_detect_misidentified(
    *,
    detect_title: str,
    detect_year: int | None,
    plex_imdb_id: str | None,
    omdb_data: Mapping[str, object],
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
    trace: Callable[[str], None],
) -> str:
    trace("misidentified | running")
    try:
        out_obj: object = detect_misidentified(
            plex_title=detect_title,
            plex_year=detect_year,
            plex_imdb_id=plex_imdb_id,
            omdb_data=omdb_data,
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rt_score=rt_score,
        )
    except Exception as exc:
        _m("misidentified.fail", 1)
        trace(f"misidentified fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")
        return ""

    if isinstance(out_obj, str):
        return out_obj
    if out_obj is None:
        return ""
    try:
        return str(out_obj)
    except Exception:
        return ""


# =============================================================================
# Phase B (OMDb)
# =============================================================================
@dataclass(frozen=True)
class _PhaseBResult:
    used_omdb: bool
    omdb_usable: bool
    omdb_data: dict[str, object]
    imdb_rating: float | None
    imdb_votes: int | None
    rt_score: int | None
    metacritic_score: int | None
    decision: str
    reason: str
    imdb_bayes: float | None
    scoring: Mapping[str, object] | None


def _coerce_ratings_tuple(out_obj: object) -> tuple[float | None, int | None, int | None, int | None]:
    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None
    metacritic_score: int | None = None

    if not isinstance(out_obj, tuple):
        return None, None, None, None

    if len(out_obj) >= 1 and (isinstance(out_obj[0], (int, float)) or out_obj[0] is None):
        imdb_rating = float(out_obj[0]) if isinstance(out_obj[0], (int, float)) else None

    if len(out_obj) >= 2 and (isinstance(out_obj[1], int) or out_obj[1] is None):
        imdb_votes = out_obj[1] if isinstance(out_obj[1], int) else None

    if len(out_obj) >= 3 and (isinstance(out_obj[2], int) or out_obj[2] is None):
        rt_score = out_obj[2] if isinstance(out_obj[2], int) else None

    if len(out_obj) >= 4 and (isinstance(out_obj[3], int) or out_obj[3] is None):
        metacritic_score = out_obj[3] if isinstance(out_obj[3], int) else None

    return imdb_rating, imdb_votes, rt_score, metacritic_score


def _run_phaseB_omdb(
    *,
    movie: MovieInput,
    lookup_title: str,
    lookup_year: int | None,
    fetch_omdb: FetchOmdbCallable,
    metacritic_score: int | None,
    trace: Callable[[str], None],
) -> _PhaseBResult:
    used_omdb = False
    omdb_usable = False
    omdb_data: dict[str, object] = {}
    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None

    metacritic_from_omdb: int | None = None
    metacritic_used: int | None = metacritic_score

    if not lookup_title:
        _m("phaseB.skipped_empty_title", 1)
        trace("phaseB | skipped omdb (empty lookup_title)")
        decisionB, reasonB, imdb_bayesB, scoringB = _compute_scoring_safe(
            imdb_rating=None,
            imdb_votes=None,
            rt_score=None,
            year=movie.year,
            metacritic_score=metacritic_used,
            trace=trace,
        )
        return _PhaseBResult(
            used_omdb=False,
            omdb_usable=False,
            omdb_data={},
            imdb_rating=None,
            imdb_votes=None,
            rt_score=None,
            metacritic_score=metacritic_used,
            decision=decisionB,
            reason=reasonB,
            imdb_bayes=imdb_bayesB,
            scoring=scoringB,
        )

    _m("phaseB.fetch_attempted", 1)
    trace("phaseB | fetching omdb (needed)")

    try:
        raw = fetch_omdb(lookup_title, lookup_year)
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
            out_obj: object = extract_ratings_from_omdb(omdb_data)
            imdb_rating, imdb_votes, rt_score, metacritic_from_omdb = _coerce_ratings_tuple(out_obj)

            if metacritic_used is None:
                metacritic_used = metacritic_from_omdb

            _m("ratings.extract_ok", 1)
            trace(
                "ratings ok | "
                f"imdb_rating={_safe_str(imdb_rating, max_len=32)} "
                f"votes={_safe_str(imdb_votes, max_len=32)} "
                f"rt={_safe_str(rt_score, max_len=32)} "
                f"mc={_safe_str(metacritic_used, max_len=32)}"
            )
        except Exception as exc:
            imdb_rating, imdb_votes, rt_score, metacritic_from_omdb = None, None, None, None
            _m("ratings.extract_fail", 1)
            trace(f"ratings fail | err={_safe_str(exc, max_len=_TRACE_LINE_MAX_CHARS)}")

    trace("phaseB | scoring with omdb-derived signals")
    decisionB, reasonB, imdb_bayesB, scoringB = _compute_scoring_safe(
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        year=movie.year,
        metacritic_score=metacritic_used,
        trace=trace,
    )

    return _PhaseBResult(
        used_omdb=used_omdb,
        omdb_usable=omdb_usable,
        omdb_data=omdb_data,
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        metacritic_score=metacritic_used,
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
    _m("calls", 1)

    with _RM_Timer("seconds"):
        trace = _make_tracer(analysis_trace)

        lib = movie.library or ""
        title_raw = movie.title or ""

        lookup_title, lookup_year, lookup_imdb = _get_lookup_identity(movie, trace=trace)

        trace(
            "start | "
            f"src={_safe_str(movie.source, max_len=32)} "
            f"lib={_safe_str(lib, max_len=80)} "
            f"title={_safe_str(title_raw, max_len=120)} "
            f"year={_safe_str(movie.year, max_len=16)} "
            f"lookup_title={_safe_str(lookup_title, max_len=120)} "
            f"lookup_year={_safe_str(lookup_year, max_len=16)}"
        )

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
        metacritic_final: int | None = metacritic_score

        if strongA:
            _m("decision_strong_without_omdb", 1)
            trace(f"phaseB | skipped omdb (decision strong: {decisionA})")

            if _STRONG_POTENTIAL_CONTRADICTION_ENABLED:
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
                lookup_year=lookup_year,
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
            metacritic_final = phaseB.metacritic_score

            decision_final = phaseB.decision
            reason_final = phaseB.reason
            imdb_bayes = phaseB.imdb_bayes
            scoring_final = phaseB.scoring

        decision_phaseB = decision_final

        if used_omdb and decision_phaseA != decision_phaseB:
            _m("decision_changed_after_omdb", 1)
            _m(f"decision_changed_after_omdb.to_{decision_phaseB.lower()}", 1)
            trace(f"decision changed | {decision_phaseA} -> {decision_phaseB}")

        # Evitamos emitir métricas con value=0
        _m("used_omdb.true", 1 if used_omdb else 0)
        _m("used_omdb.false", 1 if not used_omdb else 0)

        misidentified_hint = ""
        omdb_data_map: Mapping[str, object] | None = omdb_data if omdb_data else None

        if omdb_data_map is not None:
            _m("misidentified.ran", 1)

            detect_title = plex_title if isinstance(plex_title, str) and plex_title.strip() else title_raw
            detect_year = plex_year if isinstance(plex_year, int) else movie.year

            misidentified_hint = _safe_detect_misidentified(
                detect_title=detect_title,
                detect_year=detect_year,
                plex_imdb_id=movie.imdb_id_hint if isinstance(movie.imdb_id_hint, str) else None,
                omdb_data=omdb_data_map,
                imdb_rating=imdb_rating,
                imdb_votes=imdb_votes,
                rt_score=rt_score,
                trace=trace,
            )

            if misidentified_hint.strip():
                _m("misidentified.yes", 1)
                trace("misidentified | YES")
            else:
                _m("misidentified.no", 1)
                trace("misidentified | no")
        else:
            trace("misidentified | skipped (no usable omdb data)")

        if decision_phaseB == "DELETE" and misidentified_hint.strip():
            _m("delete_but_misidentified", 1)
            trace("inconsistency | DELETE but misidentified_hint is present")

        votes_i = imdb_votes if isinstance(imdb_votes, int) else None
        if (
            decision_phaseB == "DELETE"
            and isinstance(imdb_rating, (int, float))
            and float(imdb_rating) >= float(_INCONS_DELETE_MIN_RATING)
            and votes_i is not None
            and votes_i >= _INCONS_DELETE_MIN_VOTES
        ):
            _m("inconsistency.delete_with_high_imdb", 1)
            trace(
                "inconsistency | DELETE with high imdb "
                f"(rating={imdb_rating} votes={votes_i}) "
                f"min_rating={_INCONS_DELETE_MIN_RATING} min_votes={_INCONS_DELETE_MIN_VOTES}"
            )

        votes_i2 = imdb_votes if isinstance(imdb_votes, int) else None
        if (
            decision_phaseB == "KEEP"
            and isinstance(imdb_rating, (int, float))
            and float(imdb_rating) <= float(_INCONS_KEEP_MAX_RATING)
            and votes_i2 is not None
            and votes_i2 >= _INCONS_KEEP_MIN_VOTES
        ):
            _m("inconsistency.keep_with_low_imdb", 1)
            trace(
                "inconsistency | KEEP with low imdb "
                f"(rating={imdb_rating} votes={votes_i2}) "
                f"max_rating={_INCONS_KEEP_MAX_RATING} min_votes={_INCONS_KEEP_MIN_VOTES}"
            )

        has_signals = any(v is not None for v in (imdb_rating, imdb_votes, rt_score, metacritic_final))
        reason_code = _derive_reason_code(
            scoring=scoring_final,
            decision=decision_phaseB,
            used_omdb=used_omdb,
            omdb_usable=omdb_usable,
            has_signals=has_signals,
        )

        row: AnalysisRow = {
            "source": movie.source,
            "library": movie.library,
            "title": movie.title,
            "year": movie.year,
            "imdb_rating": imdb_rating,
            "imdb_bayes": imdb_bayes,
            "rt_score": rt_score,
            "metacritic_score": metacritic_final,
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

        # Preferimos imdb_id_hint real del MovieInput; si no existe, usamos el coalesced.
        if isinstance(movie.imdb_id_hint, str) and movie.imdb_id_hint.strip():
            row["imdb_id_hint"] = movie.imdb_id_hint.strip()
        elif isinstance(lookup_imdb, str) and lookup_imdb.strip():
            row["imdb_id_hint"] = lookup_imdb.strip()

        trace("done")
        return row
