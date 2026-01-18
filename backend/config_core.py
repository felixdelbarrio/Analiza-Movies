from __future__ import annotations

from backend.config_base import (
    _cap_float_min,
    _cap_int,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
)

# ============================================================
# CORE (backend/analyze_input_core.py)
# ============================================================

ANALYZE_TRACE_REASON_MAX_CHARS: int = _cap_int(
    "ANALYZE_TRACE_REASON_MAX_CHARS",
    _get_env_int("ANALYZE_TRACE_REASON_MAX_CHARS", 140),
    min_v=40,
    max_v=5000,
)

ANALYZE_CORE_METRICS_ENABLED: bool = _get_env_bool("ANALYZE_CORE_METRICS_ENABLED", True)

# ------------------------------------------------------------
# Inconsistencias (solo observación; NO cambia decisiones)
# ------------------------------------------------------------

ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_RATING: float = _cap_float_min(
    "ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_RATING",
    _get_env_float("ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_RATING", 7.5),
    min_v=0.0,
)

ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_VOTES: int = _cap_int(
    "ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_VOTES",
    _get_env_int("ANALYZE_INCONSISTENCY_DELETE_IMDB_MIN_VOTES", 10_000),
    min_v=0,
    max_v=50_000_000,
)

# KEEP con rating muy bajo (posible mismatch / mala señal) – observación
ANALYZE_INCONSISTENCY_KEEP_IMDB_MAX_RATING: float = _cap_float_min(
    "ANALYZE_INCONSISTENCY_KEEP_IMDB_MAX_RATING",
    _get_env_float("ANALYZE_INCONSISTENCY_KEEP_IMDB_MAX_RATING", 4.5),
    min_v=0.0,
)

ANALYZE_INCONSISTENCY_KEEP_IMDB_MIN_VOTES: int = _cap_int(
    "ANALYZE_INCONSISTENCY_KEEP_IMDB_MIN_VOTES",
    _get_env_int("ANALYZE_INCONSISTENCY_KEEP_IMDB_MIN_VOTES", 25_000),
    min_v=0,
    max_v=50_000_000,
)

# ------------------------------------------------------------
# Fallback lookup_title: si normalized_title_for_lookup() devuelve vacío
# ------------------------------------------------------------

ANALYZE_LOOKUP_TITLE_FALLBACK_ENABLED: bool = _get_env_bool(
    "ANALYZE_LOOKUP_TITLE_FALLBACK_ENABLED",
    True,
)

ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS: int = _cap_int(
    "ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS",
    _get_env_int("ANALYZE_LOOKUP_TITLE_FALLBACK_MAX_CHARS", 180),
    min_v=20,
    max_v=1000,
)

# ------------------------------------------------------------
# Métricas/observabilidad extra
# ------------------------------------------------------------

# Emitir métricas “potencialmente habría cambiado con OMDb” (solo heurística local / no hace red)
ANALYZE_METRICS_STRONG_POTENTIAL_CONTRADICTION_ENABLED: bool = _get_env_bool(
    "ANALYZE_METRICS_STRONG_POTENTIAL_CONTRADICTION_ENABLED",
    True,
)

# ------------------------------------------------------------
# run_metrics binding optimization
# ------------------------------------------------------------

# Cachea el binding de funciones de run_metrics (reduce getattr en hot path)
ANALYZE_METRICS_LAZY_BIND_ENABLED: bool = _get_env_bool(
    "ANALYZE_METRICS_LAZY_BIND_ENABLED",
    True,
)
