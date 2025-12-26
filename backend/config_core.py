from __future__ import annotations

from backend.config_base import _cap_int, _get_env_bool, _get_env_int

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