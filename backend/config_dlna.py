from __future__ import annotations

from backend.config_base import (
    _cap_float_min,
    _cap_int,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    _get_env_str,
    _parse_env_csv_tokens,
)

# ============================================================
# DLNA (browse workers + circuit breaker)
# ============================================================

DLNA_SCAN_WORKERS: int = _cap_int("DLNA_SCAN_WORKERS", _get_env_int("DLNA_SCAN_WORKERS", 2), min_v=1, max_v=8)
DLNA_BROWSE_MAX_RETRIES: int = _cap_int("DLNA_BROWSE_MAX_RETRIES", _get_env_int("DLNA_BROWSE_MAX_RETRIES", 2), min_v=0, max_v=10)

DLNA_CB_FAILURE_THRESHOLD: int = _cap_int("DLNA_CB_FAILURE_THRESHOLD", _get_env_int("DLNA_CB_FAILURE_THRESHOLD", 5), min_v=1, max_v=50)
DLNA_CB_OPEN_SECONDS: float = _cap_float_min("DLNA_CB_OPEN_SECONDS", _get_env_float("DLNA_CB_OPEN_SECONDS", 20.0), min_v=0.1)

# ============================================================
# DLNA traversal fuses (anti-loop + anti-duplicados)
# ============================================================

DLNA_TRAVERSE_MAX_DEPTH: int = _cap_int("DLNA_TRAVERSE_MAX_DEPTH", _get_env_int("DLNA_TRAVERSE_MAX_DEPTH", 30), min_v=1, max_v=500)
DLNA_TRAVERSE_MAX_CONTAINERS: int = _cap_int("DLNA_TRAVERSE_MAX_CONTAINERS", _get_env_int("DLNA_TRAVERSE_MAX_CONTAINERS", 20_000), min_v=100, max_v=5_000_000)
DLNA_TRAVERSE_MAX_ITEMS_TOTAL: int = _cap_int("DLNA_TRAVERSE_MAX_ITEMS_TOTAL", _get_env_int("DLNA_TRAVERSE_MAX_ITEMS_TOTAL", 300_000), min_v=1_000, max_v=50_000_000)
DLNA_TRAVERSE_MAX_EMPTY_PAGES: int = _cap_int("DLNA_TRAVERSE_MAX_EMPTY_PAGES", _get_env_int("DLNA_TRAVERSE_MAX_EMPTY_PAGES", 3), min_v=1, max_v=100)
DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER: int = _cap_int("DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER", _get_env_int("DLNA_TRAVERSE_MAX_PAGES_PER_CONTAINER", 20_000), min_v=10, max_v=5_000_000)

# ============================================================
# DLNA Discovery (SSDP) + device description fetch
# ============================================================

DLNA_DISCOVERY_TIMEOUT_SECONDS: float = _cap_float_min("DLNA_DISCOVERY_TIMEOUT_SECONDS", _get_env_float("DLNA_DISCOVERY_TIMEOUT_SECONDS", 3.0), min_v=0.3)
DLNA_DISCOVERY_MX: int = _cap_int("DLNA_DISCOVERY_MX", _get_env_int("DLNA_DISCOVERY_MX", 2), min_v=1, max_v=10)
DLNA_DISCOVERY_ST: str = _get_env_str("DLNA_DISCOVERY_ST", "ssdp:all") or "ssdp:all"

DLNA_DEVICE_DESC_TIMEOUT_SECONDS: float = _cap_float_min("DLNA_DEVICE_DESC_TIMEOUT_SECONDS", _get_env_float("DLNA_DEVICE_DESC_TIMEOUT_SECONDS", 3.0), min_v=0.2)
DLNA_DEVICE_DESC_MAX_BYTES: int = _cap_int("DLNA_DEVICE_DESC_MAX_BYTES", _get_env_int("DLNA_DEVICE_DESC_MAX_BYTES", 2_000_000), min_v=16_384, max_v=50_000_000)

_DLNA_DISCOVERY_DENY_TOKENS_RAW: str = _get_env_str(
    "DLNA_DISCOVERY_DENY_TOKENS",
    "internetgatewaydevice,wanipconnection,wanpppconnection,wandevice,igd",
) or "internetgatewaydevice,wanipconnection,wanpppconnection,wandevice,igd"
DLNA_DISCOVERY_DENY_TOKENS: list[str] = _parse_env_csv_tokens(_DLNA_DISCOVERY_DENY_TOKENS_RAW)

_DLNA_DISCOVERY_ALLOW_HINT_TOKENS_RAW: str = _get_env_str(
    "DLNA_DISCOVERY_ALLOW_HINT_TOKENS",
    "mediaserver,contentdirectory",
) or "mediaserver,contentdirectory"
DLNA_DISCOVERY_ALLOW_HINT_TOKENS: list[str] = _parse_env_csv_tokens(_DLNA_DISCOVERY_ALLOW_HINT_TOKENS_RAW)

# ============================================================
# DLNA-only: switches de enriquecimiento (sin afectar Plex/otros)
# ============================================================

# Si False, el pipeline DLNA no intentará enriquecer con Wiki (solo parseo local + OMDb si aplica).
DLNA_ENABLE_WIKI_ENRICHMENT: bool = _get_env_bool("DLNA_ENABLE_WIKI_ENRICHMENT", True)

# Si True, el pipeline DLNA puede usar Wiki para "resolver" un imdb_id (vía Wikidata) cuando OMDb no ayuda.
# (La implementación real ocurre en collection_analysis; aquí solo knob)
DLNA_ENABLE_WIKI_IMDB_RESOLUTION: bool = _get_env_bool("DLNA_ENABLE_WIKI_IMDB_RESOLUTION", True)