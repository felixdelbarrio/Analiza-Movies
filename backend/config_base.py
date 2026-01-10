"""
backend/config_base.py

- Carga .env UNA vez
- Define PATHS base (BASE_DIR/DATA_DIR) temprano
- Helpers defensivos (_get_env_*, _cap_*, parsers)
- Flags globales (DEBUG_MODE/SILENT_MODE/LOG_LEVEL/HTTP_DEBUG)
- REPORTS_DIR_PATH
- LOGGER_FILE_* + congelado de LOGGER_FILE_PATH

Este módulo NO debe importar config_*.py para evitar ciclos.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# En producción suele ser deseable NO sobre-escribir env vars ya definidas.
load_dotenv(override=False)

# Import tardío para minimizar riesgo de ciclos
from backend import logger as _logger  # noqa: E402


# ============================================================
# Paths base (DEBEN definirse pronto)
# ============================================================

# Directorio del módulo backend/
BASE_DIR: Final[Path] = Path(__file__).resolve().parent

# Raíz del proyecto (un nivel por encima de backend/)
PROJECT_DIR: Final[Path] = BASE_DIR.parent

# data/ en la raíz del proyecto
_DATA_DIR_RAW: Final[str] = (os.getenv("DATA_DIR") or "data").strip() or "data"
_DATA_DIR_CANDIDATE = Path(_DATA_DIR_RAW)
DATA_DIR: Final[Path] = (
    _DATA_DIR_CANDIDATE if _DATA_DIR_CANDIDATE.is_absolute() else (PROJECT_DIR / _DATA_DIR_CANDIDATE)
)


# ============================================================
# Helpers: parseo defensivo de env vars
# ============================================================

_TRUE_SET: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}
_FALSE_SET: Final[set[str]] = {"0", "false", "f", "no", "n", "off"}


def _clean_env_raw(v: object | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if len(s) >= 2 and (s[0] == s[-1]) and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s or None


def _get_env_str(name: str, default: str | None = None) -> str | None:
    v = _clean_env_raw(os.getenv(name))
    return default if v is None else v


def _get_env_int(name: str, default: int) -> int:
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        _logger.warning(f"Invalid int for {name!r}: {v!r}, using default {default}", always=True)
        return default


def _get_env_float(name: str, default: float) -> float:
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        _logger.warning(f"Invalid float for {name!r}: {v!r}, using default {default}", always=True)
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default
    s = v.strip().lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False
    _logger.warning(f"Invalid bool for {name!r}: {v!r}, using default {default}", always=True)
    return default


def _get_env_enum_str(
    name: str,
    *,
    default: str,
    allowed: set[str],
    normalize: bool = True,
) -> str:
    raw = _get_env_str(name, None)
    if raw is None:
        return default
    s = raw.strip()
    if normalize:
        s = s.lower()
    if s in allowed:
        return s
    _logger.warning(
        f"Invalid value for {name!r}: {raw!r}. Allowed={sorted(allowed)}. Using default {default!r}.",
        always=True,
    )
    return default


def _cap_int(name: str, value: int, *, min_v: int, max_v: int) -> int:
    if value < min_v:
        _logger.warning(f"{name} < {min_v}; forcing to {min_v}", always=True)
        return min_v
    if value > max_v:
        _logger.warning(f"{name} too high; capping to {max_v}", always=True)
        return max_v
    return value


def _cap_float_min(name: str, value: float, *, min_v: float) -> float:
    if value < min_v:
        _logger.warning(f"{name} < {min_v}; forcing to {min_v}", always=True)
        return min_v
    return value


def _log_config_debug(label: str, value: object, *, debug_mode: bool, silent_mode: bool) -> None:
    if not debug_mode or silent_mode:
        return
    try:
        _logger.info(f"{label}: {value}")
    except Exception:
        print(f"{label}: {value}")


def _parse_env_kv_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cleaned = (raw or "").strip().strip('"').strip("'").strip()
    if not cleaned:
        return out

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs:
                        out[ks] = vs
            else:
                _logger.warning(
                    f"Invalid dict for env map: expected JSON object, got {type(obj).__name__}",
                    always=True,
                )
            return out
        except Exception as exc:
            _logger.warning(
                f"Invalid JSON for env map; falling back to 'k:v' parsing. err={exc!r}",
                always=True,
            )

    for part in cleaned.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            _logger.warning(f"Invalid map chunk (missing ':') ignored: {chunk!r}", always=True)
            continue
        k, v = chunk.split(":", 1)
        ks = k.strip()
        vs = v.strip()
        if not ks or not vs:
            _logger.warning(f"Invalid map chunk (empty key/value) ignored: {chunk!r}", always=True)
            continue
        out[ks] = vs

    return out


def _parse_env_csv_tokens(raw: str) -> list[str]:
    cleaned = (raw or "").strip().strip('"').strip("'").strip()
    if not cleaned:
        return []

    parts = [p.strip().lower() for p in cleaned.split(",") if p.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


# ============================================================
# MODO DE EJECUCIÓN
# ============================================================

DEBUG_MODE: bool = _get_env_bool("DEBUG_MODE", False)
SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)

HTTP_DEBUG: bool = _get_env_bool("HTTP_DEBUG", False)
LOG_LEVEL: str | None = _get_env_str("LOG_LEVEL", None)

# Ejecutar automáticamente el dashboard tras Plex/DLNA (por defecto: activado)
ANALIZA_AUTO_DASHBOARD: bool = _get_env_bool("ANALIZA_AUTO_DASHBOARD", True)


# ============================================================
# Reports (paths) -> en la raíz del proyecto
# ============================================================

_REPORTS_DIR_RAW: Final[str] = _get_env_str("REPORTS_DIR", "reports") or "reports"
_REPORTS_DIR_PATH_CANDIDATE = Path(_REPORTS_DIR_RAW)
REPORTS_DIR_PATH: Final[Path] = (
    _REPORTS_DIR_PATH_CANDIDATE
    if _REPORTS_DIR_PATH_CANDIDATE.is_absolute()
    else (PROJECT_DIR / _REPORTS_DIR_PATH_CANDIDATE)
)


# ============================================================
# LOGGER (persistencia opcional a fichero por ejecución)
# ============================================================
# Logs se quedan dentro de backend/ (BASE_DIR) a propósito.

LOGGER_FILE_ENABLED: bool = _get_env_bool("LOGGER_FILE_ENABLED", False)

_LOGGER_FILE_DIR_RAW: Final[str] = _get_env_str("LOGGER_FILE_DIR", "logs") or "logs"
_LOGGER_FILE_DIR_CANDIDATE = Path(_LOGGER_FILE_DIR_RAW)
LOGGER_FILE_DIR: Final[Path] = (
    _LOGGER_FILE_DIR_CANDIDATE if _LOGGER_FILE_DIR_CANDIDATE.is_absolute() else (BASE_DIR / _LOGGER_FILE_DIR_CANDIDATE)
)

LOGGER_FILE_PREFIX: Final[str] = _get_env_str("LOGGER_FILE_PREFIX", "run") or "run"
LOGGER_FILE_TIMESTAMP_FORMAT: Final[str] = _get_env_str("LOGGER_FILE_TIMESTAMP_FORMAT", "%Y-%m-%d_%H-%M-%S") or "%Y-%m-%d_%H-%M-%S"
LOGGER_FILE_INCLUDE_PID: bool = _get_env_bool("LOGGER_FILE_INCLUDE_PID", True)

_LOGGER_FILE_PATH_EXPLICIT_RAW: str | None = _get_env_str("LOGGER_FILE_PATH", None)

_LOGGER_FILE_PATH_SENTINEL: Final[object] = object()
_LOGGER_FILE_PATH_CACHED: Path | None | object = _LOGGER_FILE_PATH_SENTINEL


def _sanitize_filename_component(s: str) -> str:
    out_chars: list[str] = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("-", "_", ".", "@"):
            out_chars.append(ch)
        else:
            out_chars.append("_")
    cleaned = "".join(out_chars).strip("._-")
    return cleaned or "run"


def _build_logger_file_path() -> Path | None:
    global _LOGGER_FILE_PATH_CACHED

    if _LOGGER_FILE_PATH_CACHED is not _LOGGER_FILE_PATH_SENTINEL:
        return None if _LOGGER_FILE_PATH_CACHED is None else _LOGGER_FILE_PATH_CACHED  # type: ignore[return-value]

    try:
        if not LOGGER_FILE_ENABLED:
            _LOGGER_FILE_PATH_CACHED = None
            return None

        env_path = _clean_env_raw(os.getenv("LOGGER_FILE_PATH"))
        if env_path:
            p = Path(env_path)
            resolved = (p if p.is_absolute() else (BASE_DIR / p)).resolve()
            _LOGGER_FILE_PATH_CACHED = resolved
            return resolved

        if isinstance(_LOGGER_FILE_PATH_EXPLICIT_RAW, str) and _LOGGER_FILE_PATH_EXPLICIT_RAW.strip():
            p = Path(_LOGGER_FILE_PATH_EXPLICIT_RAW.strip())
            resolved = (p if p.is_absolute() else (BASE_DIR / p)).resolve()
            os.environ["LOGGER_FILE_PATH"] = str(resolved)
            _LOGGER_FILE_PATH_CACHED = resolved
            return resolved

        ts = datetime.now().strftime(LOGGER_FILE_TIMESTAMP_FORMAT)
        ts = _sanitize_filename_component(ts)

        prefix = _sanitize_filename_component(LOGGER_FILE_PREFIX)
        pid_part = f"_{os.getpid()}" if LOGGER_FILE_INCLUDE_PID else ""

        filename = f"{prefix}_{ts}{pid_part}.log"
        resolved = (LOGGER_FILE_DIR / filename).resolve()

        os.environ["LOGGER_FILE_PATH"] = str(resolved)
        _LOGGER_FILE_PATH_CACHED = resolved
        return resolved

    except Exception as exc:
        _logger.warning(f"LOGGER_FILE_PATH build failed: {exc!r}", always=True)
        _LOGGER_FILE_PATH_CACHED = None
        return None


LOGGER_FILE_PATH: Path | None = _build_logger_file_path()
