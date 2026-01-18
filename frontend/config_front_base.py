"""
frontend/config_front_base.py

Config base del FRONTEND (desacoplado del backend).

Objetivos:
- Cargar variables de entorno SOLAMENTE desde .env.front (sin fallback a .env).
- Proveer defaults razonables si una clave no existe en .env.front.
- Definir PROJECT_DIR, DATA_DIR, REPORTS_DIR para consumo del front.

Notas:
- No importa nada de backend.
- No reimplementa logging del backend: usa print minimal o el logger del front (si existiera).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Final

from dotenv import dotenv_values

# ---------------------------------------------------------------------
# Ubicación del proyecto (asumimos frontend/ como carpeta dentro de repo)
# ---------------------------------------------------------------------

FRONTEND_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIR: Final[Path] = FRONTEND_DIR.parent

_ENV_FRONT_PATH: Final[Path] = PROJECT_DIR / ".env.front"

# Carga SOLO desde .env.front (si no existe, config vacía -> defaults)
_ENV: Final[dict[str, str]] = {
    k: v
    for k, v in (
        dotenv_values(_ENV_FRONT_PATH).items() if _ENV_FRONT_PATH.exists() else []
    )
    if v is not None
}

# ---------------------------------------------------------------------
# Helpers defensivos (sin backend/logger)
# ---------------------------------------------------------------------

_TRUE_SET: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}
_FALSE_SET: Final[set[str]] = {"0", "false", "f", "no", "n", "off"}


def _clean(v: object | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if len(s) >= 2 and (s[0] == s[-1]) and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s or None


def _get_env_str(name: str, default: str | None = None) -> str | None:
    # Prioridad: .env.front -> env real del proceso (por si se exporta) -> default
    v = _clean(_ENV.get(name))
    if v is not None:
        return v
    v2 = _clean(os.getenv(name))
    if v2 is not None:
        return v2
    return default


def _read_env_front() -> dict[str, str]:
    if not _ENV_FRONT_PATH.exists():
        return {}
    return {k: v for k, v in dotenv_values(_ENV_FRONT_PATH).items() if v is not None}


def _get_env_front_bool(name: str, default: bool) -> bool:
    env = _read_env_front()
    raw = _clean(env.get(name))
    if raw is None:
        return default
    s = raw.lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False
    return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = _get_env_str(name, None)
    if raw is None:
        return default
    s = raw.lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False
    return default


def _get_env_int(name: str, default: int) -> int:
    raw = _get_env_str(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = _get_env_str(name, None)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_dir(raw: str, *, base: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p)


def _sanitize_filename_component(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._-")


# ---------------------------------------------------------------------
# Flags del front (si quieres controlarlo desde .env.front)
# ---------------------------------------------------------------------

FRONT_DEBUG: bool = _get_env_bool("FRONT_DEBUG", False)

# Señaletica de color en tablas
FRONT_GRID_COLORIZE: bool = _get_env_bool("FRONT_GRID_COLORIZE", True)


def get_front_grid_colorize() -> bool:
    return _get_env_front_bool("FRONT_GRID_COLORIZE", True)


# ---------------------------------------------------------------------
# Logging (frontend) - usa .env.front
# ---------------------------------------------------------------------

LOGGER_FILE_ENABLED: bool = _get_env_bool("LOGGER_FILE_ENABLED", False)

_LOGGER_FILE_DIR_RAW: Final[str] = _get_env_str("LOGGER_FILE_DIR", "logs") or "logs"
LOGGER_FILE_DIR: Final[Path] = _resolve_dir(_LOGGER_FILE_DIR_RAW, base=FRONTEND_DIR)

LOGGER_FILE_PREFIX: Final[str] = _get_env_str("LOGGER_FILE_PREFIX", "run") or "run"
LOGGER_FILE_TIMESTAMP_FORMAT: Final[str] = (
    _get_env_str("LOGGER_FILE_TIMESTAMP_FORMAT", "%Y-%m-%d_%H-%M-%S")
    or "%Y-%m-%d_%H-%M-%S"
)
LOGGER_FILE_INCLUDE_PID: bool = _get_env_bool("LOGGER_FILE_INCLUDE_PID", True)

_LOGGER_FILE_PATH_EXPLICIT_RAW: str | None = _get_env_str("LOGGER_FILE_PATH", None)
_LOGGER_FILE_PATH_SENTINEL: Final[object] = object()
_LOGGER_FILE_PATH_CACHED: Path | None | object = _LOGGER_FILE_PATH_SENTINEL


def _build_logger_file_path() -> Path | None:
    global _LOGGER_FILE_PATH_CACHED

    if _LOGGER_FILE_PATH_CACHED is not _LOGGER_FILE_PATH_SENTINEL:
        return None if _LOGGER_FILE_PATH_CACHED is None else _LOGGER_FILE_PATH_CACHED  # type: ignore[return-value]

    if not LOGGER_FILE_ENABLED:
        _LOGGER_FILE_PATH_CACHED = None
        return None

    if (
        isinstance(_LOGGER_FILE_PATH_EXPLICIT_RAW, str)
        and _LOGGER_FILE_PATH_EXPLICIT_RAW.strip()
    ):
        p = Path(_LOGGER_FILE_PATH_EXPLICIT_RAW.strip())
        resolved = p if p.is_absolute() else (FRONTEND_DIR / p)
        _LOGGER_FILE_PATH_CACHED = resolved.resolve()
        return _LOGGER_FILE_PATH_CACHED

    ts = datetime.now().strftime(LOGGER_FILE_TIMESTAMP_FORMAT)
    prefix = _sanitize_filename_component(LOGGER_FILE_PREFIX) or "run"
    pid_part = f"_{os.getpid()}" if LOGGER_FILE_INCLUDE_PID else ""
    filename = f"{prefix}_{ts}{pid_part}.log"
    _LOGGER_FILE_PATH_CACHED = (LOGGER_FILE_DIR / filename).resolve()
    return _LOGGER_FILE_PATH_CACHED


LOGGER_FILE_PATH: Path | None = _build_logger_file_path()

# ---------------------------------------------------------------------
# Modo de ejecución (sin fallback)
#
# Valores válidos:
# - "api"
# - "disk"
# ---------------------------------------------------------------------

_FRONT_MODE_RAW: str = (_get_env_str("FRONT_MODE", "disk") or "disk").lower().strip()
if _FRONT_MODE_RAW not in {"api", "disk"}:
    raise RuntimeError(
        f"FRONT_MODE inválido: {_FRONT_MODE_RAW!r}. Valores válidos: 'api' | 'disk'"
    )

FRONT_MODE: Final[str] = _FRONT_MODE_RAW

# ---------------------------------------------------------------------
# API config (solo relevante si FRONT_MODE == "api")
# ---------------------------------------------------------------------

FRONT_API_BASE_URL: str = (
    _get_env_str("FRONT_API_BASE_URL", "http://localhost:8000")
    or "http://localhost:8000"
).rstrip("/")
FRONT_API_TIMEOUT_S: float = _get_env_float("FRONT_API_TIMEOUT_S", 30.0)
FRONT_API_PAGE_SIZE: int = _get_env_int("FRONT_API_PAGE_SIZE", 2000)
FRONT_API_CACHE_TTL_S: int = _get_env_int("FRONT_API_CACHE_TTL_S", 60)


def save_front_grid_colorize(value: bool) -> None:
    payload = "true" if value else "false"
    lines: list[str] = []
    if _ENV_FRONT_PATH.exists():
        lines = _ENV_FRONT_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FRONT_GRID_COLORIZE="):
            out.append(f"FRONT_GRID_COLORIZE={payload}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"FRONT_GRID_COLORIZE={payload}")
    _ENV_FRONT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Paths: data/ y reports/ (defaults alineados con tu estructura actual)
# ---------------------------------------------------------------------

_DATA_DIR_RAW: Final[str] = _get_env_str("FRONT_DATA_DIR", "data") or "data"
_REPORTS_DIR_RAW: Final[str] = _get_env_str("FRONT_REPORTS_DIR", "reports") or "reports"

DATA_DIR: Final[Path] = _resolve_dir(_DATA_DIR_RAW, base=PROJECT_DIR)
REPORTS_DIR: Final[Path] = _resolve_dir(_REPORTS_DIR_RAW, base=PROJECT_DIR)

# ---------------------------------------------------------------------
# Dashboard / borrado seguro
# (si ya los tienes en otro config, puedes moverlos, pero no lo hago aquí)
# ---------------------------------------------------------------------

DELETE_DRY_RUN: bool = _get_env_bool("DELETE_DRY_RUN", True)
DELETE_REQUIRE_CONFIRM: bool = _get_env_bool("DELETE_REQUIRE_CONFIRM", True)

# ---------------------------------------------------------------------
# MODO DE EJECUCIÓN (compat)
# ---------------------------------------------------------------------

DEBUG_MODE: bool = _get_env_bool("DEBUG_MODE", False)
SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)
