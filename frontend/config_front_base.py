from __future__ import annotations

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

import os
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
    k: v for k, v in (dotenv_values(_ENV_FRONT_PATH).items() if _ENV_FRONT_PATH.exists() else []) if v is not None
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


def _resolve_dir(raw: str, *, base: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p)


# ---------------------------------------------------------------------
# Flags del front (si quieres controlarlo desde .env.front)
# ---------------------------------------------------------------------

FRONT_DEBUG: bool = _get_env_bool("FRONT_DEBUG", False)

# ---------------------------------------------------------------------
# Paths: data/ y reports/ (defaults alineados con tu estructura actual)
# ---------------------------------------------------------------------

_DATA_DIR_RAW: Final[str] = _get_env_str("FRONT_DATA_DIR", "data") or "data"
_REPORTS_DIR_RAW: Final[str] = _get_env_str("FRONT_REPORTS_DIR", "reports") or "reports"

DATA_DIR: Final[Path] = _resolve_dir(_DATA_DIR_RAW, base=PROJECT_DIR)
REPORTS_DIR: Final[Path] = _resolve_dir(_REPORTS_DIR_RAW, base=PROJECT_DIR)