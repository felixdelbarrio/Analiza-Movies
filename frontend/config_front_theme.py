"""
frontend/config_front_theme.py

Config del tema del frontend (desacoplado del backend).

Objetivos:
- Leer FRONT_THEME solo desde .env.front (sin fallback a .env).
- Definir un tema oscuro por defecto.
- Persistir la preferencia en .env.front cuando se guarde desde la UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dotenv import dotenv_values

# ---------------------------------------------------------------------
# Ubicacion del proyecto (asumimos frontend/ como carpeta dentro de repo)
# ---------------------------------------------------------------------

FRONTEND_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIR: Final[Path] = FRONTEND_DIR.parent

_ENV_FRONT_PATH: Final[Path] = PROJECT_DIR / ".env.front"

DEFAULT_THEME_KEY: Final[str] = "noir"
THEME_ALIASES: Final[dict[str, str]] = {
    "dark": "noir",
    "oscuro": "noir",
}
DARK_THEME_KEYS: Final[set[str]] = {"noir", "sapphire", "verdant", "bordeaux"}


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if len(raw) >= 2 and (raw[0] == raw[-1]) and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()
    return raw or None


def _get_env_str(name: str, default: str | None = None) -> str | None:
    if _ENV_FRONT_PATH.exists():
        env = {k: v for k, v in dotenv_values(_ENV_FRONT_PATH).items() if v is not None}
        v = _clean(env.get(name))
        if v is not None:
            return v
    return default


def normalize_theme_key(raw: str | None) -> str:
    if raw is None:
        return DEFAULT_THEME_KEY
    key = raw.strip().lower()
    if not key:
        return DEFAULT_THEME_KEY
    return THEME_ALIASES.get(key, key)


def is_dark_theme(raw: str | None) -> bool:
    key = normalize_theme_key(raw)
    return key in DARK_THEME_KEYS


def get_front_theme() -> str:
    return normalize_theme_key(_get_env_str("FRONT_THEME", DEFAULT_THEME_KEY))


FRONT_THEME: Final[str] = get_front_theme()


def save_front_theme(value: str) -> None:
    payload = normalize_theme_key(value) or DEFAULT_THEME_KEY
    lines: list[str] = []
    if _ENV_FRONT_PATH.exists():
        lines = _ENV_FRONT_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FRONT_THEME="):
            out.append(f"FRONT_THEME={payload}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"FRONT_THEME={payload}")
    _ENV_FRONT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
