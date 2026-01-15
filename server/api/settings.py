# lectura de env vars + defaults
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import dotenv_values


SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SERVER_DIR.parent
_ENV_SERVER_PATH = PROJECT_DIR / ".env.server"

_ENV: dict[str, str] = {
    k: v
    for k, v in (
        dotenv_values(_ENV_SERVER_PATH).items() if _ENV_SERVER_PATH.exists() else []
    )
    if v is not None
}

_TRUE_SET = {"1", "true", "t", "yes", "y", "on"}
_FALSE_SET = {"0", "false", "f", "no", "n", "off"}


def _env_str(name: str, default: str) -> str:
    raw = _ENV.get(name)
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip()
    return val if val else default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if not raw:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, "")
    if not raw:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "")
    if not raw:
        return default
    s = raw.strip().lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """
    Settings centralizados (env vars). Esto evita `os.getenv(...)` disperso.

    Notas importantes:
    - CORS: si CORS_ORIGINS="*" -> allow_credentials=False para compatibilidad browser.
    - API_RELOAD: default "0" (seguro para producción).
    """

    log_level: str

    cors_origins_raw: str
    cors_allow_credentials: bool

    gzip_min_size: int

    file_cache_max_entries: int
    file_cache_ttl_seconds: float

    file_read_max_attempts: int
    file_read_retry_sleep_s: float

    def cors_allow_origins(self) -> list[str]:
        raw = self.cors_origins_raw.strip()
        if raw == "*":
            return ["*"]
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]

    @staticmethod
    def from_env() -> "Settings":
        cors_raw = _env_str("CORS_ORIGINS", "*")
        allow_origins = (
            ["*"]
            if cors_raw.strip() == "*"
            else [p.strip() for p in cors_raw.split(",") if p.strip()]
        )

        # Regla browser: "*" + credentials=True no es válido
        cors_allow_credentials = True
        if allow_origins == ["*"]:
            cors_allow_credentials = False

        return Settings(
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
            cors_origins_raw=cors_raw,
            cors_allow_credentials=cors_allow_credentials,
            gzip_min_size=_env_int("GZIP_MIN_SIZE", 800),
            file_cache_max_entries=_env_int("FILE_CACHE_MAX_ENTRIES", 16),
            file_cache_ttl_seconds=_env_float("FILE_CACHE_TTL_SECONDS", 0.0),
            file_read_max_attempts=max(1, _env_int("FILE_READ_MAX_ATTEMPTS", 3)),
            file_read_retry_sleep_s=max(
                0.0, _env_float("FILE_READ_RETRY_SLEEP_S", 0.05)
            ),
        )
