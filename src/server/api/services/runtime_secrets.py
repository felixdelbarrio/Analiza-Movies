from __future__ import annotations

import os
from threading import Lock

from shared.runtime_profiles import RuntimeConfig, SourceProfile

_LOCK = Lock()
_PROFILE_TOKENS: dict[str, str] = {}
_OMDB_API_KEYS: str | None = None


def remember_profile_token(profile_id: str | None, token: str | None) -> None:
    clean_profile_id = str(profile_id or "").strip()
    clean_token = str(token or "").strip()
    if not clean_profile_id or not clean_token:
        return
    with _LOCK:
        _PROFILE_TOKENS[clean_profile_id] = clean_token


def resolve_profile_token(profile: SourceProfile) -> str | None:
    direct_token = str(profile.plex_token or "").strip()
    if direct_token:
        return direct_token
    with _LOCK:
        stored_token = _PROFILE_TOKENS.get(profile.id)
    return None if stored_token is None else stored_token.strip() or None


def remember_omdb_api_keys(value: str | None) -> None:
    clean_value = str(value or "").strip()
    with _LOCK:
        global _OMDB_API_KEYS
        _OMDB_API_KEYS = clean_value or None


def resolve_omdb_api_keys(config: RuntimeConfig | None = None) -> str:
    with _LOCK:
        runtime_keys = _OMDB_API_KEYS
    if runtime_keys:
        return runtime_keys

    env_value = str(os.getenv("OMDB_API_KEYS") or "").strip()
    if env_value:
        return env_value

    fallback_value = str(os.getenv("OMDB_API_KEY") or "").strip()
    if fallback_value:
        return fallback_value

    if config is None:
        return ""
    return str(config.omdb_api_keys or "").strip()


def has_omdb_api_keys(config: RuntimeConfig | None = None) -> bool:
    return bool(resolve_omdb_api_keys(config))
