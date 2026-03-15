from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Final

import keyring  # type: ignore[import-untyped]
from keyring.errors import KeyringError, PasswordDeleteError  # type: ignore[import-untyped]

from shared.runtime_profiles import SourceProfile

SERVICE_NAME: Final = "analiza-movies"
OMDB_ENTRY: Final = "omdb_api_keys"
PROFILE_TOKEN_PREFIX: Final = "profile_token:"

_UNSET = object()
_LOCK = Lock()
_PROFILE_TOKENS: dict[str, str | None] = {}
_OMDB_API_KEYS: str | None | object = _UNSET

logger = logging.getLogger(__name__)


def _profile_entry(profile_id: str) -> str:
    return f"{PROFILE_TOKEN_PREFIX}{profile_id}"


def _read_secret(entry: str) -> str | None:
    try:
        secret = keyring.get_password(SERVICE_NAME, entry)
    except KeyringError:
        logger.warning("No se pudo leer el secreto '%s' del keyring.", entry)
        return None
    clean_secret = str(secret or "").strip()
    return clean_secret or None


def _write_secret(entry: str, value: str | None) -> None:
    clean_value = str(value or "").strip()
    try:
        if clean_value:
            keyring.set_password(SERVICE_NAME, entry, clean_value)
            return
        try:
            keyring.delete_password(SERVICE_NAME, entry)
        except PasswordDeleteError:
            return
    except KeyringError:
        logger.warning("No se pudo persistir el secreto '%s' en el keyring.", entry)


def reset_runtime_secrets_cache() -> None:
    global _OMDB_API_KEYS
    with _LOCK:
        _PROFILE_TOKENS.clear()
        _OMDB_API_KEYS = _UNSET


def remember_profile_token(profile_id: str | None, token: str | None) -> None:
    clean_profile_id = str(profile_id or "").strip()
    clean_token = str(token or "").strip()
    if not clean_profile_id or not clean_token:
        return
    with _LOCK:
        _PROFILE_TOKENS[clean_profile_id] = clean_token
    _write_secret(_profile_entry(clean_profile_id), clean_token)


def resolve_profile_token(profile: SourceProfile) -> str | None:
    direct_token = str(profile.plex_token or "").strip()
    if direct_token:
        return direct_token

    clean_profile_id = str(profile.id or "").strip()
    if not clean_profile_id:
        return None

    with _LOCK:
        if clean_profile_id in _PROFILE_TOKENS:
            return _PROFILE_TOKENS[clean_profile_id]

    stored_token = _read_secret(_profile_entry(clean_profile_id))
    with _LOCK:
        _PROFILE_TOKENS[clean_profile_id] = stored_token
    return stored_token


def remember_omdb_api_keys(value: str | None) -> None:
    global _OMDB_API_KEYS
    clean_value = str(value or "").strip() or None
    with _LOCK:
        _OMDB_API_KEYS = clean_value
    _write_secret(OMDB_ENTRY, clean_value)


def resolve_omdb_api_keys() -> str:
    global _OMDB_API_KEYS
    with _LOCK:
        runtime_keys = _OMDB_API_KEYS

    if runtime_keys is _UNSET:
        stored_keys = _read_secret(OMDB_ENTRY)
        with _LOCK:
            _OMDB_API_KEYS = stored_keys
            runtime_keys = _OMDB_API_KEYS

    if isinstance(runtime_keys, str) and runtime_keys:
        return runtime_keys

    env_value = str(os.getenv("OMDB_API_KEYS") or "").strip()
    if env_value:
        return env_value

    fallback_value = str(os.getenv("OMDB_API_KEY") or "").strip()
    if fallback_value:
        return fallback_value

    return ""


def has_omdb_api_keys() -> bool:
    return bool(resolve_omdb_api_keys())
