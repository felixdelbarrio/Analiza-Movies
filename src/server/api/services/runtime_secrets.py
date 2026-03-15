from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from shared.runtime_profiles import DATA_DIR, SourceProfile

SECRETS_PATH = DATA_DIR / "runtime_secrets.json"

_LOCK = Lock()
_PROFILE_TOKENS: dict[str, str] = {}
_OMDB_API_KEYS: str | None = None
_CACHE_LOADED = False


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_store(path: Path, *, omdb_api_keys: str | None, profile_tokens: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "omdb_api_keys": omdb_api_keys or "",
        "profile_tokens": profile_tokens,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ensure_loaded_locked() -> None:
    global _CACHE_LOADED, _OMDB_API_KEYS, _PROFILE_TOKENS
    if _CACHE_LOADED:
        return
    payload = _read_store(SECRETS_PATH)
    raw_tokens = payload.get("profile_tokens")
    if isinstance(raw_tokens, dict):
        _PROFILE_TOKENS = {
            str(profile_id).strip(): str(token).strip()
            for profile_id, token in raw_tokens.items()
            if str(profile_id).strip() and str(token).strip()
        }
    else:
        _PROFILE_TOKENS = {}
    raw_omdb = str(payload.get("omdb_api_keys") or "").strip()
    _OMDB_API_KEYS = raw_omdb or None
    _CACHE_LOADED = True


def _persist_locked() -> None:
    _write_store(
        SECRETS_PATH,
        omdb_api_keys=_OMDB_API_KEYS,
        profile_tokens=_PROFILE_TOKENS,
    )


def reset_runtime_secrets_cache() -> None:
    global _CACHE_LOADED, _OMDB_API_KEYS, _PROFILE_TOKENS
    with _LOCK:
        _CACHE_LOADED = False
        _OMDB_API_KEYS = None
        _PROFILE_TOKENS = {}


def remember_profile_token(profile_id: str | None, token: str | None) -> None:
    clean_profile_id = str(profile_id or "").strip()
    clean_token = str(token or "").strip()
    if not clean_profile_id or not clean_token:
        return
    with _LOCK:
        _ensure_loaded_locked()
        _PROFILE_TOKENS[clean_profile_id] = clean_token
        _persist_locked()


def resolve_profile_token(profile: SourceProfile) -> str | None:
    direct_token = str(profile.plex_token or "").strip()
    if direct_token:
        return direct_token
    with _LOCK:
        _ensure_loaded_locked()
        stored_token = _PROFILE_TOKENS.get(profile.id)
    return None if stored_token is None else stored_token.strip() or None


def remember_omdb_api_keys(value: str | None) -> None:
    clean_value = str(value or "").strip()
    with _LOCK:
        global _OMDB_API_KEYS
        _ensure_loaded_locked()
        _OMDB_API_KEYS = clean_value or None
        _persist_locked()


def resolve_omdb_api_keys() -> str:
    with _LOCK:
        _ensure_loaded_locked()
        runtime_keys = _OMDB_API_KEYS
    if runtime_keys:
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
