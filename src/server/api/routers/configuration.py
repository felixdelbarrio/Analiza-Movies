from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from backend.dlna_discovery import discover_dlna_devices
from server.api.services.analysis_runs import (
    get_run_log_tail,
    get_run_status,
    start_profile_run,
    stop_current_run,
)
from server.api.services.plex_sources import (
    discover_plex_servers,
    get_plex_auth_session,
    poll_plex_auth_session,
    sanitize_plex_servers,
    start_plex_auth_session,
)
from server.api.services.runtime_secrets import (
    has_omdb_api_keys,
    remember_omdb_api_keys,
    remember_profile_token,
    resolve_profile_token,
)
from shared.runtime_profiles import (
    RuntimeConfig,
    SourceProfile,
    build_profile_from_discovery,
    load_runtime_config,
    save_runtime_config,
)

router = APIRouter(prefix="/config", tags=["config"])
logger = logging.getLogger(__name__)


def _state_payload(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or load_runtime_config()
    payload = cfg.to_public_dict()
    payload["has_omdb_api_keys"] = has_omdb_api_keys()
    return payload


def _merge_profile(
    existing: SourceProfile | None, incoming: SourceProfile
) -> SourceProfile:
    if existing is None:
        return incoming

    updates: dict[str, Any] = {}
    incoming_data = incoming.to_internal_dict(mask_secrets=False)
    for key, value in incoming_data.items():
        if key in {"id", "created_at", "updated_at"}:
            continue
        if value not in (None, ""):
            updates[key] = value
    return existing.with_updates(**updates)


@router.get("/state")
def config_state() -> dict[str, Any]:
    return _state_payload()


@router.put("/state")
def update_config_state(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    config = load_runtime_config()
    if "omdb_api_keys" in payload:
        omdb_api_keys = str(payload.get("omdb_api_keys") or "")
        remember_omdb_api_keys(omdb_api_keys)
    if "active_profile_id" in payload:
        config = config.with_active_profile(payload.get("active_profile_id"))
    config = save_runtime_config(config)
    return _state_payload(config)


@router.post("/profiles")
def save_profile(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    profile_payload = payload.get("profile")
    data = profile_payload if isinstance(profile_payload, dict) else payload
    incoming_plex_token = str(data.get("plex_token") or "").strip() or None

    source_type = str(data.get("source_type") or "plex").strip().lower()
    if source_type not in {"plex", "dlna"}:
        raise HTTPException(status_code=400, detail="source_type inválido")

    profile = build_profile_from_discovery(
        source_type="dlna" if source_type == "dlna" else "plex",
        name=str(data.get("name") or "Origen").strip() or "Origen",
        host=data.get("host"),
        port=data.get("port"),
        base_url=data.get("base_url"),
        location=data.get("location"),
        device_id=data.get("device_id"),
        machine_identifier=data.get("machine_identifier"),
        plex_token=incoming_plex_token,
        profile_id=data.get("id"),
    )
    if source_type == "plex":
        resolved_token = incoming_plex_token or resolve_profile_token(profile)
        if not resolved_token:
            raise HTTPException(
                status_code=400,
                detail="Vincula Plex y autoriza el servidor antes de guardarlo.",
            )
        remember_profile_token(profile.id, resolved_token)

    config = load_runtime_config()
    existing = config.get_profile(profile.id)
    merged = _merge_profile(existing, profile)
    config = config.upsert_profile(
        merged, set_active=bool(payload.get("set_active", True))
    )
    config = save_runtime_config(config)
    return _state_payload(config)


@router.post("/profiles/active")
def set_active_profile(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    profile_id = payload.get("profile_id")
    config = load_runtime_config().with_active_profile(profile_id)
    config = save_runtime_config(config)
    return _state_payload(config)


@router.post("/discover/dlna")
def discover_dlna() -> dict[str, Any]:
    devices = discover_dlna_devices()
    return {
        "devices": [
            {
                "name": device.friendly_name,
                "source_type": "dlna",
                "host": device.host,
                "port": device.port,
                "location": device.location,
                "device_id": device.device_id,
            }
            for device in devices
        ]
    }


@router.post("/discover/plex")
def discover_plex(
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    body = payload or {}
    session_id = str(body.get("session_id") or "").strip()
    user_token: str | None = None
    if session_id:
        session = get_plex_auth_session(session_id)
        if session and session.get("status") == "complete":
            token = session.get("user_token")
            if isinstance(token, str) and token.strip():
                user_token = token.strip()

    servers = sanitize_plex_servers(discover_plex_servers(user_token))
    return {
        "servers": servers,
        "session_id": session_id or None,
        "auth_complete": bool(user_token),
    }


@router.post("/plex/auth/start")
def plex_auth_start(
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    body = payload or {}
    try:
        return start_plex_auth_session(
            open_browser=bool(body.get("open_browser", True))
        )
    except Exception:
        logger.exception("Error iniciando auth Plex")
        raise HTTPException(
            status_code=502, detail="No se pudo iniciar la autenticación Plex."
        )


@router.get("/plex/auth/{session_id}")
def plex_auth_status(session_id: str) -> dict[str, Any]:
    try:
        payload = poll_plex_auth_session(session_id)
        servers = payload.get("servers")
        if isinstance(servers, list):
            payload["servers"] = sanitize_plex_servers(
                [item for item in servers if isinstance(item, dict)]
            )
        return payload
    except KeyError:
        raise HTTPException(status_code=404, detail="Sesión Plex no encontrada")
    except Exception:
        logger.exception("Error consultando auth Plex")
        raise HTTPException(
            status_code=502, detail="No se pudo consultar la autenticación Plex."
        )


@router.get("/run")
def run_status() -> dict[str, Any]:
    return get_run_status()


@router.get("/run/logs")
def run_logs(limit: int = Query(80, ge=0, le=400)) -> dict[str, Any]:
    return get_run_log_tail(limit=limit)


@router.post("/run")
def run_profile(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    config = load_runtime_config()
    profile_id = str(
        payload.get("profile_id") or config.active_profile_id or ""
    ).strip()
    if not profile_id:
        raise HTTPException(status_code=400, detail="No hay perfil activo")

    profile = config.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")

    try:
        return start_profile_run(config=config, profile=profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("No se pudo iniciar el análisis del perfil %s", profile.id)
        raise HTTPException(status_code=500, detail="No se pudo iniciar el análisis.")


@router.delete("/run")
def stop_run() -> dict[str, Any]:
    try:
        return stop_current_run()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
