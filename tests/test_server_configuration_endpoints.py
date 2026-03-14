from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.dlna_discovery import DLNADevice
from server.api.app import create_app
import server.api.routers.configuration as configuration_router
from shared import runtime_profiles


def _patch_runtime_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "source_profiles.json"

    def _load():
        return runtime_profiles.load_runtime_config(config_path)

    def _save(config):
        return runtime_profiles.save_runtime_config(config, config_path)

    monkeypatch.setattr(configuration_router, "load_runtime_config", _load)
    monkeypatch.setattr(configuration_router, "save_runtime_config", _save)


def test_config_profile_roundtrip_and_run(monkeypatch, tmp_path):
    _patch_runtime_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        configuration_router,
        "start_profile_run",
        lambda **kwargs: {"run": {"status": "running", "profile_id": "plex-machine-a"}},
    )

    client = TestClient(create_app())

    initial = client.get("/config/state")
    assert initial.status_code == 200
    assert initial.json()["profiles"] == []

    save_res = client.post(
        "/config/profiles",
        json={
            "profile": {
                "source_type": "plex",
                "name": "Plex Sala",
                "host": "192.168.1.20",
                "port": 32400,
                "base_url": "http://192.168.1.20",
                "machine_identifier": "machine-a",
                "plex_token": "secret-token",
            },
            "set_active": True,
        },
    )
    assert save_res.status_code == 200
    payload = save_res.json()
    assert payload["active_profile_id"] == "plex-machine-a"
    assert payload["profiles"][0]["plex_token"] != "secret-token"

    omdb_res = client.put("/config/state", json={"omdb_api_keys": "key-a,key-b"})
    assert omdb_res.status_code == 200
    assert omdb_res.json()["has_omdb_api_keys"] is True

    run_res = client.post("/config/run", json={"profile_id": "plex-machine-a"})
    assert run_res.status_code == 200
    assert run_res.json()["run"]["status"] == "running"


def test_config_discovery_endpoints(monkeypatch, tmp_path):
    _patch_runtime_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        configuration_router,
        "discover_dlna_devices",
        lambda: [
            DLNADevice(
                friendly_name="DLNA Salon",
                location="http://192.168.1.30:8200/root.xml",
                host="192.168.1.30",
                port=8200,
                device_id="uuid:dlna-1",
            )
        ],
    )
    monkeypatch.setattr(
        configuration_router,
        "start_plex_auth_session",
        lambda **kwargs: {
            "session_id": "auth-1",
            "auth_url": "https://app.plex.tv/auth",
            "status": "pending",
        },
    )
    monkeypatch.setattr(
        configuration_router,
        "poll_plex_auth_session",
        lambda session_id: {
            "session_id": session_id,
            "status": "complete",
            "user_token": "user-token",
            "servers": [
                {
                    "source_type": "plex",
                    "name": "Plex Despacho",
                    "host": "192.168.1.40",
                    "port": 32400,
                    "base_url": "http://192.168.1.40",
                    "machine_identifier": "machine-b",
                    "plex_token": "server-token",
                }
            ],
        },
    )
    monkeypatch.setattr(
        configuration_router,
        "get_plex_auth_session",
        lambda session_id: {
            "session_id": session_id,
            "status": "complete",
            "user_token": "user-token",
        },
    )
    monkeypatch.setattr(
        configuration_router,
        "discover_plex_servers",
        lambda token=None: [
            {
                "source_type": "plex",
                "name": "Plex Despacho",
                "host": "192.168.1.40",
                "port": 32400,
                "base_url": "http://192.168.1.40",
                "machine_identifier": "machine-b",
                "plex_token": "server-token" if token else None,
            }
        ],
    )

    client = TestClient(create_app())

    dlna_res = client.post("/config/discover/dlna")
    assert dlna_res.status_code == 200
    assert dlna_res.json()["devices"][0]["name"] == "DLNA Salon"

    auth_res = client.post("/config/plex/auth/start", json={"open_browser": True})
    assert auth_res.status_code == 200
    assert auth_res.json()["session_id"] == "auth-1"

    poll_res = client.get("/config/plex/auth/auth-1")
    assert poll_res.status_code == 200
    assert poll_res.json()["status"] == "complete"

    plex_res = client.post("/config/discover/plex", json={"session_id": "auth-1"})
    assert plex_res.status_code == 200
    assert plex_res.json()["auth_complete"] is True
    assert plex_res.json()["servers"][0]["plex_token"] == "server-token"
