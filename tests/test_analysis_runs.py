from __future__ import annotations

import sys

from server.api.services import analysis_runs
from server.api.services.analysis_runs import _plex_cmd_and_env, _refreshed_plex_profile
from server.api.services.runtime_secrets import (
    remember_plex_user_token,
    remember_profile_token,
    reset_runtime_secrets_cache,
)
from shared.runtime_profiles import RuntimeConfig, build_profile_from_discovery


def test_plex_cmd_uses_runtime_token_registry() -> None:
    reset_runtime_secrets_cache()
    profile = build_profile_from_discovery(
        source_type="plex",
        name="Plex Sala",
        host="192.168.1.20",
        port=32400,
        base_url="http://192.168.1.20",
        machine_identifier="machine-analysis-runs",
        plex_token=None,
    )
    remember_profile_token(profile.id, "session-token")

    cmd, env = _plex_cmd_and_env(RuntimeConfig(), profile)

    assert env["PLEX_TOKEN"] == "session-token"
    assert "ANALIZA_AUTO_DASHBOARD" not in env
    assert "--no-dashboard" not in cmd


def test_plex_cmd_falls_back_to_linked_plex_account_token(monkeypatch) -> None:
    reset_runtime_secrets_cache()
    profile = build_profile_from_discovery(
        source_type="plex",
        name="Plex Sala",
        host="192.168.1.20",
        port=32400,
        base_url="http://192.168.1.20",
        machine_identifier="machine-analysis-runs-fallback",
        plex_token=None,
    )
    remember_plex_user_token("account-token")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/tmp/AnalizaMovies", raising=False)

    cmd, env = _plex_cmd_and_env(RuntimeConfig(), profile)

    assert cmd == ["/tmp/AnalizaMovies", "--plex"]
    assert env["PLEX_TOKEN"] == "account-token"


def test_refreshed_plex_profile_updates_stale_endpoint(monkeypatch) -> None:
    reset_runtime_secrets_cache()
    profile = build_profile_from_discovery(
        source_type="plex",
        name="Plex 192.168.1.60",
        host="192.168.1.60",
        port=32400,
        base_url="http://192.168.1.60",
        machine_identifier="machine-analysis-runs-refresh",
        plex_token=None,
    )
    config = RuntimeConfig(active_profile_id=profile.id, profiles=[profile])
    remember_plex_user_token("account-token")

    monkeypatch.setattr(
        analysis_runs,
        "discover_plex_servers",
        lambda token=None: [
            {
                "source_type": "plex",
                "name": "Plex 192.168.1.56",
                "host": "192.168.1.56",
                "port": 32400,
                "base_url": "http://192.168.1.56",
                "machine_identifier": "machine-analysis-runs-refresh",
            }
        ],
    )
    monkeypatch.setattr(analysis_runs, "save_runtime_config", lambda cfg: cfg)

    refreshed_config, refreshed_profile = _refreshed_plex_profile(config, profile)

    assert refreshed_profile.host == "192.168.1.56"
    assert refreshed_profile.base_url == "http://192.168.1.56"
    assert refreshed_profile.name == "Plex 192.168.1.56"
    persisted = refreshed_config.get_profile(profile.id)
    assert persisted is not None
    assert persisted.host == "192.168.1.56"
    assert persisted.base_url == "http://192.168.1.56"
