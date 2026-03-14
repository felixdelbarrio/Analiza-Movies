from __future__ import annotations

from server.api.services.analysis_runs import _plex_cmd_and_env
from server.api.services.runtime_secrets import remember_profile_token
from shared.runtime_profiles import RuntimeConfig, build_profile_from_discovery


def test_plex_cmd_uses_runtime_token_registry() -> None:
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
