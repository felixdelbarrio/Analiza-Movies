from __future__ import annotations

from pathlib import Path

from shared.runtime_profiles import (
    artifact_paths_for_profile,
    build_profile_from_discovery,
    load_runtime_config,
    save_runtime_config,
    RuntimeConfig,
)


def test_runtime_profiles_roundtrip(tmp_path: Path) -> None:
    profile = build_profile_from_discovery(
        source_type="plex",
        name="Plex Sala",
        host="192.168.1.10",
        port=32400,
        base_url="http://192.168.1.10",
        machine_identifier="machine-a",
        plex_token="secret-token",
    )
    config = (
        RuntimeConfig()
        .upsert_profile(profile, set_active=True)
        .with_omdb_api_keys("abc123")
    )

    config_path = tmp_path / "source_profiles.json"
    save_runtime_config(config, config_path)
    loaded = load_runtime_config(config_path)
    raw_payload = config_path.read_text(encoding="utf-8")

    assert loaded.active_profile_id == profile.id
    assert loaded.omdb_api_keys == ""
    assert len(loaded.profiles) == 1
    assert loaded.profiles[0].machine_identifier == "machine-a"
    assert loaded.profiles[0].plex_token is None
    assert "secret-token" not in raw_payload
    assert "abc123" not in raw_payload


def test_artifact_paths_for_profile_are_namespaced() -> None:
    paths = artifact_paths_for_profile("plex-sala")

    assert paths.profile_id == "plex-sala"
    assert paths.data_dir.parts[-2:] == ("profiles", "plex-sala")
    assert paths.reports_dir.parts[-2:] == ("profiles", "plex-sala")
    assert paths.report_all_path.name == "report_all.csv"
    assert paths.omdb_cache_path.name == "omdb_cache.json"
