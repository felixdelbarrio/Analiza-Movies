from __future__ import annotations

from pathlib import Path

import shared.runtime_profiles as runtime_profiles
from shared.runtime_profiles import (
    artifact_paths_for_profile,
    bootstrap_runtime_config,
    build_profile_from_discovery,
    load_runtime_config,
    migrate_legacy_artifacts_to_profile,
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
    config = RuntimeConfig().upsert_profile(profile, set_active=True)

    config_path = tmp_path / "source_profiles.json"
    save_runtime_config(config, config_path)
    loaded = load_runtime_config(config_path)
    raw_payload = config_path.read_text(encoding="utf-8")

    assert loaded.active_profile_id == profile.id
    assert len(loaded.profiles) == 1
    assert loaded.profiles[0].machine_identifier == "machine-a"
    assert loaded.profiles[0].plex_token is None
    assert "secret-token" not in raw_payload
    assert '"omdb_api_keys": ""' in raw_payload


def test_runtime_profiles_public_payload_never_reads_secrets() -> None:
    profile = build_profile_from_discovery(
        source_type="plex",
        name="Plex Sala",
        host="192.168.1.10",
        port=32400,
        machine_identifier="machine-a",
        plex_token="secret-token",
    )
    payload = RuntimeConfig().upsert_profile(profile, set_active=True).to_public_dict()

    assert payload["omdb_api_keys"] == ""
    assert payload["profiles"][0]["plex_token"] is None


def test_artifact_paths_for_profile_are_namespaced() -> None:
    paths = artifact_paths_for_profile("plex-sala")

    assert paths.profile_id == "plex-sala"
    assert paths.data_dir.parts[-2:] == ("profiles", "plex-sala")
    assert paths.reports_dir.parts[-2:] == ("profiles", "plex-sala")
    assert paths.report_all_path.name == "report_all.csv"
    assert paths.omdb_cache_path.name == "omdb_cache.json"


def test_migrate_legacy_artifacts_to_profile_moves_unscoped_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_profiles, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime_profiles, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_DATA_ROOT", tmp_path / "data" / "profiles"
    )
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_REPORTS_ROOT", tmp_path / "reports" / "profiles"
    )

    (runtime_profiles.DATA_DIR / "omdb_cache.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (runtime_profiles.REPORTS_DIR / "report_all.csv").parent.mkdir(
        parents=True, exist_ok=True
    )
    (runtime_profiles.DATA_DIR / "omdb_cache.json").write_text(
        '{"title": "cache"}', encoding="utf-8"
    )
    (runtime_profiles.REPORTS_DIR / "report_all.csv").write_text(
        "title\nMovie\n", encoding="utf-8"
    )

    migrated = migrate_legacy_artifacts_to_profile("plex-sala")

    assert migrated.omdb_cache_path.exists()
    assert migrated.report_all_path.exists()
    assert not (runtime_profiles.DATA_DIR / "omdb_cache.json").exists()
    assert not (runtime_profiles.REPORTS_DIR / "report_all.csv").exists()


def test_migrate_legacy_artifacts_to_profile_skips_when_namespaced_data_exists(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_profiles, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime_profiles, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_DATA_ROOT", tmp_path / "data" / "profiles"
    )
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_REPORTS_ROOT", tmp_path / "reports" / "profiles"
    )

    legacy_report = runtime_profiles.REPORTS_DIR / "report_all.csv"
    legacy_report.parent.mkdir(parents=True, exist_ok=True)
    legacy_report.write_text("title\nLegacy\n", encoding="utf-8")

    existing_scoped = (
        runtime_profiles.PROFILE_REPORTS_ROOT / "plex-otro" / "report_all.csv"
    )
    existing_scoped.parent.mkdir(parents=True, exist_ok=True)
    existing_scoped.write_text("title\nScoped\n", encoding="utf-8")

    target = migrate_legacy_artifacts_to_profile("plex-sala")

    assert target.report_all_path == (
        runtime_profiles.PROFILE_REPORTS_ROOT / "plex-sala" / "report_all.csv"
    )
    assert legacy_report.exists()
    assert not target.report_all_path.exists()


def test_bootstrap_runtime_config_activates_single_profile_and_migrates_legacy_data(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_profiles, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime_profiles, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_DATA_ROOT", tmp_path / "data" / "profiles"
    )
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_REPORTS_ROOT", tmp_path / "reports" / "profiles"
    )

    profile = build_profile_from_discovery(
        source_type="plex",
        name="Plex Sala",
        host="192.168.1.10",
        port=32400,
        machine_identifier="machine-a",
    )
    config_path = tmp_path / "data" / "source_profiles.json"
    save_runtime_config(
        RuntimeConfig(active_profile_id=None, profiles=[profile]), config_path
    )

    legacy_report = runtime_profiles.REPORTS_DIR / "report_all.csv"
    legacy_report.parent.mkdir(parents=True, exist_ok=True)
    legacy_report.write_text("title\nMovie\n", encoding="utf-8")

    bootstrapped = bootstrap_runtime_config(config_path)

    assert bootstrapped.active_profile_id == profile.id
    assert not legacy_report.exists()
    assert (
        runtime_profiles.PROFILE_REPORTS_ROOT / profile.id / "report_all.csv"
    ).exists()


def test_bootstrap_runtime_config_does_not_guess_owner_with_multiple_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_profiles, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime_profiles, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_DATA_ROOT", tmp_path / "data" / "profiles"
    )
    monkeypatch.setattr(
        runtime_profiles, "PROFILE_REPORTS_ROOT", tmp_path / "reports" / "profiles"
    )

    profiles = [
        build_profile_from_discovery(
            source_type="plex",
            name="Plex A",
            host="192.168.1.10",
            port=32400,
            machine_identifier="machine-a",
        ),
        build_profile_from_discovery(
            source_type="plex",
            name="Plex B",
            host="192.168.1.11",
            port=32400,
            machine_identifier="machine-b",
        ),
    ]
    config_path = tmp_path / "data" / "source_profiles.json"
    save_runtime_config(
        RuntimeConfig(active_profile_id=None, profiles=profiles), config_path
    )

    legacy_report = runtime_profiles.REPORTS_DIR / "report_all.csv"
    legacy_report.parent.mkdir(parents=True, exist_ok=True)
    legacy_report.write_text("title\nLegacy\n", encoding="utf-8")

    bootstrapped = bootstrap_runtime_config(config_path)

    assert bootstrapped.active_profile_id is None
    assert legacy_report.exists()
    assert not any(runtime_profiles.PROFILE_REPORTS_ROOT.rglob("report_all.csv"))
