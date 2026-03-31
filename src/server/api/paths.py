from __future__ import annotations

import os
from pathlib import Path

from shared.runtime_profiles import (
    ArtifactPaths,
    PROJECT_DIR,
    artifact_paths_for_active_profile,
    artifact_paths_for_profile,
)

BASE_DIR = PROJECT_DIR


def resolve_path(env_name: str, candidates: list[Path]) -> Path:
    raw = (os.getenv(env_name) or "").strip().strip('"').strip("'")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p

    # Avoid probing the filesystem during app startup. The path is resolved
    # lazily and existence is checked only when an operation actually needs it.
    return candidates[0]


DEFAULT_OMDB_CACHE_PATH = resolve_path(
    "OMDB_CACHE_PATH",
    [
        BASE_DIR / "data" / "omdb_cache.json",
    ],
)

DEFAULT_WIKI_CACHE_PATH = resolve_path(
    "WIKI_CACHE_PATH",
    [
        BASE_DIR / "data" / "wiki_cache.json",
    ],
)

DEFAULT_REPORT_ALL_PATH = resolve_path(
    "REPORT_ALL_PATH",
    [
        BASE_DIR / "reports" / "report_all.csv",
    ],
)

DEFAULT_REPORT_FILTERED_PATH = resolve_path(
    "REPORT_FILTERED_PATH",
    [
        BASE_DIR / "reports" / "report_filtered.csv",
    ],
)

DEFAULT_METADATA_FIX_PATH = resolve_path(
    "METADATA_FIX_PATH",
    [
        BASE_DIR / "reports" / "metadata_fix.csv",
    ],
)


def get_artifact_paths(profile_id: str | None = None) -> ArtifactPaths:
    raw_profile_id = (profile_id or "").strip()
    runtime_paths = (
        artifact_paths_for_profile(raw_profile_id)
        if raw_profile_id
        else artifact_paths_for_active_profile()
    )

    if runtime_paths.profile_id is None:
        return ArtifactPaths(
            profile_id=None,
            data_dir=runtime_paths.data_dir,
            reports_dir=runtime_paths.reports_dir,
            omdb_cache_path=DEFAULT_OMDB_CACHE_PATH,
            wiki_cache_path=DEFAULT_WIKI_CACHE_PATH,
            report_all_path=DEFAULT_REPORT_ALL_PATH,
            report_filtered_path=DEFAULT_REPORT_FILTERED_PATH,
            metadata_fix_path=DEFAULT_METADATA_FIX_PATH,
        )

    return runtime_paths


def get_omdb_cache_path(profile_id: str | None = None) -> Path:
    return get_artifact_paths(profile_id).omdb_cache_path


def get_wiki_cache_path(profile_id: str | None = None) -> Path:
    return get_artifact_paths(profile_id).wiki_cache_path


def get_report_all_path(profile_id: str | None = None) -> Path:
    return get_artifact_paths(profile_id).report_all_path


def get_report_filtered_path(profile_id: str | None = None) -> Path:
    return get_artifact_paths(profile_id).report_filtered_path


def get_metadata_fix_path(profile_id: str | None = None) -> Path:
    return get_artifact_paths(profile_id).metadata_fix_path
