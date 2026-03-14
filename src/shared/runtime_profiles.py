from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SourceType = Literal["plex", "dlna"]

_CONFIG_VERSION = 1
_PROFILE_SLUG_RE = re.compile(r"[^a-z0-9]+")

SHARED_DIR = Path(__file__).resolve().parent


def _project_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    for candidate in SHARED_DIR.parents:
        if (candidate / "setup.py").exists() and (candidate / "web").exists():
            return candidate
    return SHARED_DIR.parents[2]


PROJECT_DIR = _project_root()
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
PROFILES_CONFIG_PATH = DATA_DIR / "source_profiles.json"
PROFILE_DATA_ROOT = DATA_DIR / "profiles"
PROFILE_REPORTS_ROOT = REPORTS_DIR / "profiles"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_source_type(raw: object) -> SourceType:
    text = str(raw or "").strip().lower()
    if text == "dlna":
        return "dlna"
    return "plex"


def _clean_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_port(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        port = int(value)
    elif isinstance(value, int):
        port = value
    elif isinstance(value, float):
        port = int(value)
    elif isinstance(value, str):
        try:
            port = int(value.strip())
        except ValueError:
            return None
    else:
        return None
    if 1 <= port <= 65535:
        return port
    return None


def _slugify(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = _PROFILE_SLUG_RE.sub("-", lowered)
    lowered = lowered.strip("-")
    return lowered or "perfil"


def _stable_profile_id(
    *,
    source_type: SourceType,
    machine_identifier: str | None,
    device_id: str | None,
    host: str | None,
    port: int | None,
    name: str,
) -> str:
    preferred = (
        _clean_str(machine_identifier)
        or _clean_str(device_id)
        or _clean_str(host)
        or _clean_str(name)
    )
    suffix = _slugify(preferred or name)
    if host and port and preferred == _clean_str(host):
        suffix = _slugify(f"{host}-{port}")
    return f"{source_type}-{suffix}"


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    profile_id: str | None
    data_dir: Path
    reports_dir: Path
    omdb_cache_path: Path
    wiki_cache_path: Path
    report_all_path: Path
    report_filtered_path: Path
    metadata_fix_path: Path


@dataclass(frozen=True, slots=True)
class SourceProfile:
    id: str
    name: str
    source_type: SourceType
    host: str | None = None
    port: int | None = None
    base_url: str | None = None
    location: str | None = None
    device_id: str | None = None
    machine_identifier: str | None = None
    plex_token: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SourceProfile":
        name = _clean_str(payload.get("name")) or "Origen"
        source_type = _normalize_source_type(payload.get("source_type"))
        machine_identifier = _clean_str(payload.get("machine_identifier"))
        device_id = _clean_str(payload.get("device_id"))
        host = _clean_str(payload.get("host"))
        port = _clean_port(payload.get("port"))
        explicit_id = _clean_str(payload.get("id"))
        profile_id = explicit_id or _stable_profile_id(
            source_type=source_type,
            machine_identifier=machine_identifier,
            device_id=device_id,
            host=host,
            port=port,
            name=name,
        )
        return SourceProfile(
            id=profile_id,
            name=name,
            source_type=source_type,
            host=host,
            port=port,
            base_url=_clean_str(payload.get("base_url")),
            location=_clean_str(payload.get("location")),
            device_id=device_id,
            machine_identifier=machine_identifier,
            plex_token=_clean_str(payload.get("plex_token")),
            created_at=_clean_str(payload.get("created_at")) or _now_iso(),
            updated_at=_clean_str(payload.get("updated_at")) or _now_iso(),
        )

    def to_internal_dict(self, *, mask_secrets: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if mask_secrets:
            data["plex_token"] = mask_secret(self.plex_token)
        return data

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "location": self.location,
            "device_id": self.device_id,
            "machine_identifier": self.machine_identifier,
            "plex_token": None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def with_updates(self, **updates: Any) -> "SourceProfile":
        merged = self.to_internal_dict(mask_secrets=False)
        merged.update(updates)
        merged["updated_at"] = _now_iso()
        return SourceProfile.from_dict(merged)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    version: int = _CONFIG_VERSION
    active_profile_id: str | None = None
    profiles: list[SourceProfile] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    @staticmethod
    def from_dict(payload: dict[str, Any] | None) -> "RuntimeConfig":
        if not isinstance(payload, dict):
            return RuntimeConfig()

        profiles_raw = payload.get("profiles")
        profiles: list[SourceProfile] = []
        if isinstance(profiles_raw, list):
            for item in profiles_raw:
                if isinstance(item, dict):
                    try:
                        profiles.append(SourceProfile.from_dict(item))
                    except Exception:
                        continue

        active_profile_id = _clean_str(payload.get("active_profile_id"))
        if active_profile_id and active_profile_id not in {p.id for p in profiles}:
            active_profile_id = None

        return RuntimeConfig(
            version=int(payload.get("version") or _CONFIG_VERSION),
            active_profile_id=active_profile_id,
            profiles=profiles,
            updated_at=_clean_str(payload.get("updated_at")) or _now_iso(),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "omdb_api_keys": "",
            "active_profile_id": self.active_profile_id,
            "profiles": [p.to_public_dict() for p in self.profiles],
            "updated_at": self.updated_at,
        }

    def get_profile(self, profile_id: str | None) -> SourceProfile | None:
        wanted = _clean_str(profile_id)
        if wanted is None:
            return None
        for profile in self.profiles:
            if profile.id == wanted:
                return profile
        return None

    def with_active_profile(self, profile_id: str | None) -> "RuntimeConfig":
        wanted = _clean_str(profile_id)
        if wanted is not None and wanted not in {p.id for p in self.profiles}:
            wanted = None
        return RuntimeConfig(
            version=self.version,
            active_profile_id=wanted,
            profiles=list(self.profiles),
            updated_at=_now_iso(),
        )

    def upsert_profile(
        self, profile: SourceProfile, *, set_active: bool = False
    ) -> "RuntimeConfig":
        out: list[SourceProfile] = []
        replaced = False
        for current in self.profiles:
            if current.id == profile.id:
                out.append(profile.with_updates(created_at=current.created_at))
                replaced = True
            else:
                out.append(current)
        if not replaced:
            out.append(profile)

        active_profile_id = self.active_profile_id
        if set_active or active_profile_id is None:
            active_profile_id = profile.id

        return RuntimeConfig(
            version=self.version,
            active_profile_id=active_profile_id,
            profiles=out,
            updated_at=_now_iso(),
        )


def mask_secret(value: str | None) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * (len(text) - 8)}{text[-4:]}"


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    resolved_path = PROFILES_CONFIG_PATH if path is None else path
    if not resolved_path.exists():
        return RuntimeConfig()
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception:
        return RuntimeConfig()
    return RuntimeConfig.from_dict(payload)


def save_runtime_config(
    config: RuntimeConfig, path: Path | None = None
) -> RuntimeConfig:
    resolved_path = PROFILES_CONFIG_PATH if path is None else path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    # Runtime secrets stay outside the JSON config by design.
    payload = config.to_public_dict()
    resolved_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RuntimeConfig.from_dict(payload)


def active_profile(
    config: RuntimeConfig | None = None, *, profile_id: str | None = None
) -> SourceProfile | None:
    cfg = config or load_runtime_config()
    if profile_id is not None:
        return cfg.get_profile(profile_id)
    return cfg.get_profile(cfg.active_profile_id)


def artifact_paths_for_profile(
    profile_id: str | None = None,
) -> ArtifactPaths:
    clean_profile_id = _clean_str(profile_id)
    if clean_profile_id is None:
        return ArtifactPaths(
            profile_id=None,
            data_dir=DATA_DIR,
            reports_dir=REPORTS_DIR,
            omdb_cache_path=DATA_DIR / "omdb_cache.json",
            wiki_cache_path=DATA_DIR / "wiki_cache.json",
            report_all_path=REPORTS_DIR / "report_all.csv",
            report_filtered_path=REPORTS_DIR / "report_filtered.csv",
            metadata_fix_path=REPORTS_DIR / "metadata_fix.csv",
        )

    data_dir = PROFILE_DATA_ROOT / clean_profile_id
    reports_dir = PROFILE_REPORTS_ROOT / clean_profile_id
    return ArtifactPaths(
        profile_id=clean_profile_id,
        data_dir=data_dir,
        reports_dir=reports_dir,
        omdb_cache_path=data_dir / "omdb_cache.json",
        wiki_cache_path=data_dir / "wiki_cache.json",
        report_all_path=reports_dir / "report_all.csv",
        report_filtered_path=reports_dir / "report_filtered.csv",
        metadata_fix_path=reports_dir / "metadata_fix.csv",
    )


def artifact_paths_for_active_profile(
    config: RuntimeConfig | None = None,
) -> ArtifactPaths:
    cfg = config or load_runtime_config()
    return artifact_paths_for_profile(cfg.active_profile_id)


def ensure_profile_dirs(profile_id: str | None) -> ArtifactPaths:
    paths = artifact_paths_for_profile(profile_id)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _artifact_file_pairs(target: ArtifactPaths) -> list[tuple[Path, Path]]:
    global_paths = artifact_paths_for_profile(None)
    return [
        (global_paths.omdb_cache_path, target.omdb_cache_path),
        (global_paths.wiki_cache_path, target.wiki_cache_path),
        (global_paths.report_all_path, target.report_all_path),
        (global_paths.report_filtered_path, target.report_filtered_path),
        (global_paths.metadata_fix_path, target.metadata_fix_path),
    ]


def _has_legacy_global_artifacts() -> bool:
    global_paths = artifact_paths_for_profile(None)
    legacy_files = (
        global_paths.omdb_cache_path,
        global_paths.wiki_cache_path,
        global_paths.report_all_path,
        global_paths.report_filtered_path,
        global_paths.metadata_fix_path,
    )
    return any(path.exists() for path in legacy_files)


def _has_namespaced_artifacts() -> bool:
    scoped_roots = (
        (PROFILE_DATA_ROOT, ("omdb_cache.json", "wiki_cache.json")),
        (
            PROFILE_REPORTS_ROOT,
            ("report_all.csv", "report_filtered.csv", "metadata_fix.csv"),
        ),
    )
    for root, names in scoped_roots:
        if not root.exists():
            continue
        for name in names:
            if any(root.rglob(name)):
                return True
    return False


def migrate_legacy_artifacts_to_profile(profile_id: str | None) -> ArtifactPaths:
    target = artifact_paths_for_profile(profile_id)
    if target.profile_id is None or _has_namespaced_artifacts():
        return target

    pairs = _artifact_file_pairs(target)
    if not any(source.exists() for source, _ in pairs):
        return target

    target.data_dir.mkdir(parents=True, exist_ok=True)
    target.reports_dir.mkdir(parents=True, exist_ok=True)

    for source, destination in pairs:
        if not source.exists() or destination.exists():
            continue
        shutil.move(str(source), str(destination))

    return target


def bootstrap_runtime_config(path: Path | None = None) -> RuntimeConfig:
    resolved_path = PROFILES_CONFIG_PATH if path is None else path
    config = load_runtime_config(resolved_path)
    updated = config

    if updated.active_profile_id is None and len(updated.profiles) == 1:
        updated = updated.with_active_profile(updated.profiles[0].id)

    if updated.active_profile_id and _has_legacy_global_artifacts():
        migrate_legacy_artifacts_to_profile(updated.active_profile_id)

    if updated != config:
        return save_runtime_config(updated, resolved_path)
    return updated


def build_profile_from_discovery(
    *,
    source_type: SourceType,
    name: str,
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
    location: str | None = None,
    device_id: str | None = None,
    machine_identifier: str | None = None,
    plex_token: str | None = None,
    profile_id: str | None = None,
) -> SourceProfile:
    payload = {
        "id": profile_id,
        "name": name,
        "source_type": source_type,
        "host": host,
        "port": port,
        "base_url": base_url,
        "location": location,
        "device_id": device_id,
        "machine_identifier": machine_identifier,
        "plex_token": plex_token,
    }
    return SourceProfile.from_dict(payload)
