from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from shared.runtime_profiles import (
    PROJECT_DIR,
    RuntimeConfig,
    SourceProfile,
    ensure_profile_dirs,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class _RunState:
    run_id: str
    profile_id: str
    profile_name: str
    source_type: str
    status: str
    started_at: str
    log_path: str
    pid: int | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "source_type": self.source_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "log_path": self.log_path,
            "pid": self.pid,
        }


_RUN_LOCK = threading.Lock()
_CURRENT_RUN: _RunState | None = None
_LAST_RUN: _RunState | None = None


def _base_env(config: RuntimeConfig, profile: SourceProfile) -> dict[str, str]:
    paths = ensure_profile_dirs(profile.id)
    env = os.environ.copy()
    env["DATA_DIR"] = str(paths.data_dir)
    env["REPORTS_DIR"] = str(paths.reports_dir)
    env["ANALIZA_AUTO_DASHBOARD"] = "0"
    omdb_api_keys = (config.omdb_api_keys or "").strip()
    if omdb_api_keys:
        env["OMDB_API_KEYS"] = omdb_api_keys
        env.pop("OMDB_API_KEY", None)
    else:
        env.pop("OMDB_API_KEYS", None)
    return env


def _plex_cmd_and_env(
    config: RuntimeConfig, profile: SourceProfile
) -> tuple[list[str], dict[str, str]]:
    env = _base_env(config, profile)
    host = (profile.host or "").strip()
    port = int(profile.port or 32400)
    token = (profile.plex_token or "").strip()
    if not host or not token:
        raise ValueError("El perfil Plex necesita host y PLEX_TOKEN.")

    base_url = (profile.base_url or "").strip()
    if not base_url:
        base_url = f"http://{host}"

    env["BASEURL"] = base_url
    env["PLEX_PORT"] = str(port)
    env["PLEX_TOKEN"] = token

    return [
        sys.executable,
        "-m",
        "backend.main",
        "--plex",
        "--no-dashboard",
    ], env


def _dlna_cmd_and_env(
    config: RuntimeConfig, profile: SourceProfile
) -> tuple[list[str], dict[str, str]]:
    env = _base_env(config, profile)
    host = (profile.host or "").strip()
    location = (profile.location or "").strip()
    port = int(profile.port or 0)
    if not host or not location or port <= 0:
        raise ValueError("El perfil DLNA necesita host, port y location.")

    cmd = [
        sys.executable,
        "-m",
        "backend.main",
        "--dlna",
        "--no-dashboard",
        "--dlna-auto-select-all",
        "--dlna-host",
        host,
        "--dlna-port",
        str(port),
        "--dlna-location",
        location,
        "--dlna-friendly-name",
        profile.name,
    ]
    if (profile.device_id or "").strip():
        cmd.extend(["--dlna-device-id", str(profile.device_id).strip()])
    return cmd, env


def _build_cmd_and_env(
    config: RuntimeConfig, profile: SourceProfile
) -> tuple[list[str], dict[str, str]]:
    if profile.source_type == "dlna":
        return _dlna_cmd_and_env(config, profile)
    return _plex_cmd_and_env(config, profile)


def _watch_run(run_id: str) -> None:
    global _CURRENT_RUN, _LAST_RUN

    with _RUN_LOCK:
        state = _CURRENT_RUN if _CURRENT_RUN and _CURRENT_RUN.run_id == run_id else None
    if state is None or state.process is None:
        return

    exit_code = state.process.wait()
    finished_at = _now_iso()
    if state.log_handle is not None:
        try:
            state.log_handle.close()
        except Exception:
            pass

    with _RUN_LOCK:
        current = (
            _CURRENT_RUN if _CURRENT_RUN and _CURRENT_RUN.run_id == run_id else None
        )
        if current is None:
            return
        current.exit_code = exit_code
        current.finished_at = finished_at
        current.status = "succeeded" if exit_code == 0 else "failed"
        current.process = None
        current.log_handle = None
        _LAST_RUN = current
        _CURRENT_RUN = None


def get_run_status() -> dict[str, Any]:
    with _RUN_LOCK:
        state = _CURRENT_RUN or _LAST_RUN
        if state is None:
            return {"run": None}
        if state.status == "running" and state.process is not None:
            poll = state.process.poll()
            if poll is not None:
                state.exit_code = poll
                state.finished_at = state.finished_at or _now_iso()
                state.status = "succeeded" if poll == 0 else "failed"
        return {"run": state.public_dict()}


def start_profile_run(
    *,
    config: RuntimeConfig,
    profile: SourceProfile,
) -> dict[str, Any]:
    global _CURRENT_RUN, _LAST_RUN

    with _RUN_LOCK:
        if _CURRENT_RUN and _CURRENT_RUN.process is not None:
            if _CURRENT_RUN.process.poll() is None:
                raise RuntimeError("Ya hay un análisis ejecutándose.")
            _LAST_RUN = _CURRENT_RUN
            _CURRENT_RUN = None

    cmd, env = _build_cmd_and_env(config, profile)
    paths = ensure_profile_dirs(profile.id)
    log_path = paths.data_dir / "last_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"# started_at={_now_iso()}\n")
    log_handle.write(f"# cmd={' '.join(cmd)}\n")
    log_handle.flush()

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    state = _RunState(
        run_id=uuid.uuid4().hex,
        profile_id=profile.id,
        profile_name=profile.name,
        source_type=profile.source_type,
        status="running",
        started_at=_now_iso(),
        log_path=str(log_path),
        pid=process.pid,
        process=process,
        log_handle=log_handle,
    )

    with _RUN_LOCK:
        _CURRENT_RUN = state

    watcher = threading.Thread(target=_watch_run, args=(state.run_id,), daemon=True)
    watcher.start()
    return {"run": state.public_dict()}
