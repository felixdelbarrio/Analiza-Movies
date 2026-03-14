from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.run_progress import (
    create_initial_progress,
    finalize_progress_file,
    load_progress,
    write_progress,
)
from shared.runtime_profiles import (
    PROJECT_DIR,
    RuntimeConfig,
    SourceProfile,
    ensure_profile_dirs,
)
from server.api.services.runtime_secrets import (
    resolve_omdb_api_keys,
    resolve_profile_token,
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
    progress_path: str
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
    env["PYTHONUNBUFFERED"] = "1"
    omdb_api_keys = resolve_omdb_api_keys(config)
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
    token = str(resolve_profile_token(profile) or "").strip()
    if not host or not token:
        raise ValueError("El perfil Plex necesita host y una sesión Plex autenticada.")

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


def _progress_payload(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return load_progress(Path(path))


def _run_payload(state: _RunState | None) -> dict[str, Any]:
    if state is None:
        return {"run": None}

    payload = state.public_dict()
    payload["progress"] = _progress_payload(state.progress_path)
    return {"run": payload}


def _close_log_handle(state: _RunState) -> None:
    if state.log_handle is None:
        return
    try:
        state.log_handle.close()
    except Exception:
        pass
    state.log_handle = None


def _tail_lines(path: Path, limit: int) -> list[str]:
    max_lines = max(0, min(400, int(limit)))
    if max_lines <= 0 or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=max_lines)]
    except Exception:
        return []


def _watch_run(run_id: str) -> None:
    global _CURRENT_RUN, _LAST_RUN

    with _RUN_LOCK:
        state = _CURRENT_RUN if _CURRENT_RUN and _CURRENT_RUN.run_id == run_id else None
    if state is None or state.process is None:
        return

    exit_code = state.process.wait()
    finished_at = _now_iso()
    _close_log_handle(state)

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
        _LAST_RUN = current
        _CURRENT_RUN = None

    finalize_progress_file(
        Path(state.progress_path),
        status="succeeded" if exit_code == 0 else "failed",
        message=(
            "Ingest completed successfully."
            if exit_code == 0
            else "Ingest finished with an error."
        ),
        message_key=(
            "run.message.plex_done"
            if exit_code == 0 and state.source_type == "plex"
            else (
                "run.message.dlna_done"
                if exit_code == 0 and state.source_type == "dlna"
                else "run.message.failed"
            )
        ),
        exit_code=exit_code,
    )


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
                state.process = None
                _close_log_handle(state)
                finalize_progress_file(
                    Path(state.progress_path),
                    status=state.status,
                    message=(
                        "Ingest completed successfully."
                        if poll == 0
                        else "Ingest finished with an error."
                    ),
                    message_key=(
                        "run.message.plex_done"
                        if poll == 0 and state.source_type == "plex"
                        else (
                            "run.message.dlna_done"
                            if poll == 0 and state.source_type == "dlna"
                            else "run.message.failed"
                        )
                    ),
                    exit_code=poll,
                )
        return _run_payload(state)


def get_run_log_tail(limit: int = 80) -> dict[str, Any]:
    with _RUN_LOCK:
        state = _CURRENT_RUN or _LAST_RUN
    if state is None:
        return {"run": None, "lines": []}

    lines = _tail_lines(Path(state.log_path), limit)
    return {
        "run": _run_payload(state).get("run"),
        "lines": lines,
    }


def stop_current_run() -> dict[str, Any]:
    global _CURRENT_RUN, _LAST_RUN

    with _RUN_LOCK:
        state = _CURRENT_RUN
        if state is None or state.process is None or state.process.poll() is not None:
            raise RuntimeError("No hay ninguna ingesta en ejecución.")
        state.status = "stopping"
        process = state.process
        progress_path = Path(state.progress_path)

    progress_payload = load_progress(progress_path)
    if progress_payload is not None:
        now = _now_iso()
        progress_payload["status"] = "stopping"
        progress_payload["stage"] = "stopping"
        progress_payload["message"] = "Stopping ingest at user request."
        progress_payload["message_key"] = "stage.stopping"
        progress_payload["updated_at"] = now
        recent = progress_payload.setdefault("recent", [])
        if isinstance(recent, list):
            recent.append(
                {
                    "at": now,
                    "level": "info",
                    "stage": "stopping",
                    "scope": progress_payload.get("scope"),
                    "message": "Stopping ingest at user request.",
                    "message_key": "stage.stopping",
                }
            )
            if len(recent) > 24:
                del recent[:-24]
        write_progress(progress_path, progress_payload)

    try:
        process.terminate()
        exit_code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait(timeout=5)

    finished_at = _now_iso()

    with _RUN_LOCK:
        current = _CURRENT_RUN
        if current is not None and current.run_id == state.run_id:
            current.exit_code = exit_code
            current.finished_at = finished_at
            current.status = "cancelled"
            current.process = None
            _close_log_handle(current)
            _LAST_RUN = current
            _CURRENT_RUN = None
            state = current

    finalize_progress_file(
        progress_path,
        status="cancelled",
        message="Ingest cancelled by the user.",
        message_key="run.message.cancelled",
        exit_code=exit_code,
    )
    return _run_payload(state)


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
    progress_path = paths.data_dir / "last_run.progress.json"
    run_id = uuid.uuid4().hex
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress(
        progress_path,
        create_initial_progress(
            run_id=run_id,
            profile_id=profile.id,
            profile_name=profile.name,
            source_type=profile.source_type,
            stage="queued",
            message="Preparing background ingest process.",
            message_key="run.message.background_prepare",
        ),
    )
    env["ANALIZA_PROGRESS_PATH"] = str(progress_path)
    env["ANALIZA_RUN_ID"] = run_id
    env["ANALIZA_PROFILE_ID"] = profile.id
    env["ANALIZA_PROFILE_NAME"] = profile.name
    env["ANALIZA_SOURCE_TYPE"] = profile.source_type
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
        bufsize=1,
    )

    state = _RunState(
        run_id=run_id,
        profile_id=profile.id,
        profile_name=profile.name,
        source_type=profile.source_type,
        status="running",
        started_at=_now_iso(),
        log_path=str(log_path),
        progress_path=str(progress_path),
        pid=process.pid,
        process=process,
        log_handle=log_handle,
    )

    with _RUN_LOCK:
        _CURRENT_RUN = state

    watcher = threading.Thread(target=_watch_run, args=(state.run_id,), daemon=True)
    watcher.start()
    return _run_payload(state)
