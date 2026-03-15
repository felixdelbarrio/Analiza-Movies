from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ENV_PROGRESS_PATH = "ANALIZA_PROGRESS_PATH"
_ENV_RUN_ID = "ANALIZA_RUN_ID"
_ENV_PROFILE_ID = "ANALIZA_PROFILE_ID"
_ENV_PROFILE_NAME = "ANALIZA_PROFILE_NAME"
_ENV_SOURCE_TYPE = "ANALIZA_SOURCE_TYPE"

_RECENT_EVENTS_LIMIT = 24
_WRITER_LOCK = threading.Lock()
_WRITER: "RunProgressWriter | None" = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _progress_percent(current: int | None, total: int | None) -> float | None:
    if current is None or total is None or total <= 0:
        return None
    return round(max(0.0, min(100.0, (current / total) * 100.0)), 1)


def _normalize_params(
    params: Mapping[str, object] | None,
) -> dict[str, str | int | float | bool] | None:
    if not params:
        return None

    out: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out or None


def create_initial_progress(
    *,
    run_id: str,
    profile_id: str,
    profile_name: str,
    source_type: str,
    status: str = "running",
    stage: str = "queued",
    message: str = "Preparing background ingest process.",
    message_key: str | None = None,
    message_params: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    normalized_params = _normalize_params(message_params)
    return {
        "run_id": run_id,
        "profile_id": profile_id,
        "profile_name": profile_name,
        "source_type": source_type,
        "status": status,
        "stage": stage,
        "message": message,
        "message_key": message_key,
        "message_params": normalized_params,
        "scope": None,
        "current": None,
        "total": None,
        "unit": None,
        "percent": None,
        "errors": 0,
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "exit_code": None,
        "counters": {
            "processed": 0,
            "rows_written": 0,
            "suggestions_written": 0,
            "filtered_rows": 0,
            "libraries_total": 0,
            "libraries_completed": 0,
            "containers_total": 0,
            "containers_completed": 0,
        },
        "recent": [
            {
                "at": now,
                "level": "info",
                "stage": stage,
                "scope": None,
                "message": message,
                "message_key": message_key,
                "message_params": normalized_params,
            }
        ],
    }


def load_progress(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_progress(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(serialized + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return dict(payload)


class RunProgressWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._state = load_progress(path) or {}

    def ensure_initialized(
        self,
        *,
        run_id: str,
        profile_id: str,
        profile_name: str,
        source_type: str,
    ) -> None:
        with self._lock:
            if self._state:
                self._state.setdefault("run_id", run_id)
                self._state.setdefault("profile_id", profile_id)
                self._state.setdefault("profile_name", profile_name)
                self._state.setdefault("source_type", source_type)
                self._state.setdefault("recent", [])
                self._state.setdefault("counters", {})
            else:
                self._state = create_initial_progress(
                    run_id=run_id,
                    profile_id=profile_id,
                    profile_name=profile_name,
                    source_type=source_type,
                )
            write_progress(self._path, self._state)

    def update(
        self,
        *,
        status: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        message_key: str | None = None,
        message_params: Mapping[str, object] | None = None,
        scope: str | None = None,
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        errors: int | None = None,
        counters: Mapping[str, int] | None = None,
        record_event: bool = False,
        level: str = "info",
        exit_code: int | None = None,
        finished: bool = False,
    ) -> None:
        with self._lock:
            if not self._state:
                return

            if status is not None:
                self._state["status"] = status
            if stage is not None:
                self._state["stage"] = stage
            if message is not None:
                self._state["message"] = message
            if message_key is not None:
                self._state["message_key"] = message_key
            if message_params is not None:
                self._state["message_params"] = _normalize_params(message_params)
            if scope is not None:
                self._state["scope"] = scope
            if current is not None:
                self._state["current"] = max(0, int(current))
            if total is not None:
                self._state["total"] = max(0, int(total))
            if unit is not None:
                self._state["unit"] = unit
            if errors is not None:
                self._state["errors"] = max(0, int(errors))
            if exit_code is not None:
                self._state["exit_code"] = int(exit_code)

            current_value = _safe_int(self._state.get("current"))
            total_value = _safe_int(self._state.get("total"))
            self._state["percent"] = _progress_percent(current_value, total_value)
            self._state["updated_at"] = _now_iso()

            counters_payload = self._state.setdefault("counters", {})
            if isinstance(counters_payload, dict) and counters:
                for key, value in counters.items():
                    counters_payload[key] = max(0, int(value))

            event_message = (
                message if message is not None else self._state.get("message")
            )
            event_key = (
                message_key
                if message_key is not None
                else self._state.get("message_key")
            )
            event_params = (
                _normalize_params(message_params)
                if message_params is not None
                else self._state.get("message_params")
            )

            if record_event and (event_message or event_key):
                recent = self._state.setdefault("recent", [])
                if isinstance(recent, list):
                    recent.append(
                        {
                            "at": self._state["updated_at"],
                            "level": level,
                            "stage": stage or self._state.get("stage"),
                            "scope": (
                                scope if scope is not None else self._state.get("scope")
                            ),
                            "message": event_message,
                            "message_key": event_key,
                            "message_params": event_params,
                        }
                    )
                    if len(recent) > _RECENT_EVENTS_LIMIT:
                        del recent[:-_RECENT_EVENTS_LIMIT]

            if finished:
                self._state["finished_at"] = self._state["updated_at"]

            write_progress(self._path, self._state)

    def finish(
        self,
        *,
        status: str,
        message: str | None = None,
        message_key: str | None = None,
        message_params: Mapping[str, object] | None = None,
        exit_code: int | None = None,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        self.update(
            status=status,
            stage="finished",
            message=message,
            message_key=message_key,
            message_params=message_params,
            counters=counters,
            exit_code=exit_code,
            record_event=bool(message or message_key),
            level="error" if status == "failed" else "info",
            finished=True,
        )


def bind_progress_from_env() -> RunProgressWriter | None:
    global _WRITER

    with _WRITER_LOCK:
        if _WRITER is not None:
            return _WRITER

        raw_path = _clean_str(os.getenv(_ENV_PROGRESS_PATH))
        if raw_path is None:
            return None

        writer = RunProgressWriter(Path(raw_path))
        writer.ensure_initialized(
            run_id=_clean_str(os.getenv(_ENV_RUN_ID)) or "run",
            profile_id=_clean_str(os.getenv(_ENV_PROFILE_ID)) or "default",
            profile_name=_clean_str(os.getenv(_ENV_PROFILE_NAME)) or "Origen",
            source_type=_clean_str(os.getenv(_ENV_SOURCE_TYPE)) or "unknown",
        )
        _WRITER = writer
        return writer


def update_run_progress(**kwargs: Any) -> None:
    writer = bind_progress_from_env()
    if writer is None:
        return
    writer.update(**kwargs)


def finish_run_progress(
    status: str,
    *,
    message: str | None = None,
    message_key: str | None = None,
    message_params: Mapping[str, object] | None = None,
    exit_code: int | None = None,
    counters: Mapping[str, int] | None = None,
) -> None:
    writer = bind_progress_from_env()
    if writer is None:
        return
    writer.finish(
        status=status,
        message=message,
        message_key=message_key,
        message_params=message_params,
        exit_code=exit_code,
        counters=counters,
    )


def finalize_progress_file(
    path: Path,
    *,
    status: str,
    message: str | None = None,
    message_key: str | None = None,
    message_params: Mapping[str, object] | None = None,
    exit_code: int | None = None,
) -> dict[str, Any] | None:
    payload = load_progress(path)
    if payload is None:
        return None

    now = _now_iso()
    payload["status"] = status
    payload["stage"] = "finished"
    if message:
        payload["message"] = message
    if message_key is not None:
        payload["message_key"] = message_key
    if message_params is not None:
        payload["message_params"] = _normalize_params(message_params)
    if exit_code is not None:
        payload["exit_code"] = int(exit_code)
    payload["updated_at"] = now
    payload["finished_at"] = now
    payload["percent"] = 100.0 if status == "succeeded" else payload.get("percent")

    recent = payload.setdefault("recent", [])
    if (message or message_key) and isinstance(recent, list):
        recent.append(
            {
                "at": now,
                "level": "error" if status == "failed" else "info",
                "stage": "finished",
                "scope": payload.get("scope"),
                "message": message,
                "message_key": message_key,
                "message_params": _normalize_params(message_params),
            }
        )
        if len(recent) > _RECENT_EVENTS_LIMIT:
            del recent[:-_RECENT_EVENTS_LIMIT]

    return write_progress(path, payload)
