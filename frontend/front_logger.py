"""
frontend/front_logger.py

Logger ultra simple para frontend:
- No depende del backend
- En Streamlit, lo más fiable suele ser no spamear stdout.
- Si FRONT_DEBUG=True, permite warnings en consola.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from frontend.config_front_base import (
    FRONT_DEBUG,
    LOGGER_FILE_ENABLED,
    LOGGER_FILE_PATH,
)


_FILE_LOCK = threading.Lock()


def _should_print(always: bool) -> bool:
    return bool(always or FRONT_DEBUG)


def _log(level: str, msg: str, *, always: bool = False) -> None:
    if _should_print(always):
        print(str(msg))
    _append_to_file(level, str(msg))


def _append_to_file(level: str, msg: str) -> None:
    if not LOGGER_FILE_ENABLED:
        return
    if LOGGER_FILE_PATH is None:
        return
    try:
        path = Path(LOGGER_FILE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level}] {msg}\n"
        with _FILE_LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:
        pass


def log_warning(msg: str, *, always: bool = False) -> None:
    _log("WARN", msg, always=always)


def log_info(msg: str, *, always: bool = False) -> None:
    _log("INFO", msg, always=always)


def warning(msg: str, *, always: bool = False) -> None:
    _log("WARN", msg, always=always)


def info(msg: str, *, always: bool = False) -> None:
    _log("INFO", msg, always=always)


def error(msg: str, *, always: bool = False) -> None:
    _log("ERROR", msg, always=always)


def progress(msg: str, *, always: bool = False) -> None:
    _log("PROGRESS", msg, always=always)
