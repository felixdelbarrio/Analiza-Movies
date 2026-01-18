# logger y utilidades de logging
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from server.api.settings import Settings, _env_bool, _env_str

_FILE_HANDLER_TAG = "_analiza_api_file_handler"
_LOGGER_FILE_PATH_SENTINEL: object = object()
_LOGGER_FILE_PATH_CACHED: Path | None | object = _LOGGER_FILE_PATH_SENTINEL

SERVER_DIR = Path(__file__).resolve().parents[1]


def _sanitize_filename_component(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._-")


def _resolve_dir(raw: str, *, base: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p)


def _build_logger_file_path() -> Path | None:
    global _LOGGER_FILE_PATH_CACHED

    if _LOGGER_FILE_PATH_CACHED is not _LOGGER_FILE_PATH_SENTINEL:
        return None if _LOGGER_FILE_PATH_CACHED is None else _LOGGER_FILE_PATH_CACHED  # type: ignore[return-value]

    if not _env_bool("LOGGER_FILE_ENABLED", False):
        _LOGGER_FILE_PATH_CACHED = None
        return None

    raw_path = _env_str("LOGGER_FILE_PATH", "").strip()
    if raw_path:
        p = Path(raw_path)
        resolved = p if p.is_absolute() else (SERVER_DIR / p)
        _LOGGER_FILE_PATH_CACHED = resolved.resolve()
        return _LOGGER_FILE_PATH_CACHED

    raw_dir = _env_str("LOGGER_FILE_DIR", "logs") or "logs"
    log_dir = _resolve_dir(raw_dir, base=SERVER_DIR)
    prefix = (
        _sanitize_filename_component(_env_str("LOGGER_FILE_PREFIX", "run") or "run")
        or "run"
    )
    ts_fmt = (
        _env_str("LOGGER_FILE_TIMESTAMP_FORMAT", "%Y-%m-%d_%H-%M-%S")
        or "%Y-%m-%d_%H-%M-%S"
    )
    include_pid = _env_bool("LOGGER_FILE_INCLUDE_PID", True)

    ts = datetime.now().strftime(ts_fmt)
    pid_part = f"_{os.getpid()}" if include_pid else ""
    filename = f"{prefix}_{ts}{pid_part}.log"
    _LOGGER_FILE_PATH_CACHED = (log_dir / filename).resolve()
    return _LOGGER_FILE_PATH_CACHED


def _has_our_file_handler(root: logging.Logger) -> bool:
    for handler in root.handlers:
        try:
            if bool(getattr(handler, _FILE_HANDLER_TAG, False)):
                return True
        except Exception:
            pass
    return False


def _ensure_file_handler(root: logging.Logger, *, level: str) -> None:
    path = _build_logger_file_path()
    if path is None:
        return

    if _has_our_file_handler(root):
        for handler in root.handlers:
            try:
                if bool(getattr(handler, _FILE_HANDLER_TAG, False)):
                    handler.setLevel(level)
            except Exception:
                pass
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        setattr(handler, _FILE_HANDLER_TAG, True)
        root.addHandler(handler)
    except Exception:
        pass


def configure_logging(settings: Settings) -> logging.Logger:
    """
    Configuración mínima:
    - Respetamos handlers/format de quien ejecute (uvicorn, gunicorn, etc.).
    - Ajustamos nivel global según env.
    """
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    _ensure_file_handler(root, level=settings.log_level)

    logger = logging.getLogger("analiza_api")
    logger.setLevel(settings.log_level)
    return logger
