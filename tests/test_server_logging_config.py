import logging
import os

from server.api import logging_config
from server.api.settings import Settings


def _settings(level: str = "INFO") -> Settings:
    return Settings(
        log_level=level,
        cors_origins_raw="*",
        cors_allow_credentials=False,
        gzip_min_size=0,
        file_cache_max_entries=1,
        file_cache_ttl_seconds=0.0,
        file_read_max_attempts=1,
        file_read_retry_sleep_s=0.0,
    )


def _reset_logger_cache() -> None:
    logging_config._LOGGER_FILE_PATH_CACHED = (
        logging_config._LOGGER_FILE_PATH_SENTINEL
    )


def test_sanitize_filename_component():
    assert logging_config._sanitize_filename_component(" run 1 ") == "run_1"
    assert logging_config._sanitize_filename_component("..") == ""
    assert logging_config._sanitize_filename_component("a/b") == "a_b"


def test_resolve_dir_relative_and_absolute(tmp_path):
    base = tmp_path / "base"
    assert logging_config._resolve_dir(str(tmp_path), base=base) == tmp_path
    assert logging_config._resolve_dir("logs", base=base) == base / "logs"


def test_build_logger_file_path_disabled(monkeypatch):
    _reset_logger_cache()
    monkeypatch.delenv("LOGGER_FILE_ENABLED", raising=False)
    monkeypatch.setenv("LOGGER_FILE_ENABLED", "0")
    assert logging_config._build_logger_file_path() is None
    assert logging_config._build_logger_file_path() is None


def test_build_logger_file_path_explicit(monkeypatch, tmp_path):
    _reset_logger_cache()
    target = tmp_path / "app.log"
    monkeypatch.setenv("LOGGER_FILE_ENABLED", "1")
    monkeypatch.setenv("LOGGER_FILE_PATH", str(target))
    path = logging_config._build_logger_file_path()
    assert path == target.resolve()


def test_build_logger_file_path_dir_prefix(monkeypatch, tmp_path):
    _reset_logger_cache()
    monkeypatch.setenv("LOGGER_FILE_ENABLED", "1")
    monkeypatch.setenv("LOGGER_FILE_DIR", str(tmp_path))
    monkeypatch.setenv("LOGGER_FILE_PREFIX", "run*bad")
    monkeypatch.setenv("LOGGER_FILE_TIMESTAMP_FORMAT", "%Y")
    monkeypatch.setenv("LOGGER_FILE_INCLUDE_PID", "0")

    path = logging_config._build_logger_file_path()
    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith("run_bad_")
    assert path.suffix == ".log"
    assert str(os.getpid()) not in path.name


def test_configure_logging_adds_file_handler(monkeypatch, tmp_path):
    _reset_logger_cache()
    monkeypatch.setenv("LOGGER_FILE_ENABLED", "1")
    monkeypatch.setenv("LOGGER_FILE_DIR", str(tmp_path))

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, logging_config._FILE_HANDLER_TAG, False):
            root.removeHandler(handler)

    logger = logging_config.configure_logging(_settings(level="DEBUG"))
    assert logger.level == logging.DEBUG
    assert logging_config._has_our_file_handler(root) is True

    for handler in list(root.handlers):
        if getattr(handler, logging_config._FILE_HANDLER_TAG, False):
            root.removeHandler(handler)
