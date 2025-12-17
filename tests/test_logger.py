# tests/test_logger.py
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from backend import logger as logger_mod


# ----------------------------------------------------------------------
# Tests para _ensure_configured
# ----------------------------------------------------------------------


def test_ensure_configured_initializes_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reset estado interno
    monkeypatch.setattr(logger_mod, "_CONFIGURED", False)
    monkeypatch.setattr(logger_mod, "_LOGGER", None)

    log = logger_mod._ensure_configured()

    assert isinstance(log, logging.Logger)
    assert logger_mod._CONFIGURED is True
    assert logger_mod._LOGGER is log
    # logger name from module constant
    assert log.name == logger_mod.LOGGER_NAME


def test_ensure_configured_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reset y configurar una vez
    monkeypatch.setattr(logger_mod, "_CONFIGURED", False)
    monkeypatch.setattr(logger_mod, "_LOGGER", None)

    log1 = logger_mod._ensure_configured()
    log2 = logger_mod._ensure_configured()

    assert log1 is log2
    assert logger_mod._CONFIGURED is True


# ----------------------------------------------------------------------
# Tests para _should_log
# ----------------------------------------------------------------------


def test_should_log_when_silent_mode_false(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cfg = SimpleNamespace(SILENT_MODE=False)
    monkeypatch.setitem(sys.modules, "backend.config", fake_cfg)

    assert logger_mod._should_log() is True
    assert logger_mod._should_log(always=False) is True


def test_should_log_silent_mode_true(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cfg = SimpleNamespace(SILENT_MODE=True)
    monkeypatch.setitem(sys.modules, "backend.config", fake_cfg)

    assert logger_mod._should_log() is False
    assert logger_mod._should_log(always=True) is True


def test_should_log_config_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simula que backend.config no existe
    monkeypatch.delitem(sys.modules, "backend.config", raising=False)

    # Debe default a True
    assert logger_mod._should_log() is True


# ----------------------------------------------------------------------
# Fixture para capturar llamadas a logger
# ----------------------------------------------------------------------


@pytest.fixture
def capture_logger(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, list[str]], object]:
    """
    Reemplaza el logger real por uno falso capturando msgs en una lista.
    """
    messages: dict[str, list[str]] = {
        "info": [],
        "warning": [],
        "error": [],
        "debug": [],
    }

    class FakeLogger:
        def info(self, msg: str) -> None:
            messages["info"].append(msg)

        def warning(self, msg: str) -> None:
            messages["warning"].append(msg)

        def error(self, msg: str) -> None:
            messages["error"].append(msg)

        def debug(self, msg: str) -> None:
            messages["debug"].append(msg)

    fake = FakeLogger()
    monkeypatch.setattr(logger_mod, "_LOGGER", fake)
    monkeypatch.setattr(logger_mod, "_CONFIGURED", True)

    return messages, fake


# ----------------------------------------------------------------------
# Tests de los métodos públicos: info / warning / error / debug
# ----------------------------------------------------------------------


def test_logger_info_writes_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=False),
    )

    logger_mod.info("Hello")
    assert messages["info"] == ["Hello"]


def test_logger_info_suppressed_when_silent(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=True),
    )

    logger_mod.info("Hello")
    # No escribe
    assert messages["info"] == []


def test_logger_info_always_forces_log(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=True),
    )

    logger_mod.info("Forced", always=True)
    assert messages["info"] == ["Forced"]


def test_logger_warning(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=False),
    )

    logger_mod.warning("Warn")
    assert messages["warning"] == ["Warn"]


def test_logger_error(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=False),
    )

    logger_mod.error("ERR")
    assert messages["error"] == ["ERR"]


def test_logger_debug(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=False),
    )

    logger_mod.debug("DBG")
    assert messages["debug"] == ["DBG"]


def test_logger_debug_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=True),
    )

    logger_mod.debug("DBG")
    assert messages["debug"] == []


def test_logger_error_always(
    monkeypatch: pytest.MonkeyPatch,
    capture_logger: tuple[dict[str, list[str]], object],
) -> None:
    messages, _ = capture_logger
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(SILENT_MODE=True),
    )

    logger_mod.error("ERR", always=True)
    assert messages["error"] == ["ERR"]