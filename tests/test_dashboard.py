from __future__ import annotations

import os
from typing import Any, List

import pandas as pd
import pytest
import streamlit as st

import dashboard  # se importa el script principal


# -------------------------------------------------------------------
# Tests de _env_bool
# -------------------------------------------------------------------


def test_env_bool_default_when_not_set(monkeypatch) -> None:
    """Si la variable no existe, devuelve el default."""
    monkeypatch.delenv("X_TEST_BOOL", raising=False)
    assert dashboard._env_bool("X_TEST_BOOL", True) is True
    assert dashboard._env_bool("X_TEST_BOOL", False) is False


def test_env_bool_true_values(monkeypatch) -> None:
    """Reconoce correctamente 'true' (case/espacios) como True."""
    for val in ["true", " True ", "TrUe", "\ttrue\n"]:
        monkeypatch.setenv("X_TEST_BOOL", val)
        assert dashboard._env_bool("X_TEST_BOOL", False) is True


def test_env_bool_false_values(monkeypatch) -> None:
    """Cualquier cosa distinta de 'true' se considera False."""
    for val in ["false", "0", "yes", "on", "", " "]:
        monkeypatch.setenv("X_TEST_BOOL", val)
        assert dashboard._env_bool("X_TEST_BOOL", True) is False


# -------------------------------------------------------------------
# Tests de _init_modal_state
# -------------------------------------------------------------------


def test_init_modal_state_creates_keys(monkeypatch) -> None:
    """Debe crear modal_open=False y modal_row=None cuando no existen."""
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)

    dashboard._init_modal_state()

    assert state["modal_open"] is False
    assert state["modal_row"] is None


def test_init_modal_state_preserves_existing(monkeypatch) -> None:
    """Si ya hay valores en session_state, no debe machacarlos."""
    state: dict[str, Any] = {
        "modal_open": True,
        "modal_row": {"id": 123},
        "other": "keep",
    }
    monkeypatch.setattr(st, "session_state", state, raising=False)

    dashboard._init_modal_state()

    assert state["modal_open"] is True
    assert state["modal_row"] == {"id": 123}
    assert state["other"] == "keep"


# -------------------------------------------------------------------
# Tests de _hide_streamlit_chrome
# -------------------------------------------------------------------


def test_hide_streamlit_chrome_calls_markdown_with_css(monkeypatch) -> None:
    """Debe llamar a st.markdown con el CSS adecuado y unsafe_allow_html=True."""
    calls: List[dict[str, Any]] = []

    def fake_markdown(text: str, unsafe_allow_html: bool = False):
        calls.append({"text": text, "unsafe": unsafe_allow_html})

    monkeypatch.setattr(st, "markdown", fake_markdown, raising=False)

    dashboard._hide_streamlit_chrome()

    assert len(calls) == 1
    call = calls[0]
    assert "stHeader" in call["text"]  # algo muy característico del CSS
    assert call["unsafe"] is True


# -------------------------------------------------------------------
# Tests de _log_effective_thresholds_once
# -------------------------------------------------------------------


def test_log_effective_thresholds_once_already_logged(monkeypatch) -> None:
    """Si thresholds_logged ya está a True, no debe llamar a stats ni logger."""
    # Session state simulado
    state: dict[str, Any] = {"thresholds_logged": True}
    monkeypatch.setattr(st, "session_state", state, raising=False)

    # Si se llaman estas funciones, fallamos el test
    def fail_keep():
        raise AssertionError("get_auto_keep_rating_threshold no debería llamarse")

    def fail_delete():
        raise AssertionError("get_auto_delete_rating_threshold no debería llamarse")

    def fail_mean():
        raise AssertionError("get_global_imdb_mean_info no debería llamarse")

    monkeypatch.setattr(
        dashboard,
        "get_auto_keep_rating_threshold",
        fail_keep,
        raising=False,
    )
    monkeypatch.setattr(
        dashboard,
        "get_auto_delete_rating_threshold",
        fail_delete,
        raising=False,
    )
    monkeypatch.setattr(
        dashboard,
        "get_global_imdb_mean_info",
        fail_mean,
        raising=False,
    )

    # Logger.info tampoco debería llamarse
    def fail_info(msg: str):
        raise AssertionError(f"_logger.info no debería llamarse (msg={msg!r})")

    monkeypatch.setattr(dashboard._logger, "info", fail_info, raising=False)

    dashboard._log_effective_thresholds_once()

    # El flag debe seguir en True
    assert state["thresholds_logged"] is True


def test_log_effective_thresholds_once_silent_mode(monkeypatch) -> None:
    """Con SILENT_MODE=True debe marcar thresholds_logged sin llamar a stats ni logger."""
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)

    # Forzamos SILENT_MODE=True en el módulo dashboard
    monkeypatch.setattr(dashboard, "SILENT_MODE", True, raising=False)

    # Stats y logger no deben llamarse
    def fail_keep():
        raise AssertionError("get_auto_keep_rating_threshold no debería llamarse en SILENT_MODE")

    def fail_delete():
        raise AssertionError("get_auto_delete_rating_threshold no debería llamarse en SILENT_MODE")

    def fail_mean():
        raise AssertionError("get_global_imdb_mean_info no debería llamarse en SILENT_MODE")

    def fail_info(msg: str):
        raise AssertionError(f"_logger.info no debería llamarse en SILENT_MODE (msg={msg!r})")

    monkeypatch.setattr(
        dashboard,
        "get_auto_keep_rating_threshold",
        fail_keep,
        raising=False,
    )
    monkeypatch.setattr(
        dashboard,
        "get_auto_delete_rating_threshold",
        fail_delete,
        raising=False,
    )
    monkeypatch.setattr(
        dashboard,
        "get_global_imdb_mean_info",
        fail_mean,
        raising=False,
    )
    monkeypatch.setattr(dashboard._logger, "info", fail_info, raising=False)

    dashboard._log_effective_thresholds_once()

    assert state.get("thresholds_logged") is True


def test_log_effective_thresholds_once_normal_logging(monkeypatch) -> None:
    """Con SILENT_MODE=False debe llamar a stats y logger, y marcar thresholds_logged."""
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)

    monkeypatch.setattr(dashboard, "SILENT_MODE", False, raising=False)

    # Stats dummy con valores conocidos
    def fake_keep():
        return 6.5

    def fake_delete():
        return 4.2

    def fake_mean_info():
        return 6.1, "cache", 1234

    monkeypatch.setattr(
        dashboard,
        "get_auto_keep_rating_threshold",
        fake_keep,
        raising=False,
    )
    monkeypatch.setattr(
        dashboard,
        "get_auto_delete_rating_threshold",
        fake_delete,
        raising=False,
    )
    monkeypatch.setattr(
        dashboard,
        "get_global_imdb_mean_info",
        fake_mean_info,
        raising=False,
    )

    logs: list[str] = []

    def fake_info(msg: str):
        logs.append(msg)

    monkeypatch.setattr(dashboard._logger, "info", fake_info, raising=False)

    dashboard._log_effective_thresholds_once()

    # Debe marcar el flag
    assert state.get("thresholds_logged") is True

    # Debe haber habido algún log
    assert logs, "Se esperaba al menos una llamada a _logger.info"

    # Comprobamos que se han logueado algunos textos clave
    joined = "\n".join(logs)
    assert "UMBRAL" in joined.upper() or "UMBR" in joined.upper()
    assert "KEEP efectivo" in joined or "KEEP" in joined
    assert "DELETE" in joined