from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from typing import List

import pytest

# ---------------------------------------------------------
# Carga robusta de analiza.py desde la raíz del proyecto
# ---------------------------------------------------------

THIS_FILE = pathlib.Path(__file__).resolve()
TESTS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = TESTS_DIR.parent
ANALIZA_PATH = PROJECT_ROOT / "analiza.py"

if not ANALIZA_PATH.exists():
    pytest.skip(
        f"analiza.py no encontrado en {ANALIZA_PATH}, se omiten tests de analiza.",
        allow_module_level=True,
    )

spec = importlib.util.spec_from_file_location("analiza", ANALIZA_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo crear spec para {ANALIZA_PATH}")

analiza_mod = importlib.util.module_from_spec(spec)
sys.modules["analiza"] = analiza_mod
spec.loader.exec_module(analiza_mod)  # type: ignore[assignment]

# Importamos DLNADevice solo para construir datos de prueba
from backend.dlna_discovery import DLNADevice  # noqa: E402


# ---------------------------------------------------------
# Tests para _ask_source
# ---------------------------------------------------------


def test_ask_source_accepts_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ask_source debe devolver '1' cuando el usuario introduce 1."""
    inputs = ["1"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    choice = analiza_mod._ask_source()
    assert choice == "1"


def test_ask_source_accepts_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ask_source debe devolver '2' cuando el usuario introduce 2."""
    inputs = ["2"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    choice = analiza_mod._ask_source()
    assert choice == "2"


def test_ask_source_repeats_until_valid(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Si el usuario introduce valores inválidos, debe repetir hasta recibir 1/2."""
    inputs = ["x", "", "3", "2"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    choice = analiza_mod._ask_source()
    captured = capsys.readouterr()

    assert choice == "2"
    # Debe haber mostrado al menos un mensaje de opción no válida
    assert "Opción no válida" in captured.out


# ---------------------------------------------------------
# Tests para _select_dlna_device
# ---------------------------------------------------------


def test_select_dlna_device_no_devices(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Si discover_dlna_devices devuelve lista vacía, _select_dlna_device debe devolver None."""
    monkeypatch.setattr(analiza_mod, "discover_dlna_devices", lambda: [])

    result = analiza_mod._select_dlna_device()
    captured = capsys.readouterr()

    assert result is None
    assert "No se han encontrado servidores DLNA/UPnP" in captured.out


def test_select_dlna_device_cancel(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Si el usuario pulsa Enter en la selección, debe devolver None."""
    devices: List[DLNADevice] = [
        DLNADevice(
            friendly_name="Servidor 1",
            location="http://192.168.1.10:1234/desc.xml",
            host="192.168.1.10",
            port=1234,
        ),
    ]

    monkeypatch.setattr(analiza_mod, "discover_dlna_devices", lambda: devices)
    # Usuario ve la lista y pulsa Enter para cancelar
    inputs = [""]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    result = analiza_mod._select_dlna_device()
    captured = capsys.readouterr()

    assert result is None
    assert "Operación cancelada" in captured.out


def test_select_dlna_device_happy_path(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Camino normal: varios servidores, usuario selecciona uno válido."""
    devices: List[DLNADevice] = [
        DLNADevice(
            friendly_name="Servidor A",
            location="http://10.0.0.1:1234/desc.xml",
            host="10.0.0.1",
            port=1234,
        ),
        DLNADevice(
            friendly_name="Servidor B",
            location="http://10.0.0.2:5678/desc.xml",
            host="10.0.0.2",
            port=5678,
        ),
    ]

    monkeypatch.setattr(analiza_mod, "discover_dlna_devices", lambda: devices)
    # Usuario elige la opción 2 (Servidor B)
    inputs = ["2"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    result = analiza_mod._select_dlna_device()
    captured = capsys.readouterr()

    assert isinstance(result, DLNADevice)
    assert result.friendly_name == "Servidor B"
    # Debe haber mostrado la confirmación
    assert "Has seleccionado: Servidor B" in captured.out


def test_select_dlna_device_invalid_then_valid(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Si el usuario introduce un número fuera de rango, se le vuelve a pedir."""
    devices: List[DLNADevice] = [
        DLNADevice(
            friendly_name="Servidor A",
            location="http://10.0.0.1:1234/desc.xml",
            host="10.0.0.1",
            port=1234,
        ),
    ]

    monkeypatch.setattr(analiza_mod, "discover_dlna_devices", lambda: devices)
    # Primero '0' (inválido), luego '1' (válido)
    inputs = ["0", "1"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    result = analiza_mod._select_dlna_device()
    captured = capsys.readouterr()

    assert isinstance(result, DLNADevice)
    assert result.friendly_name == "Servidor A"
    assert "Opción no válida" in captured.out


# ---------------------------------------------------------
# Tests para main()
# ---------------------------------------------------------


def test_main_plex_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si _ask_source devuelve '1', main debe llamar a analyze_all_libraries."""
    calls = {"plex": 0, "dlna": 0}

    monkeypatch.setattr(analiza_mod, "_ask_source", lambda: "1")
    monkeypatch.setattr(
        analiza_mod,
        "analyze_all_libraries",
        lambda: calls.__setitem__("plex", calls["plex"] + 1),
    )
    monkeypatch.setattr(
        analiza_mod,
        "analyze_dlna_server",
        lambda: calls.__setitem__("dlna", calls["dlna"] + 1),
    )

    analiza_mod.main()

    assert calls["plex"] == 1
    assert calls["dlna"] == 0


def test_main_dlna_branch_with_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si _ask_source devuelve '2' y hay dispositivo seleccionado, se debe llamar a analyze_dlna_server."""
    calls = {"plex": 0, "dlna": 0}

    monkeypatch.setattr(analiza_mod, "_ask_source", lambda: "2")

    # Simulamos que el usuario selecciona un dispositivo (no importa cuál)
    fake_device = DLNADevice(
        friendly_name="Servidor X",
        location="http://10.0.0.3:1234/desc.xml",
        host="10.0.0.3",
        port=1234,
    )
    monkeypatch.setattr(analiza_mod, "_select_dlna_device", lambda: fake_device)

    monkeypatch.setattr(
        analiza_mod,
        "analyze_all_libraries",
        lambda: calls.__setitem__("plex", calls["plex"] + 1),
    )
    monkeypatch.setattr(
        analiza_mod,
        "analyze_dlna_server",
        lambda: calls.__setitem__("dlna", calls["dlna"] + 1),
    )

    analiza_mod.main()

    assert calls["plex"] == 0
    assert calls["dlna"] == 1


def test_main_dlna_branch_with_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si _select_dlna_device devuelve None, main no debe llamar a analyze_dlna_server."""
    calls = {"plex": 0, "dlna": 0}

    monkeypatch.setattr(analiza_mod, "_ask_source", lambda: "2")
    monkeypatch.setattr(analiza_mod, "_select_dlna_device", lambda: None)

    monkeypatch.setattr(
        analiza_mod,
        "analyze_all_libraries",
        lambda: calls.__setitem__("plex", calls["plex"] + 1),
    )
    monkeypatch.setattr(
        analiza_mod,
        "analyze_dlna_server",
        lambda: calls.__setitem__("dlna", calls["dlna"] + 1),
    )

    analiza_mod.main()

    assert calls["plex"] == 0
    assert calls["dlna"] == 0