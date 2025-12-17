from __future__ import annotations

import os

import pytest

from backend import config


# -------------------------------------------------------------------
# Tests para _get_env_int
# -------------------------------------------------------------------


def test_get_env_int_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Valor válido
    monkeypatch.setenv("TEST_INT_OK", "123")
    assert config._get_env_int("TEST_INT_OK", 5) == 123

    # Valor no convertible → usa default
    monkeypatch.setenv("TEST_INT_BAD", "not-an-int")
    assert config._get_env_int("TEST_INT_BAD", 7) == 7

    # Variable no definida → usa default
    monkeypatch.delenv("TEST_INT_MISSING", raising=False)
    assert config._get_env_int("TEST_INT_MISSING", 9) == 9

    # Cadena vacía → default
    monkeypatch.setenv("TEST_INT_EMPTY", "")
    assert config._get_env_int("TEST_INT_EMPTY", 11) == 11


# -------------------------------------------------------------------
# Tests para _get_env_float
# -------------------------------------------------------------------


def test_get_env_float_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Valor válido
    monkeypatch.setenv("TEST_FLOAT_OK", "3.14")
    val = config._get_env_float("TEST_FLOAT_OK", 1.0)
    assert isinstance(val, float)
    assert abs(val - 3.14) < 1e-6

    # Valor no convertible → default
    monkeypatch.setenv("TEST_FLOAT_BAD", "not-float")
    assert config._get_env_float("TEST_FLOAT_BAD", 2.5) == 2.5

    # Variable no definida → default
    monkeypatch.delenv("TEST_FLOAT_MISSING", raising=False)
    assert config._get_env_float("TEST_FLOAT_MISSING", 4.2) == 4.2

    # Cadena vacía → default
    monkeypatch.setenv("TEST_FLOAT_EMPTY", "")
    assert config._get_env_float("TEST_FLOAT_EMPTY", 8.8) == 8.8


# -------------------------------------------------------------------
# Tests para _get_env_bool
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),  # se cae al default si lo usamos así
    ],
)
def test_get_env_bool_variants(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("TEST_BOOL", value)
    # Default no importa salvo cuando value == "" (en cuyo caso no se usa VALUE)
    assert config._get_env_bool("TEST_BOOL", False) is expected


def test_get_env_bool_uses_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_BOOL_MISSING", raising=False)
    assert config._get_env_bool("TEST_BOOL_MISSING", True) is True
    assert config._get_env_bool("TEST_BOOL_MISSING", False) is False


# -------------------------------------------------------------------
# Tests para _parse_votes_by_year
# -------------------------------------------------------------------


def test_parse_votes_by_year_basic() -> None:
    raw = "1980:500,2000:2000,2010:5000,9999:10000"
    table = config._parse_votes_by_year(raw)  # type: ignore[attr-defined]
    assert table == [
        (1980, 500),
        (2000, 2000),
        (2010, 5000),
        (9999, 10000),
    ]


def test_parse_votes_by_year_with_quotes_and_spaces() -> None:
    raw = '" 1980:500 ,  2000:2000 ,2010:5000 , 9999:10000  "'
    table = config._parse_votes_by_year(raw)  # type: ignore[attr-defined]
    assert table == [
        (1980, 500),
        (2000, 2000),
        (2010, 5000),
        (9999, 10000),
    ]


def test_parse_votes_by_year_ignores_malformed_chunks() -> None:
    raw = "1980:500, bad-chunk , 1990:not-int, 2000:2000"
    table = config._parse_votes_by_year(raw)  # type: ignore[attr-defined]
    # Solo deben entrar los bien formados
    assert table == [
        (1980, 500),
        (2000, 2000),
    ]


def test_parse_votes_by_year_empty_or_none() -> None:
    assert config._parse_votes_by_year("") == []  # type: ignore[attr-defined]


# -------------------------------------------------------------------
# Tests para get_votes_threshold_for_year
# -------------------------------------------------------------------


def test_get_votes_threshold_for_year_with_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # Forzamos una tabla conocida
    monkeypatch.setattr(
        config,
        "IMDB_VOTES_BY_YEAR",
        [(1980, 100), (2000, 200), (9999, 300)],
        raising=True,
    )

    f = config.get_votes_threshold_for_year

    # 1970 → primera entrada (1980,100)
    assert f(1970) == 100
    # 1980 → sigue primera
    assert f(1980) == 100
    # 1995 → segunda (2000,200)
    assert f(1995) == 200
    # 2010 → última (9999,300)
    assert f(2010) == 300


def test_get_votes_threshold_for_year_none_or_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config,
        "IMDB_VOTES_BY_YEAR",
        [(1980, 100), (2000, 200), (9999, 300)],
        raising=True,
    )
    f = config.get_votes_threshold_for_year

    # year=None → tramo más exigente (último)
    assert f(None) == 300
    # year no convertible → también último
    assert f("not-a-year") == 300  # type: ignore[arg-type]


def test_get_votes_threshold_for_year_empty_table_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tabla vacía → usa IMDB_KEEP_MIN_VOTES como fallback
    monkeypatch.setattr(config, "IMDB_VOTES_BY_YEAR", [], raising=True)
    monkeypatch.setattr(config, "IMDB_KEEP_MIN_VOTES", 12345, raising=True)

    assert config.get_votes_threshold_for_year(1990) == 12345
    assert config.get_votes_threshold_for_year(None) == 12345