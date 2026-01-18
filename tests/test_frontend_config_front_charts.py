from __future__ import annotations

from pathlib import Path

import pytest

from frontend import config_front_charts as charts_config


def test_get_dashboard_views_falls_back_to_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env.front"
    monkeypatch.setattr(charts_config, "_ENV_FRONT_PATH", env_path)
    available = [
        "Ratings IMDb vs Metacritic",
        "Distribución por decisión",
        "Boxplot IMDb por biblioteca",
        "Otro",
    ]

    assert charts_config.get_dashboard_views(available) == list(
        charts_config.DASHBOARD_DEFAULT_VIEWS
    )


def test_get_dashboard_views_parses_env_and_dedupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env.front"
    env_path.write_text(
        "FRONT_DASHBOARD_VIEWS=Otro,Distribución por decisión,Otro\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(charts_config, "_ENV_FRONT_PATH", env_path)
    available = [
        "Ratings IMDb vs Metacritic",
        "Distribución por decisión",
        "Boxplot IMDb por biblioteca",
        "Otro",
    ]

    assert charts_config.get_dashboard_views(available) == [
        "Otro",
        "Distribución por decisión",
    ]


def test_get_show_numeric_filters_reads_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env.front"
    env_path.write_text("FRONT_SHOW_NUMERIC_FILTERS=true\n", encoding="utf-8")
    monkeypatch.setattr(charts_config, "_ENV_FRONT_PATH", env_path)

    assert charts_config.get_show_numeric_filters() is True


def test_save_dashboard_views_overwrites_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env.front"
    env_path.write_text("FRONT_DASHBOARD_VIEWS=Old\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(charts_config, "_ENV_FRONT_PATH", env_path)

    charts_config.save_dashboard_views(
        ["Distribución por decisión", "Boxplot IMDb por biblioteca"]
    )

    contents = env_path.read_text(encoding="utf-8")
    assert (
        "FRONT_DASHBOARD_VIEWS=Distribución por decisión,Boxplot IMDb por biblioteca"
        in contents
    )
