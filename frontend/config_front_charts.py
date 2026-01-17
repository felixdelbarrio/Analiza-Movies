"""
frontend/config_front_charts.py

Configuracion de graficos del frontend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

from dotenv import dotenv_values

from frontend.config_front_base import PROJECT_DIR

ENV_KEY_DASHBOARD_VIEWS: Final[str] = "FRONT_DASHBOARD_VIEWS"
ENV_KEY_SHOW_NUMERIC_FILTERS: Final[str] = "FRONT_SHOW_NUMERIC_FILTERS"
ENV_KEY_SHOW_CHART_THRESHOLDS: Final[str] = "FRONT_SHOW_CHART_THRESHOLDS"

DASHBOARD_DEFAULT_VIEWS: Final[tuple[str, ...]] = (
    "Ratings IMDb vs Metacritic",
    "Distribución por decisión",
    "Boxplot IMDb por biblioteca",
)

_ENV_FRONT_PATH: Final[Path] = PROJECT_DIR / ".env.front"


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_views(raw: str | None, available: Iterable[str]) -> list[str]:
    if not raw:
        return []
    allowed = {v for v in available}
    out: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name in allowed and name not in out:
            out.append(name)
    return out


def get_dashboard_views(available: Iterable[str]) -> list[str]:
    available_list = list(available)
    if not _ENV_FRONT_PATH.exists():
        return [v for v in DASHBOARD_DEFAULT_VIEWS if v in available_list]
    env = dotenv_values(_ENV_FRONT_PATH)
    raw = env.get(ENV_KEY_DASHBOARD_VIEWS)
    parsed = _parse_views(raw, available_list)
    if parsed:
        return parsed[:3]
    fallback = [v for v in DASHBOARD_DEFAULT_VIEWS if v in available_list]
    return fallback[:3] if fallback else available_list[:3]


def get_show_numeric_filters() -> bool:
    if not _ENV_FRONT_PATH.exists():
        return False
    env = dotenv_values(_ENV_FRONT_PATH)
    return _parse_bool(env.get(ENV_KEY_SHOW_NUMERIC_FILTERS), False)


def get_show_chart_thresholds() -> bool:
    if not _ENV_FRONT_PATH.exists():
        return False
    env = dotenv_values(_ENV_FRONT_PATH)
    return _parse_bool(env.get(ENV_KEY_SHOW_CHART_THRESHOLDS), False)


def _save_bool_env(key: str, value: bool) -> None:
    payload = "true" if value else "false"
    lines: list[str] = []
    if _ENV_FRONT_PATH.exists():
        lines = _ENV_FRONT_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            out.append(f"{key}={payload}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={payload}")
    _ENV_FRONT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def save_dashboard_views(views: Iterable[str]) -> None:
    values = [v.strip() for v in views if str(v).strip()]
    payload = ",".join(values[:3])
    lines: list[str] = []
    if _ENV_FRONT_PATH.exists():
        lines = _ENV_FRONT_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{ENV_KEY_DASHBOARD_VIEWS}="):
            out.append(f"{ENV_KEY_DASHBOARD_VIEWS}={payload}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{ENV_KEY_DASHBOARD_VIEWS}={payload}")
    _ENV_FRONT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def save_show_numeric_filters(value: bool) -> None:
    _save_bool_env(ENV_KEY_SHOW_NUMERIC_FILTERS, value)


def save_show_chart_thresholds(value: bool) -> None:
    _save_bool_env(ENV_KEY_SHOW_CHART_THRESHOLDS, value)
