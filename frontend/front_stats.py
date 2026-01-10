"""
frontend/front_stats.py

Stats del frontend (desacoplado del backend).

- compute_global_imdb_mean_from_df: para DataFrame cargado desde report_all.csv
- compute_global_imdb_mean_from_report_all: helper que lee REPORT_ALL_PATH
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from frontend.config_front_artifacts import REPORT_ALL_PATH


def compute_global_imdb_mean_from_df(df_all: pd.DataFrame) -> float | None:
    if "imdb_rating" not in df_all.columns:
        return None

    ratings = pd.to_numeric(df_all["imdb_rating"], errors="coerce").dropna()
    if ratings.empty:
        return None
    return float(ratings.mean())


def compute_global_imdb_mean_from_report_all(path: Path | None = None) -> float | None:
    p = path or REPORT_ALL_PATH
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    return compute_global_imdb_mean_from_df(df)
