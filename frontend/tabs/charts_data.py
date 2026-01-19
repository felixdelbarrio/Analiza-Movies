"""
Data helpers for charts.
"""

from __future__ import annotations

import pandas as pd

from frontend.data_utils import (
    build_word_counts,
    directors_from_omdb_json_or_cache,
    explode_genres_from_omdb_json,
)
from frontend.tabs.charts_shared import (
    IMDB_OUTLIER_HIGH,
    IMDB_OUTLIER_LOW,
    _cache_data_decorator,
)


def _mark_imdb_outliers(
    data: pd.DataFrame,
    low: float = IMDB_OUTLIER_LOW,
    high: float = IMDB_OUTLIER_HIGH,
) -> pd.DataFrame:
    out = data.copy()
    out["imdb_outlier"] = pd.NA
    out.loc[out["imdb_rating"] >= high, "imdb_outlier"] = "Alta"
    out.loc[out["imdb_rating"] <= low, "imdb_outlier"] = "Baja"
    return out


@_cache_data_decorator()
def _genres_agg(df: pd.DataFrame) -> pd.DataFrame:
    if "omdb_json" not in df.columns and "imdb_id" not in df.columns:
        return pd.DataFrame(columns=["genre", "decision", "count"])

    cols: list[str] = ["decision", "title"]
    if "omdb_json" in df.columns:
        cols.append("omdb_json")
    if "imdb_id" in df.columns:
        cols.append("imdb_id")

    df_gen = explode_genres_from_omdb_json(df.loc[:, cols].copy())
    if "genre" in df_gen.columns:
        df_gen = df_gen[df_gen["genre"].notna() & (df_gen["genre"] != "")]

    if df_gen.empty:
        return pd.DataFrame(columns=["genre", "decision", "count"])

    return (
        df_gen.groupby(["genre", "decision"], dropna=False)["title"]
        .count()
        .reset_index()
        .rename(columns={"title": "count"})
    )


@_cache_data_decorator()
def _director_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "omdb_json" not in df.columns and "imdb_id" not in df.columns:
        return pd.DataFrame(columns=["director_list", "imdb_mean", "count"])

    cols: list[str] = ["imdb_rating", "title"]
    if "omdb_json" in df.columns:
        cols.append("omdb_json")
    if "imdb_id" in df.columns:
        cols.append("imdb_id")

    df_dir = df.loc[:, cols].copy()
    if "omdb_json" in df_dir.columns:
        omdb_vals = df_dir["omdb_json"]
    else:
        omdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    if "imdb_id" in df_dir.columns:
        imdb_vals = df_dir["imdb_id"]
    else:
        imdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    df_dir["director_list"] = [
        directors_from_omdb_json_or_cache(omdb_raw, imdb_id)
        for omdb_raw, imdb_id in zip(omdb_vals, imdb_vals)
    ]
    df_dir = df_dir.explode("director_list", ignore_index=True)
    df_dir = df_dir[df_dir["director_list"].notna() & (df_dir["director_list"] != "")]

    if df_dir.empty:
        return pd.DataFrame(columns=["director_list", "imdb_mean", "count"])

    return (
        df_dir.groupby("director_list", dropna=False)
        .agg(
            imdb_mean=("imdb_rating", "mean"),
            count=("title", "count"),
        )
        .reset_index()
    )


@_cache_data_decorator()
def _director_decision_stats(df: pd.DataFrame) -> pd.DataFrame:
    if not {"imdb_rating", "title", "decision"}.issubset(df.columns):
        return pd.DataFrame(
            columns=["director_list", "decision", "count", "imdb_mean", "count_total"]
        )
    if "omdb_json" not in df.columns and "imdb_id" not in df.columns:
        return pd.DataFrame(
            columns=["director_list", "decision", "count", "imdb_mean", "count_total"]
        )

    cols: list[str] = ["imdb_rating", "title", "decision"]
    if "omdb_json" in df.columns:
        cols.append("omdb_json")
    if "imdb_id" in df.columns:
        cols.append("imdb_id")

    df_dir = df.loc[:, cols].copy()
    if "omdb_json" in df_dir.columns:
        omdb_vals = df_dir["omdb_json"]
    else:
        omdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    if "imdb_id" in df_dir.columns:
        imdb_vals = df_dir["imdb_id"]
    else:
        imdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    df_dir["director_list"] = [
        directors_from_omdb_json_or_cache(omdb_raw, imdb_id)
        for omdb_raw, imdb_id in zip(omdb_vals, imdb_vals)
    ]
    df_dir = df_dir.explode("director_list", ignore_index=True)
    df_dir = df_dir[df_dir["director_list"].notna() & (df_dir["director_list"] != "")]

    if df_dir.empty:
        return pd.DataFrame(
            columns=["director_list", "decision", "count", "imdb_mean", "count_total"]
        )

    stats_mean = (
        df_dir.groupby("director_list", dropna=False)
        .agg(imdb_mean=("imdb_rating", "mean"), count_total=("title", "count"))
        .reset_index()
    )
    counts = (
        df_dir.groupby(["director_list", "decision"], dropna=False)["title"]
        .count()
        .reset_index()
        .rename(columns={"title": "count"})
    )
    out = counts.merge(stats_mean, on="director_list", how="left")
    return out


@_cache_data_decorator()
def _word_counts(df: pd.DataFrame, decisions: tuple[str, ...]) -> pd.DataFrame:
    return build_word_counts(df, decisions)
