from __future__ import annotations

from typing import Any, cast

import pandas as pd

_SEARCH_COLUMNS = ["title", "name", "file", "path", "imdb_id", "imdbID"]
_INTERNAL_SEARCH_COL = "__search_blob"


def prepare_search_blob(df: pd.DataFrame) -> None:
    cols = [c for c in _SEARCH_COLUMNS if c in df.columns]
    if not cols:
        return

    try:
        parts: list[pd.Series] = []
        for c in cols:
            parts.append(df[c].astype("string").fillna("").str.lower())

        df[_INTERNAL_SEARCH_COL] = parts[0]
        for s in parts[1:]:
            df[_INTERNAL_SEARCH_COL] = df[_INTERNAL_SEARCH_COL] + " | " + s
    except Exception:
        return


def strip_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("__")]
    if not internal_cols:
        return df
    return df.drop(columns=internal_cols, errors="ignore")


def df_to_page(
    df: pd.DataFrame,
    *,
    offset: int,
    limit: int,
    query: str | None,
) -> dict[str, Any]:
    view = df

    if query:
        q = query.strip().lower()
        if q:
            if _INTERNAL_SEARCH_COL in view.columns:
                try:
                    mask = (
                        view[_INTERNAL_SEARCH_COL]
                        .astype("string")
                        .fillna("")
                        .str.contains(q)
                    )
                    view = view[mask]
                except Exception:
                    pass

            if _INTERNAL_SEARCH_COL not in view.columns:
                candidate_cols = [c for c in _SEARCH_COLUMNS if c in view.columns]
                if candidate_cols:
                    mask: pd.Series | None = None
                    for c in candidate_cols:
                        s = (
                            view[c]
                            .astype("string")
                            .fillna("")
                            .str.lower()
                            .str.contains(q)
                        )
                        mask = s if mask is None else (mask | s)

                    if mask is not None:
                        view = view[mask]

    total = int(len(view))

    page = cast(pd.DataFrame, view.iloc[offset : offset + limit])
    page = strip_internal_cols(page)

    items = page.where(pd.notnull(page), None).to_dict(orient="records")
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }