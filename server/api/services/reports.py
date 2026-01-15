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


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        return value

    return value


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
                    blob_mask = (
                        view[_INTERNAL_SEARCH_COL]
                        .astype("string")
                        .fillna("")
                        .str.contains(q)
                    )
                    view = view[blob_mask]
                except Exception:
                    pass

            if _INTERNAL_SEARCH_COL not in view.columns:
                candidate_cols = [c for c in _SEARCH_COLUMNS if c in view.columns]
                if candidate_cols:
                    cols_mask: pd.Series | None = None
                    for c in candidate_cols:
                        s = (
                            view[c]
                            .astype("string")
                            .fillna("")
                            .str.lower()
                            .str.contains(q)
                        )
                        cols_mask = s if cols_mask is None else (cols_mask | s)

                    if cols_mask is not None:
                        view = view[cols_mask]

    total = int(len(view))

    page = cast(pd.DataFrame, view.iloc[offset : offset + limit])
    page = strip_internal_cols(page)

    raw_items = page.to_dict(orient="records")
    records = cast(list[dict[str, Any]], raw_items)

    items: list[dict[str, Any]] = [
        {k: _json_safe_value(v) for k, v in row.items()} for row in records
    ]

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
