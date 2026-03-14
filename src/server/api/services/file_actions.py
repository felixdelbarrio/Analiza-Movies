from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


Row = Mapping[str, Any]


def _safe_path_from_row(row: Row) -> Path | None:
    raw = row.get("file")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser()
    except Exception:
        return None


def _is_probably_safe_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


def delete_files_from_rows(
    rows: Iterable[Row],
    *,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    logs: list[str] = []
    ok = 0
    err = 0

    for index, row in enumerate(rows, start=1):
        path = _safe_path_from_row(row)
        title = str(row.get("title") or "").strip()
        if path is None:
            err += 1
            logs.append(f"[DELETE] row#{index}: sin 'file' valido (title={title!r})")
            continue

        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        if not _is_probably_safe_file(resolved):
            logs.append(
                f"[DELETE] row#{index}: skip (no existe/no es fichero): {resolved} (title={title!r})"
            )
            continue

        if dry_run:
            ok += 1
            logs.append(f"[DELETE][DRY] would delete: {resolved} (title={title!r})")
            continue

        try:
            os.remove(resolved)
            ok += 1
            logs.append(f"[DELETE] deleted: {resolved} (title={title!r})")
        except Exception as exc:
            err += 1
            logs.append(
                f"[DELETE] ERROR deleting {resolved} (title={title!r}): {exc!r}"
            )

    return ok, err, logs
