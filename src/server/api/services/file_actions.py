from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from server.api.paths import get_report_all_path, get_report_filtered_path

Row = Mapping[str, Any]

logger = logging.getLogger(__name__)


def _requested_file_key(raw: object) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    return text


def _authorized_file_entry(raw: object) -> tuple[str, Path] | None:
    text = _requested_file_key(raw)
    if text is None:
        return None
    try:
        return text, Path(text).expanduser().resolve()
    except Exception:
        return None


def _is_probably_safe_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


def _load_authorized_files(profile_id: str | None) -> dict[str, Path]:
    authorized: dict[str, Path] = {}

    for report_path in (
        get_report_filtered_path(profile_id),
        get_report_all_path(profile_id),
    ):
        try:
            if not report_path.exists() or not report_path.is_file():
                continue
            with report_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    entry = _authorized_file_entry((row or {}).get("file"))
                    if entry is None:
                        continue
                    raw_key, authorized_path = entry
                    authorized[raw_key] = authorized_path
        except Exception:
            logger.exception(
                "No se pudo cargar el listado de ficheros autorizados desde %s",
                report_path,
            )

    return authorized


def _display_target(title: str, path: Path | str) -> str:
    if isinstance(path, Path):
        file_name = path.name
    else:
        file_name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if title:
        return f"title={title!r}, file={file_name!r}"
    return f"file={file_name!r}"


def delete_files_from_rows(
    rows: Iterable[Row],
    *,
    dry_run: bool,
    profile_id: str | None,
) -> tuple[int, int, list[str]]:
    logs: list[str] = []
    ok = 0
    err = 0
    authorized_files = _load_authorized_files(profile_id)

    for index, row in enumerate(rows, start=1):
        requested_key = _requested_file_key(row.get("file"))
        title = str(row.get("title") or "").strip()
        if requested_key is None:
            err += 1
            logs.append(f"[DELETE] row#{index}: sin 'file' valido (title={title!r})")
            continue

        authorized_path = authorized_files.get(requested_key)
        if authorized_path is None:
            err += 1
            logs.append(
                f"[DELETE] row#{index}: ruta no autorizada para borrado ({_display_target(title, requested_key)})"
            )
            continue

        if not _is_probably_safe_file(authorized_path):
            logs.append(
                f"[DELETE] row#{index}: skip (no existe/no es fichero): {_display_target(title, authorized_path)}"
            )
            continue

        if dry_run:
            ok += 1
            logs.append(
                f"[DELETE][DRY] approved: {_display_target(title, authorized_path)}"
            )
            continue

        try:
            os.remove(authorized_path)
            ok += 1
            logs.append(f"[DELETE] deleted: {_display_target(title, authorized_path)}")
        except Exception:
            err += 1
            logger.exception(
                "No se pudo borrar el fichero autorizado %s", authorized_path
            )
            logs.append(
                f"[DELETE] ERROR deleting approved file ({_display_target(title, authorized_path)})"
            )

    return ok, err, logs
