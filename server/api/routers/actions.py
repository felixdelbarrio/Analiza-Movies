from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from server.api.services.file_actions import delete_files_from_rows

router = APIRouter(prefix="/actions", tags=["actions"])


@router.post("/delete")
def delete_action(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    rows_obj = payload.get("rows")
    if not isinstance(rows_obj, list):
        raise HTTPException(status_code=400, detail="'rows' debe ser una lista")

    rows: list[dict[str, Any]] = []
    for item in rows_obj:
        if isinstance(item, dict):
            rows.append({str(k): v for k, v in item.items()})

    dry_run = bool(payload.get("dry_run", True))
    ok, err, logs = delete_files_from_rows(rows, dry_run=dry_run)
    return {
        "ok": ok,
        "err": err,
        "dry_run": dry_run,
        "logs": logs,
    }
