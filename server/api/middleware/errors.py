# exception handlers (error_id)
from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from server.api.logging_config import configure_logging
from server.api.services import metrics
from server.api.settings import Settings


def build_exception_handler(settings: Settings):
    logger = configure_logging(settings)

    async def handler(request: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex
        req_id = getattr(request.state, "request_id", None)

        logger.exception(
            "unhandled_exception",
            extra={"error_id": error_id, "request_id": req_id, "path": request.url.path},
        )
        metrics.inc("http_errors_5xx_total", 1)

        payload: dict[str, Any] = {"detail": "Internal Server Error", "error_id": error_id}
        if isinstance(req_id, str) and req_id:
            payload["request_id"] = req_id

        return JSONResponse(status_code=500, content=payload)

    return handler