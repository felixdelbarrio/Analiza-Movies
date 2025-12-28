# middleware request_id + logging request
from __future__ import annotations

import time
import uuid
from typing import Any, Awaitable, Callable

from fastapi import Request, Response

from server.api.logging_config import configure_logging
from server.api.services import metrics
from server.api.settings import Settings


def build_request_id_middleware(settings: Settings) -> Callable[[Request, Callable[..., Awaitable[Response]]], Awaitable[Response]]:
    logger = configure_logging(settings)

    async def middleware(request: Request, call_next: Any) -> Response:
        start = time.monotonic()
        req_id = (request.headers.get("x-request-id") or "").strip() or uuid.uuid4().hex
        request.state.request_id = req_id

        metrics.inc("http_requests_total", 1)

        try:
            response = await call_next(request)
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            status = getattr(locals().get("response"), "status_code", 500)
            logger.info(
                "request",
                extra={
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )

        response.headers["X-Request-ID"] = req_id
        return response

    return middleware