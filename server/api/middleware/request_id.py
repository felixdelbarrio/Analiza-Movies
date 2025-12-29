from __future__ import annotations

"""
server/api/middleware/request_id.py

Middleware:
- Inyecta/propaga X-Request-ID
- Registra métricas y log por request (duración + status)
"""

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from server.api.logging_config import configure_logging
from server.api.services import metrics
from server.api.settings import Settings


CallNext = Callable[[Request], Awaitable[Response]]
Middleware = Callable[[Request, CallNext], Awaitable[Response]]


def build_request_id_middleware(settings: Settings) -> Middleware:
    logger = configure_logging(settings)

    async def middleware(request: Request, call_next: CallNext) -> Response:
        start = time.monotonic()
        req_id = (request.headers.get("x-request-id") or "").strip() or uuid.uuid4().hex
        request.state.request_id = req_id

        metrics.inc("http_requests_total", 1)

        response: Response | None = None
        status_code: int = 500

        try:
            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 500))
            return response
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "request",
                extra={
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            # Asegura header incluso si la app devuelve Response pero luego crashea en logging
            if response is not None:
                try:
                    response.headers["X-Request-ID"] = req_id
                except Exception:
                    pass

    return middleware