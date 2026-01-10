import asyncio
import json

from fastapi import Request, Response

from server.api.middleware.errors import build_exception_handler
from server.api.middleware.request_id import build_request_id_middleware
from server.api.settings import Settings


def _make_request(headers):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": "GET",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    return Request(scope, receive)


def _settings():
    return Settings(
        log_level="INFO",
        cors_origins_raw="*",
        cors_allow_credentials=False,
        gzip_min_size=0,
        file_cache_max_entries=1,
        file_cache_ttl_seconds=0.0,
        file_read_max_attempts=1,
        file_read_retry_sleep_s=0.0,
    )


def test_request_id_middleware_sets_header(monkeypatch):
    from server.api.middleware import request_id as mod

    captured = {"info": None, "metrics": []}

    class DummyLogger:
        def info(self, msg, extra=None):
            captured["info"] = (msg, extra)

    monkeypatch.setattr(mod, "configure_logging", lambda settings: DummyLogger())
    monkeypatch.setattr(mod.metrics, "inc", lambda name, value=1: captured["metrics"].append((name, value)))

    middleware = build_request_id_middleware(_settings())

    request = _make_request([(b"x-request-id", b"req-123")])

    async def call_next(req):
        assert req.state.request_id == "req-123"
        return Response(status_code=201)

    response = asyncio.run(middleware(request, call_next))

    assert response.status_code == 201
    assert response.headers["X-Request-ID"] == "req-123"
    assert ("http_requests_total", 1) in captured["metrics"]
    assert captured["info"] is not None
    assert captured["info"][1]["status"] == 201


def test_exception_handler_includes_request_id(monkeypatch):
    from server.api.middleware import errors as mod

    captured = {"exc": None, "metrics": []}

    class DummyLogger:
        def exception(self, msg, extra=None):
            captured["exc"] = (msg, extra)

    monkeypatch.setattr(mod, "configure_logging", lambda settings: DummyLogger())
    monkeypatch.setattr(mod.metrics, "inc", lambda name, value=1: captured["metrics"].append((name, value)))

    handler = build_exception_handler(_settings())

    request = _make_request([])
    request.state.request_id = "req-xyz"

    response = asyncio.run(handler(request, RuntimeError("boom")))

    payload = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 500
    assert payload["detail"] == "Internal Server Error"
    assert payload["request_id"] == "req-xyz"
    assert isinstance(payload["error_id"], str) and payload["error_id"]
    assert ("http_errors_5xx_total", 1) in captured["metrics"]
    assert captured["exc"] is not None
