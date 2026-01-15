import json
from datetime import datetime, timezone

from fastapi import Request, Response

from server.api.caching import http_cache


def _make_request(headers):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
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


def test_stat_or_none_missing(tmp_path):
    missing = tmp_path / "missing.txt"
    assert http_cache.stat_or_none(missing) is None


def test_maybe_not_modified_etag(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    stat = path.stat()

    etag = http_cache._etag_from_stat(stat)
    request = _make_request([(b"if-none-match", etag.encode("utf-8"))])
    response = Response()

    assert (
        http_cache.maybe_not_modified(request=request, response=response, stat=stat)
        is True
    )
    assert response.status_code == 304
    assert response.headers["ETag"] == etag


def test_maybe_not_modified_if_modified_since(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    stat = path.stat()

    future_dt = datetime.fromtimestamp(stat.st_mtime + 10, tz=timezone.utc)
    ims = future_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")

    request = _make_request([(b"if-modified-since", ims.encode("utf-8"))])
    response = Response()

    assert (
        http_cache.maybe_not_modified(request=request, response=response, stat=stat)
        is True
    )
    assert response.status_code == 304


def test_maybe_not_modified_sets_headers_when_fresh(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    stat = path.stat()

    request = _make_request([])
    response = Response()

    assert (
        http_cache.maybe_not_modified(request=request, response=response, stat=stat)
        is False
    )
    assert response.headers["ETag"].startswith('W/"')
    assert "Last-Modified" in response.headers
