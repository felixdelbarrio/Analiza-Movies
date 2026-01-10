import json

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from server.api.app import create_app
from server.api.caching.file_cache import FileCache
from server.api.settings import Settings
import server.api.deps as deps
import server.api.paths as paths
import server.api.routers.health as health_router
import server.api.services.omdb as omdb_service
import server.api.services.wiki as wiki_service


def _settings() -> Settings:
    return Settings(
        log_level="INFO",
        cors_origins_raw="*",
        cors_allow_credentials=False,
        gzip_min_size=0,
        file_cache_max_entries=4,
        file_cache_ttl_seconds=0.0,
        file_read_max_attempts=1,
        file_read_retry_sleep_s=0.0,
    )


def _patch_paths(tmp_path, monkeypatch):
    omdb_path = tmp_path / "omdb_cache.json"
    wiki_path = tmp_path / "wiki_cache.json"
    report_all = tmp_path / "report_all.csv"
    metadata_fix = tmp_path / "metadata_fix.csv"

    omdb_path.write_text(json.dumps({"tt123": {"title": "Movie"}}), encoding="utf-8")
    wiki_path.write_text(json.dumps({"tt123": {"title": "Movie"}}), encoding="utf-8")
    report_all.write_text("title\nMovie\n", encoding="utf-8")
    metadata_fix.write_text("title\nMovie\n", encoding="utf-8")

    monkeypatch.setattr(paths, "OMDB_CACHE_PATH", omdb_path)
    monkeypatch.setattr(paths, "WIKI_CACHE_PATH", wiki_path)
    monkeypatch.setattr(paths, "REPORT_ALL_PATH", report_all)
    monkeypatch.setattr(paths, "REPORT_FILTERED_PATH", tmp_path / "report_filtered.csv")
    monkeypatch.setattr(paths, "METADATA_FIX_PATH", metadata_fix)

    monkeypatch.setattr(health_router, "OMDB_CACHE_PATH", omdb_path)
    monkeypatch.setattr(health_router, "WIKI_CACHE_PATH", wiki_path)
    monkeypatch.setattr(health_router, "REPORT_ALL_PATH", report_all)
    monkeypatch.setattr(health_router, "REPORT_FILTERED_PATH", tmp_path / "report_filtered.csv")
    monkeypatch.setattr(health_router, "METADATA_FIX_PATH", metadata_fix)

    monkeypatch.setattr(omdb_service, "OMDB_CACHE_PATH", omdb_path)
    monkeypatch.setattr(wiki_service, "WIKI_CACHE_PATH", wiki_path)


def test_health_ready_and_metrics(monkeypatch, tmp_path):
    _patch_paths(tmp_path, monkeypatch)

    app = create_app()
    cache = FileCache(_settings())
    app.dependency_overrides[deps.get_file_cache] = lambda: cache

    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "text/plain" in metrics.headers.get("content-type", "")
