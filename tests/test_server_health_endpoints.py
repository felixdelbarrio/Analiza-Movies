import json
import importlib

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
from shared.runtime_profiles import ArtifactPaths

api_app_module = importlib.import_module("server.api.app")


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

    bundle = ArtifactPaths(
        profile_id="plex-test",
        data_dir=tmp_path,
        reports_dir=tmp_path,
        omdb_cache_path=omdb_path,
        wiki_cache_path=wiki_path,
        report_all_path=report_all,
        report_filtered_path=tmp_path / "report_filtered.csv",
        metadata_fix_path=metadata_fix,
    )

    monkeypatch.setattr(paths, "get_artifact_paths", lambda profile_id=None: bundle)
    monkeypatch.setattr(
        health_router, "get_artifact_paths", lambda profile_id=None: bundle
    )
    monkeypatch.setattr(
        omdb_service, "get_omdb_cache_path", lambda profile_id=None: omdb_path
    )
    monkeypatch.setattr(
        wiki_service, "get_wiki_cache_path", lambda profile_id=None: wiki_path
    )


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


def test_root_returns_placeholder_when_frontend_bundle_is_missing(
    monkeypatch, tmp_path
):
    missing_dist = tmp_path / "web-dist-missing"
    monkeypatch.setattr(api_app_module, "_WEB_DIST_DIR", missing_dist)
    monkeypatch.setattr(api_app_module, "_WEB_INDEX_PATH", missing_dist / "index.html")

    client = TestClient(api_app_module.create_app())
    response = client.get("/")

    assert response.status_code == 503
    assert "Frontend React no compilado" in response.text
