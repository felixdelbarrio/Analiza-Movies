from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from server.api.deps import get_settings
from server.api.middleware import build_exception_handler, build_request_id_middleware
from server.api.routers.consolidated import router as consolidated_router
from server.api.routers.health import router as health_router
from server.api.routers.meta import router as meta_router
from server.api.routers.omdb import router as omdb_router
from server.api.routers.reports import router as reports_router
from server.api.routers.wiki import router as wiki_router

_settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(title="Analiza Movies Public API", version="1.0.0")

    app.add_middleware(GZipMiddleware, minimum_size=max(0, _settings.gzip_min_size))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_allow_origins(),
        allow_credentials=_settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.middleware("http")(build_request_id_middleware(_settings))
    app.add_exception_handler(Exception, build_exception_handler(_settings))

    app.include_router(health_router)
    app.include_router(meta_router)
    app.include_router(reports_router)
    app.include_router(omdb_router)
    app.include_router(wiki_router)
    app.include_router(consolidated_router)

    return app


app = create_app()
