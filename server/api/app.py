from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from server.api.deps import get_settings
from server.api.middleware import build_exception_handler, build_request_id_middleware
from server.api.routers.actions import router as actions_router
from server.api.routers.consolidated import router as consolidated_router
from server.api.routers.configuration import router as configuration_router
from server.api.routers.health import router as health_router
from server.api.routers.meta import router as meta_router
from server.api.routers.omdb import router as omdb_router
from server.api.routers.reports import router as reports_router
from server.api.routers.wiki import router as wiki_router

_settings = get_settings()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WEB_DIST_DIR = _PROJECT_ROOT / "web" / "dist"
_WEB_INDEX_PATH = _WEB_DIST_DIR / "index.html"
_SPA_RESERVED_PREFIXES = (
    "/actions",
    "/cache",
    "/config",
    "/docs",
    "/health",
    "/meta",
    "/metrics",
    "/openapi.json",
    "/ready",
    "/redoc",
    "/reports",
)


def _frontend_placeholder() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="es">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Analiza Movies</title>
            <style>
              body {
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                padding: 24px;
                background: #09111d;
                color: #edf2f7;
                font-family: ui-sans-serif, system-ui, sans-serif;
              }
              main {
                max-width: 640px;
                padding: 28px 32px;
                border-radius: 24px;
                border: 1px solid rgba(151, 177, 214, 0.16);
                background: rgba(11, 20, 34, 0.84);
              }
              code {
                padding: 2px 6px;
                border-radius: 999px;
                background: rgba(113, 213, 255, 0.12);
              }
            </style>
          </head>
          <body>
            <main>
              <h1>Frontend React no compilado</h1>
              <p>La API está activa, pero la SPA no está disponible todavía.</p>
              <p>Genera el bundle con <code>make frontend-build</code> o usa desarrollo con <code>make frontend</code>.</p>
            </main>
          </body>
        </html>
        """,
        status_code=503,
    )


def _is_reserved_spa_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in _SPA_RESERVED_PREFIXES)


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
    app.include_router(actions_router)
    app.include_router(configuration_router)
    app.include_router(meta_router)
    app.include_router(reports_router)
    app.include_router(omdb_router)
    app.include_router(wiki_router)
    app.include_router(consolidated_router)

    assets_dir = _WEB_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="web-assets")

    @app.get("/", include_in_schema=False, response_model=None)
    def spa_index() -> Response:
        if _WEB_INDEX_PATH.exists():
            return FileResponse(_WEB_INDEX_PATH)
        return _frontend_placeholder()

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    def spa_routes(full_path: str) -> Response:
        path = f"/{full_path.strip('/')}"
        if _is_reserved_spa_path(path):
            raise HTTPException(status_code=404, detail="Not Found")

        candidate = (_WEB_DIST_DIR / full_path).resolve()
        if _WEB_DIST_DIR.exists() and candidate.is_file() and _WEB_DIST_DIR in candidate.parents:
            return FileResponse(candidate)

        if _WEB_INDEX_PATH.exists():
            return FileResponse(_WEB_INDEX_PATH)
        return _frontend_placeholder()

    return app


app = create_app()
