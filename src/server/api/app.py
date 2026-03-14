from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request
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


def _project_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "setup.py").exists() and (candidate / "web").exists():
            return candidate
    return current.parents[3]


_PROJECT_ROOT = _project_root()
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

_PLACEHOLDER_LOCALES = {
    "en": {
        "lang": "en",
        "title": "React frontend not built",
        "body": "The API is running, but the SPA is not available yet.",
        "hint": "Build the bundle with {build} or use development mode with {dev}.",
    },
    "es": {
        "lang": "es",
        "title": "Frontend React no compilado",
        "body": "La API está activa, pero la SPA no está disponible todavía.",
        "hint": "Genera el bundle con {build} o usa desarrollo con {dev}.",
    },
    "fr": {
        "lang": "fr",
        "title": "Frontend React non compilé",
        "body": "L'API est active, mais la SPA n'est pas encore disponible.",
        "hint": "Générez le bundle avec {build} ou utilisez le mode développement avec {dev}.",
    },
    "de": {
        "lang": "de",
        "title": "React-Frontend nicht gebaut",
        "body": "Die API ist aktiv, aber die SPA ist noch nicht verfügbar.",
        "hint": "Erstelle das Bundle mit {build} oder nutze den Entwicklungsmodus mit {dev}.",
    },
    "it": {
        "lang": "it",
        "title": "Frontend React non compilato",
        "body": "L'API è attiva, ma la SPA non è ancora disponibile.",
        "hint": "Genera il bundle con {build} oppure usa la modalità sviluppo con {dev}.",
    },
    "pt": {
        "lang": "pt",
        "title": "Frontend React não compilado",
        "body": "A API está ativa, mas a SPA ainda não está disponível.",
        "hint": "Gera o bundle com {build} ou usa o modo de desenvolvimento com {dev}.",
    },
}


def _preferred_locale(request: Request | None) -> dict[str, str]:
    header = "" if request is None else request.headers.get("accept-language", "")
    for raw_chunk in header.split(","):
        chunk = raw_chunk.split(";", 1)[0].strip().lower()
        if not chunk:
            continue
        prefix = chunk.split("-", 1)[0]
        if prefix in _PLACEHOLDER_LOCALES:
            return _PLACEHOLDER_LOCALES[prefix]
    return _PLACEHOLDER_LOCALES["es"]


def _frontend_placeholder(request: Request | None = None) -> HTMLResponse:
    copy = _preferred_locale(request)
    hint = copy["hint"].format(
        build="<code>make frontend-build</code>",
        dev="<code>make frontend</code>",
    )
    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="{copy['lang']}">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Analiza Movies</title>
            <style>
              body {{
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                padding: 24px;
                background: #09111d;
                color: #edf2f7;
                font-family: ui-sans-serif, system-ui, sans-serif;
              }}
              main {{
                max-width: 640px;
                padding: 28px 32px;
                border-radius: 24px;
                border: 1px solid rgba(151, 177, 214, 0.16);
                background: rgba(11, 20, 34, 0.84);
              }}
              code {{
                padding: 2px 6px;
                border-radius: 999px;
                background: rgba(113, 213, 255, 0.12);
              }}
            </style>
          </head>
          <body>
            <main>
              <h1>{copy['title']}</h1>
              <p>{copy['body']}</p>
              <p>{hint}</p>
            </main>
          </body>
        </html>
        """,
        status_code=503,
    )


def _is_reserved_spa_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _SPA_RESERVED_PREFIXES
    )


def _safe_spa_asset_path(full_path: str) -> Path | None:
    dist_root = _WEB_DIST_DIR.resolve()
    try:
        relative = PurePosixPath(full_path.strip("/"))
    except Exception:
        return None

    if relative.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None

    candidate = dist_root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(dist_root)
    except ValueError:
        return None
    return candidate


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
    def spa_index(request: Request) -> Response:
        if _WEB_INDEX_PATH.exists():
            return FileResponse(_WEB_INDEX_PATH)
        return _frontend_placeholder(request)

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    def spa_routes(full_path: str, request: Request) -> Response:
        path = f"/{full_path.strip('/')}"
        if _is_reserved_spa_path(path):
            raise HTTPException(status_code=404, detail="Not Found")

        candidate = _safe_spa_asset_path(full_path)
        if _WEB_DIST_DIR.exists() and candidate is not None and candidate.is_file():
            return FileResponse(candidate)

        if _WEB_INDEX_PATH.exists():
            return FileResponse(_WEB_INDEX_PATH)
        return _frontend_placeholder(request)

    return app


app = create_app()
