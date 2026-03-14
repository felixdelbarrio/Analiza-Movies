from __future__ import annotations

import os

from server.api.app import app as app


def main() -> None:
    """
    Compat layer:
    - Mantiene `server.am_api:app` como entrypoint de uvicorn.
    - Mantiene `main()` para console_scripts si lo usas así.
    """
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))

    # Default seguro: sin env var -> no reload
    reload = os.getenv("API_RELOAD", "0") == "1"

    uvicorn.run("server.am_api:app", host=host, port=port, reload=reload)
