from __future__ import annotations

import os

import uvicorn

from shared.runtime_profiles import bootstrap_runtime_config


def main() -> None:
    bootstrap_runtime_config()

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "0") == "1"

    uvicorn.run(
        "server.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
