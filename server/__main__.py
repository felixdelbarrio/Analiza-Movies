import os
import uvicorn


def main():
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "1") == "1"

    uvicorn.run(
        "server.api_fastapi:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()