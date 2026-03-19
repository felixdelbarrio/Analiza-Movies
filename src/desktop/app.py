from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from urllib.request import urlopen

import uvicorn

APP_TITLE = "Analiza Movies"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_WIDTH = 1560
DEFAULT_HEIGHT = 980


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, *, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=1.0) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            time.sleep(0.2)

    if last_exc is not None:
        raise RuntimeError(f"No se pudo iniciar el servidor embebido: {last_exc!r}")
    raise RuntimeError("No se pudo iniciar el servidor embebido.")


@dataclass(slots=True)
class _EmbeddedServer:
    host: str
    port: int
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None

    def start(self) -> str:
        from server.api.app import create_app

        config = uvicorn.Config(
            app=create_app(),
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        setattr(server, "install_signal_handlers", lambda: None)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        self.server = server
        self.thread = thread

        base_url = f"http://{self.host}:{self.port}"
        _wait_until_ready(f"{base_url}/health")
        return base_url

    def stop(self) -> None:
        if self.server is None:
            return
        self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5.0)


class DesktopApi:
    def __init__(self) -> None:
        self._child_titles: set[str] = set()

    def is_desktop(self) -> bool:
        return True

    def open_external_url(self, url: str) -> bool:
        clean_url = str(url or "").strip()
        if not clean_url:
            return False
        return bool(webbrowser.open(clean_url))

    def open_url(self, url: str, title: str | None = None) -> bool:
        clean_url = str(url or "").strip()
        if not clean_url:
            return False

        clean_title = (title or APP_TITLE).strip() or APP_TITLE

        import webview

        # Mantiene la navegación dentro de la aplicación en una ventana nativa,
        # nunca en una pestaña del navegador del sistema.
        unique_title = clean_title
        suffix = 2
        while unique_title in self._child_titles:
            unique_title = f"{clean_title} {suffix}"
            suffix += 1
        self._child_titles.add(unique_title)

        webview.create_window(
            unique_title,
            clean_url,
            width=1280,
            height=820,
            min_size=(960, 640),
        )
        return True


def main() -> None:
    os.environ.setdefault("API_RELOAD", "0")

    host = DEFAULT_HOST
    port = _find_free_port(host)
    server = _EmbeddedServer(host=host, port=port)
    base_url = server.start()

    import webview

    bridge = DesktopApi()

    webview.create_window(
        APP_TITLE,
        base_url,
        js_api=bridge,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        min_size=(1180, 760),
    )

    try:
        webview.start(debug=False)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
