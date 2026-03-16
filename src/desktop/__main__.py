from __future__ import annotations

import sys
from collections.abc import Sequence


def _run_mode(argv: Sequence[str]) -> str:
    flags = set(argv[1:])
    if "--plex" in flags or "--dlna" in flags:
        return "backend"
    return "desktop"


def main(argv: Sequence[str] | None = None) -> None:
    args = tuple(sys.argv if argv is None else argv)
    if _run_mode(args) == "backend":
        from backend.main import start as start_backend

        start_backend()
        return

    from desktop.app import main as start_desktop

    start_desktop()


if __name__ == "__main__":
    main()
