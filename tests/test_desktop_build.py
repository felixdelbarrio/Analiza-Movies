from __future__ import annotations

from pathlib import Path

from desktop.__main__ import _run_mode
from desktop.build import ROOT_DIR, _pyinstaller_args


def test_pyinstaller_uses_package_entrypoint_script(tmp_path: Path) -> None:
    args = _pyinstaller_args(
        dist_dir=tmp_path / "dist-desktop",
        work_dir=tmp_path / "build-desktop",
    )

    entrypoint = next(Path(arg) for arg in args if arg.endswith("__main__.py"))

    assert entrypoint == ROOT_DIR / "src" / "desktop" / "__main__.py"


def test_desktop_entrypoint_routes_background_flags_to_backend() -> None:
    assert _run_mode(("AnalizaMovies", "--plex")) == "backend"
    assert _run_mode(("AnalizaMovies", "--dlna")) == "backend"
    assert _run_mode(("AnalizaMovies",)) == "desktop"
