from __future__ import annotations

from pathlib import Path

from desktop.build import ROOT_DIR, _pyinstaller_args


def test_pyinstaller_uses_package_entrypoint_script(tmp_path: Path) -> None:
    args = _pyinstaller_args(
        dist_dir=tmp_path / "dist-desktop",
        work_dir=tmp_path / "build-desktop",
    )

    entrypoint = next(Path(arg) for arg in args if arg.endswith("__main__.py"))

    assert entrypoint == ROOT_DIR / "src" / "desktop" / "__main__.py"
