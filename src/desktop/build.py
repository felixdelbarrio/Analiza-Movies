from __future__ import annotations

import argparse
import base64
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from PyInstaller.__main__ import run as pyinstaller_run

APP_NAME = "AnalizaMovies"
APP_DISPLAY_NAME = "Analiza Movies"
APP_COMMENT = "Desktop-first cinematic control room for Plex and DLNA libraries."
APP_ID = "com.analizamovies.desktop"
ICON_BASENAME = "media-library-analyzer"

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
DIST_ROOT = ROOT_DIR / "dist-desktop"
BUILD_ROOT = ROOT_DIR / "build" / "desktop"
ASSETS_DIR = ROOT_DIR / "assets"

WINDOWS_ICON = ASSETS_DIR / "windows" / f"{ICON_BASENAME}.ico"
MACOS_ICON = ASSETS_DIR / "macos" / f"{ICON_BASENAME}.icns"
LINUX_ICON = (
    ASSETS_DIR / "linux" / "hicolor" / "512x512" / "apps" / f"{ICON_BASENAME}.png"
)
LINUX_ICON_DIR = ASSETS_DIR / "linux" / "hicolor"
LINUX_SCALABLE_ICON = (
    ASSETS_DIR / "linux" / "scalable" / "apps" / f"{ICON_BASENAME}.svg"
)
LINUX_DESKTOP_TEMPLATE = ASSETS_DIR / "linux" / f"{ICON_BASENAME}.desktop.example"


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _platform_id() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    if system == "Linux":
        return "linux"
    raise RuntimeError(f"Plataforma no soportada: {system}")


def _data_sep() -> str:
    return ";" if platform.system() == "Windows" else ":"


def _icon_for_platform() -> Path:
    system = platform.system()
    if system == "Darwin":
        return MACOS_ICON
    if system == "Windows":
        return WINDOWS_ICON
    return LINUX_ICON


def _desktop_template_text() -> str:
    if not LINUX_DESKTOP_TEMPLATE.exists():
        raise FileNotFoundError(
            f"No se encontró la plantilla desktop: {LINUX_DESKTOP_TEMPLATE}"
        )
    return LINUX_DESKTOP_TEMPLATE.read_text(encoding="utf-8")


def _validate_branding_assets() -> None:
    required_paths = [
        WINDOWS_ICON,
        MACOS_ICON,
        LINUX_ICON,
        LINUX_SCALABLE_ICON,
        LINUX_DESKTOP_TEMPLATE,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        formatted = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(f"Faltan assets de branding requeridos:\n{formatted}")


def _ensure_frontend_build(skip_frontend: bool) -> None:
    if skip_frontend and WEB_DIST_DIR.exists():
        return
    _run(["npm", "ci"], cwd=WEB_DIR)
    _run(["npm", "run", "build"], cwd=WEB_DIR)


def _pyinstaller_args(dist_dir: Path, work_dir: Path) -> list[str]:
    args = [
        "--noconfirm",
        "--windowed",
        "--onedir",
        "--name",
        APP_NAME,
        "--icon",
        str(_icon_for_platform()),
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(BUILD_ROOT / "spec"),
        "--collect-submodules=uvicorn",
        "--collect-submodules=webview",
        "--collect-data=webview",
        f"--add-data={WEB_DIST_DIR}{_data_sep()}web/dist",
        # Bundle the package entrypoint so the frozen app always executes main().
        str(ROOT_DIR / "src" / "desktop" / "__main__.py"),
    ]

    if platform.system() == "Darwin":
        args.extend(["--osx-bundle-identifier", APP_ID])
    return args


def _mac_app_path(dist_dir: Path) -> Path:
    return dist_dir / f"{APP_NAME}.app"


def _linux_bundle_dir(dist_dir: Path) -> Path:
    return dist_dir / APP_NAME


def _bundle_marker_path(dist_dir: Path) -> Path:
    system = platform.system()
    if system == "Darwin":
        return _mac_app_path(dist_dir) / "Contents" / "MacOS" / APP_NAME
    if system == "Windows":
        return dist_dir / APP_NAME / f"{APP_NAME}.exe"
    return dist_dir / APP_NAME / APP_NAME


def _build_stamp_path(dist_dir: Path) -> Path:
    return dist_dir / ".build-stamp"


def _iter_project_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return [path for path in root.rglob("*") if path.is_file()]


def _latest_input_mtime() -> float:
    input_paths: list[Path] = [
        ROOT_DIR / "src",
        ROOT_DIR / "requirements.txt",
        ROOT_DIR / "requirements-dev.txt",
        ROOT_DIR / "setup.py",
        ASSETS_DIR,
        WEB_DIST_DIR,
    ]
    latest = 0.0
    for root in input_paths:
        for path in _iter_project_files(root):
            try:
                latest = max(latest, path.stat().st_mtime)
            except FileNotFoundError:
                continue
    return latest


def _is_build_current(dist_dir: Path) -> bool:
    stamp_path = _build_stamp_path(dist_dir)
    marker_path = _bundle_marker_path(dist_dir)
    if not stamp_path.exists() or not marker_path.exists():
        return False
    return stamp_path.stat().st_mtime >= _latest_input_mtime()


def _write_build_stamp(dist_dir: Path) -> None:
    stamp_path = _build_stamp_path(dist_dir)
    stamp_path.write_text(
        f"built_at={int(time.time())}\nplatform={_platform_id()}\n",
        encoding="utf-8",
    )


def _write_linux_desktop_file(bundle_dir: Path) -> None:
    desktop_dir = bundle_dir / "share" / "applications"
    desktop_dir.mkdir(parents=True, exist_ok=True)

    desktop_text = (
        _desktop_template_text()
        .replace("Name=Media Library Analyzer", f"Name={APP_DISPLAY_NAME}")
        .replace(
            "Comment=Analyze and explore your media library", f"Comment={APP_COMMENT}"
        )
        .replace("Exec=media-library-analyzer", f"Exec={APP_NAME}")
        .replace("Icon=media-library-analyzer", f"Icon={ICON_BASENAME}")
    )
    if "StartupWMClass=" not in desktop_text:
        desktop_text = f"{desktop_text.rstrip()}\nStartupWMClass={APP_NAME}\n"
    desktop_path = desktop_dir / "analiza-movies.desktop"
    desktop_path.write_text(desktop_text, encoding="utf-8")


def _copy_linux_icons(bundle_dir: Path) -> None:
    target_root = bundle_dir / "share" / "icons" / "hicolor"
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(LINUX_ICON_DIR, target_root)

    scalable_dir = target_root / "scalable" / "apps"
    scalable_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LINUX_SCALABLE_ICON, scalable_dir / LINUX_SCALABLE_ICON.name)


def _install_linux_branding_assets(dist_dir: Path) -> None:
    bundle_dir = _linux_bundle_dir(dist_dir)
    _copy_linux_icons(bundle_dir)
    _write_linux_desktop_file(bundle_dir)


def _maybe_codesign_macos(app_path: Path) -> None:
    cert_blob = os.getenv("APPLE_CERTIFICATE_P12_BASE64", "").strip()
    cert_password = os.getenv("APPLE_CERTIFICATE_PASSWORD", "").strip()
    sign_identity = os.getenv("APPLE_SIGN_IDENTITY", "").strip()

    if not cert_blob or not cert_password or not sign_identity:
        print("Apple signing skipped: missing certificate/signing env vars.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cert_path = tmp_path / "certificate.p12"
        cert_path.write_bytes(base64.b64decode(cert_blob))

        keychain_path = tmp_path / "build.keychain-db"
        keychain_password = ""

        _run(
            [
                "security",
                "create-keychain",
                "-p",
                keychain_password,
                str(keychain_path),
            ]
        )
        _run(
            [
                "security",
                "set-keychain-settings",
                "-lut",
                "21600",
                str(keychain_path),
            ]
        )
        _run(
            [
                "security",
                "unlock-keychain",
                "-p",
                keychain_password,
                str(keychain_path),
            ]
        )
        _run(
            [
                "security",
                "import",
                str(cert_path),
                "-k",
                str(keychain_path),
                "-P",
                cert_password,
                "-T",
                "/usr/bin/codesign",
                "-T",
                "/usr/bin/security",
            ]
        )
        _run(
            [
                "security",
                "list-keychains",
                "-d",
                "user",
                "-s",
                str(keychain_path),
                "login.keychain-db",
            ]
        )
        _run(
            [
                "codesign",
                "--force",
                "--deep",
                "--options",
                "runtime",
                "--sign",
                sign_identity,
                str(app_path),
            ]
        )

        apple_id = os.getenv("APPLE_NOTARY_APPLE_ID", "").strip()
        app_password = os.getenv("APPLE_NOTARY_APP_PASSWORD", "").strip()
        team_id = os.getenv("APPLE_NOTARY_TEAM_ID", "").strip()
        if apple_id and app_password and team_id:
            zip_path = app_path.parent / f"{APP_NAME}-macos-notarize.zip"
            _run(
                [
                    "ditto",
                    "-c",
                    "-k",
                    "--sequesterRsrc",
                    "--keepParent",
                    str(app_path),
                    str(zip_path),
                ]
            )
            _run(
                [
                    "xcrun",
                    "notarytool",
                    "submit",
                    str(zip_path),
                    "--apple-id",
                    apple_id,
                    "--password",
                    app_password,
                    "--team-id",
                    team_id,
                    "--wait",
                ]
            )
            _run(["xcrun", "stapler", "staple", str(app_path)])
        else:
            print("Apple notarization skipped: missing notary env vars.")


def _archive_output(platform_id: str, dist_dir: Path) -> Path:
    artifact_base = DIST_ROOT / f"{APP_NAME}-{platform_id}"

    if platform.system() == "Darwin":
        app_path = _mac_app_path(dist_dir)
        archive_path = artifact_base.with_suffix(".zip")
        if archive_path.exists():
            archive_path.unlink()
        _run(
            [
                "ditto",
                "-c",
                "-k",
                "--sequesterRsrc",
                "--keepParent",
                str(app_path),
                str(archive_path),
            ]
        )
        return archive_path

    bundle_dir = dist_dir / APP_NAME
    if platform.system() == "Windows":
        archive_path = Path(
            shutil.make_archive(
                str(artifact_base), "zip", root_dir=dist_dir, base_dir=APP_NAME
            )
        )
        return archive_path

    archive_path = artifact_base.with_suffix(".tar.gz")
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=APP_NAME)
    return archive_path


def _launch_output(dist_dir: Path) -> None:
    if platform.system() == "Darwin":
        _run(["open", "-W", str(_mac_app_path(dist_dir))])
        return

    if platform.system() == "Windows":
        executable = dist_dir / APP_NAME / f"{APP_NAME}.exe"
        _run([str(executable)], cwd=executable.parent)
        return

    executable = dist_dir / APP_NAME / APP_NAME
    share_dir = dist_dir / APP_NAME / "share"
    env = dict(os.environ)
    env["XDG_DATA_DIRS"] = (
        str(share_dir)
        if not env.get("XDG_DATA_DIRS")
        else f"{share_dir}{os.pathsep}{env['XDG_DATA_DIRS']}"
    )
    _run([str(executable)], cwd=executable.parent, env=env)


def build_desktop(
    *,
    skip_frontend: bool,
    archive: bool,
    reuse_existing: bool,
    quiet: bool,
) -> tuple[Path, Path | None, bool]:
    platform_id = _platform_id()
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    _validate_branding_assets()
    _ensure_frontend_build(skip_frontend=skip_frontend)

    dist_dir = DIST_ROOT / platform_id
    work_dir = BUILD_ROOT / platform_id
    if reuse_existing and _is_build_current(dist_dir):
        archive_path = _archive_output(platform_id, dist_dir) if archive else None
        return dist_dir, archive_path, True

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pyinstaller_args = _pyinstaller_args(dist_dir=dist_dir, work_dir=work_dir)
    if quiet:
        pyinstaller_args.extend(["--log-level", "ERROR"])

    pyinstaller_run(pyinstaller_args)

    if platform.system() == "Linux":
        _install_linux_branding_assets(dist_dir)
    if platform.system() == "Darwin":
        _maybe_codesign_macos(_mac_app_path(dist_dir))

    _write_build_stamp(dist_dir)

    archive_path = _archive_output(platform_id, dist_dir) if archive else None
    return dist_dir, archive_path, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local desktop bundle.")
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="No recompila React si web/dist ya existe.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Lanza la app nativa empaquetada al terminar la build.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="No genera zip/tar.gz al terminar la build.",
    )
    parser.add_argument(
        "--validate-assets",
        action="store_true",
        help="Verifica que todos los iconos y plantillas de branding existen y termina.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reutiliza el bundle existente si los inputs no han cambiado.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce el ruido de PyInstaller y deja solo el resumen final.",
    )
    args = parser.parse_args()

    if args.validate_assets:
        _validate_branding_assets()
        print("Desktop branding assets ready.")
        return

    dist_dir, archive_path, reused_existing = build_desktop(
        skip_frontend=args.skip_frontend,
        archive=not args.no_archive,
        reuse_existing=args.reuse_existing,
        quiet=args.quiet,
    )

    if archive_path is not None and reused_existing:
        print(f"Desktop build already up to date: {archive_path}")
    elif archive_path is not None:
        print(f"Desktop build ready: {archive_path}")
    elif reused_existing:
        print(f"Desktop bundle already up to date: {dist_dir}")
    else:
        print(f"Desktop bundle ready: {dist_dir}")

    if args.run:
        _launch_output(dist_dir)


if __name__ == "__main__":
    main()
