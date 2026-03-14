from __future__ import annotations

import argparse
import base64
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from PyInstaller.__main__ import run as pyinstaller_run

APP_NAME = "AnalizaMovies"
ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
DIST_ROOT = ROOT_DIR / "dist-desktop"
BUILD_ROOT = ROOT_DIR / "build" / "desktop"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


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


def _ensure_frontend_build(skip_frontend: bool) -> None:
    if skip_frontend and WEB_DIST_DIR.exists():
        return
    _run(["npm", "ci"], cwd=WEB_DIR)
    _run(["npm", "run", "build"], cwd=WEB_DIR)


def _pyinstaller_args(dist_dir: Path, work_dir: Path) -> list[str]:
    args = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name",
        APP_NAME,
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
        str(ROOT_DIR / "src" / "desktop" / "app.py"),
    ]

    if platform.system() == "Darwin":
        args.extend(["--osx-bundle-identifier", "com.analizamovies.desktop"])
    return args


def _mac_app_path(dist_dir: Path) -> Path:
    return dist_dir / f"{APP_NAME}.app"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local desktop bundle.")
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="No recompila React si web/dist ya existe.",
    )
    args = parser.parse_args()

    platform_id = _platform_id()
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    _ensure_frontend_build(skip_frontend=args.skip_frontend)

    dist_dir = DIST_ROOT / platform_id
    work_dir = BUILD_ROOT / platform_id
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)

    pyinstaller_run(_pyinstaller_args(dist_dir=dist_dir, work_dir=work_dir))

    if platform.system() == "Darwin":
        _maybe_codesign_macos(_mac_app_path(dist_dir))

    archive_path = _archive_output(platform_id, dist_dir)
    print(f"Desktop build ready: {archive_path}")


if __name__ == "__main__":
    main()
