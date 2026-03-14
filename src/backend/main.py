"""
backend/main.py

Punto de entrada unificado (CLI) para análisis de películas.

Este módulo es UI/CLI puro:
- Presenta menú origen (Plex/DLNA).
- Invoca orquestadores (Plex/DLNA).

Post-split DLNA (dlna_client.py)
--------------------------------
La interacción de DLNA (discover + selección de servidor + navegación) se realiza
dentro de backend.analiza_dlna.analyze_dlna_server() a través de DLNAClient.

Por ello este módulo NO debe duplicar discovery/selección DLNA para evitar:
- doble UX (dos menús)
- divergencia de comportamiento
- bugs por cambios en un sitio y no en otro

Reglas de consola (alineado con backend/logger.py)
-------------------------------------------------
- Menús y prompts: SIEMPRE visibles -> logger.info(..., always=True)
- Estado global (inicio / modo / fin): logger.progress(...)
- Debug contextual: logger.debug_ctx("ANALYZE", "...") (respeta DEBUG/SILENT)
- Salidas por cancelación/CTRL+C: limpias, sin stacktrace.
"""

from __future__ import annotations

import argparse
from typing import Literal

from backend import logger as logger
from backend.analiza_dlna import analyze_dlna_server
from backend.analiza_plex import analyze_all_libraries
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.dlna_discovery import DLNADevice
from shared.run_progress import (
    bind_progress_from_env,
    finish_run_progress,
    update_run_progress,
)

Choice = Literal["1", "2"]


def _parse_args() -> argparse.Namespace:
    """
    Flags opcionales para automatizar ejecución (sin menú).

    - Sin flags: UX actual por menú.
    - Con flags: ejecución directa.
    """
    parser = argparse.ArgumentParser(
        prog="start",
        add_help=True,
        description="Analiza Movies - CLI backend de ingesta (Plex/DLNA)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plex", action="store_true", help="Analizar Plex directamente (sin menú)"
    )
    mode.add_argument(
        "--dlna", action="store_true", help="Analizar DLNA directamente (sin menú)"
    )

    parser.add_argument("--dlna-host", help="Host del servidor DLNA a analizar")
    parser.add_argument("--dlna-port", type=int, help="Puerto del servidor DLNA")
    parser.add_argument("--dlna-location", help="LOCATION XML del servidor DLNA")
    parser.add_argument(
        "--dlna-friendly-name", help="Nombre legible del servidor DLNA seleccionado"
    )
    parser.add_argument("--dlna-device-id", help="Device ID del servidor DLNA")
    parser.add_argument(
        "--dlna-auto-select-all",
        action="store_true",
        help="Selecciona automáticamente todos los contenedores DLNA de vídeo",
    )

    return parser.parse_args()


def _ask_source() -> Choice | None:
    """
    Pregunta al usuario el origen a analizar.

    Reglas:
    - Visible siempre: interacción UI -> logger.info(..., always=True)
    - Validación defensiva: solo acepta Plex o DLNA
    - Enter cancela (devuelve None)
    - SILENT_MODE: prompts compactos
    """
    if SILENT_MODE:
        menu = "1) Plex\n2) DLNA\n(Enter cancela)"
        prompt = "> "
    else:
        menu = (
            "¿Qué origen quieres ejecutar?\n"
            "  1) Plex (analizar)\n"
            "  2) DLNA (analizar)\n"
            "(Pulsa Enter para cancelar)"
        )
        prompt = "Selecciona una opción (1/2): "

    valid_choices: set[str] = {"1", "2"}

    while True:
        logger.info("\n" + menu, always=True)
        raw = input(prompt).strip()

        if raw == "":
            logger.info("[AnalizaMovies] Operación cancelada.", always=True)
            return None

        if raw in valid_choices:
            return raw  # type: ignore[return-value]

        logger.info("Opción no válida (usa 1 ó 2, o Enter para cancelar).", always=True)


def _build_dlna_device_from_args(args: argparse.Namespace) -> DLNADevice | None:
    host = str(getattr(args, "dlna_host", "") or "").strip()
    location = str(getattr(args, "dlna_location", "") or "").strip()
    port = getattr(args, "dlna_port", None)
    if not host or not location or not isinstance(port, int) or port <= 0:
        return None

    friendly_name = (
        str(getattr(args, "dlna_friendly_name", "") or "").strip()
        or f"DLNA {host}:{port}"
    )
    device_id = str(getattr(args, "dlna_device_id", "") or "").strip() or None
    return DLNADevice(
        friendly_name=friendly_name,
        location=location,
        host=host,
        port=port,
        device_id=device_id,
    )


def start() -> None:
    """
    Entry-point principal (console_scripts).

    - Sin flags: comportamiento actual (menú).
    - Con flags: ejecución directa.
    """
    logger.progress("[AnalizaMovies] Inicio")

    args = _parse_args()
    bind_progress_from_env()
    update_run_progress(
        stage="booting",
        message="Initializing ingest engine.",
        message_key="run.message.boot_engine",
        record_event=True,
    )

    if SILENT_MODE:
        logger.progress(
            "[AnalizaMovies] SILENT_MODE=True"
            + (" DEBUG_MODE=True" if DEBUG_MODE else "")
        )
    elif DEBUG_MODE:
        logger.debug_ctx("ANALYZE", "SILENT_MODE=False DEBUG_MODE=True")

    try:
        dlna_device = _build_dlna_device_from_args(args)

        # =========================
        # MODO FLAGS (sin menú)
        # =========================
        if args.plex:
            logger.progress("[AnalizaMovies] Modo: Plex")
            update_run_progress(
                stage="starting",
                message="Preparing Plex analysis.",
                message_key="run.message.plex_prepare",
                record_event=True,
            )
            try:
                analyze_all_libraries()
            finally:
                logger.progress("[AnalizaMovies] Fin (Plex)")
            finish_run_progress(
                "succeeded",
                message="Plex ingest completed.",
                message_key="run.message.plex_done",
            )
            return

        if args.dlna:
            logger.progress("[AnalizaMovies] Modo: DLNA")
            update_run_progress(
                stage="starting",
                message="Preparing DLNA analysis.",
                message_key="run.message.dlna_prepare",
                record_event=True,
            )
            try:
                analyze_dlna_server(
                    device=dlna_device,
                    auto_select_all=bool(args.dlna_auto_select_all),
                )
            finally:
                logger.progress("[AnalizaMovies] Fin (DLNA)")
            finish_run_progress(
                "succeeded",
                message="DLNA ingest completed.",
                message_key="run.message.dlna_done",
            )
            return

        # =========================
        # MODO INTERACTIVO (actual)
        # =========================
        choice = _ask_source()
        if choice is None:
            return

        if choice == "1":
            logger.progress("[AnalizaMovies] Modo: Plex")
            update_run_progress(
                stage="starting",
                message="Preparing Plex analysis.",
                message_key="run.message.plex_prepare",
                record_event=True,
            )
            try:
                analyze_all_libraries()
            finally:
                logger.progress("[AnalizaMovies] Fin (Plex)")
            finish_run_progress(
                "succeeded",
                message="Plex ingest completed.",
                message_key="run.message.plex_done",
            )
            return

        logger.progress("[AnalizaMovies] Modo: DLNA")
        update_run_progress(
            stage="starting",
            message="Preparing DLNA analysis.",
            message_key="run.message.dlna_prepare",
            record_event=True,
        )
        try:
            analyze_dlna_server(
                device=dlna_device,
                auto_select_all=bool(args.dlna_auto_select_all),
            )
        finally:
            logger.progress("[AnalizaMovies] Fin (DLNA)")
        finish_run_progress(
            "succeeded",
            message="DLNA ingest completed.",
            message_key="run.message.dlna_done",
        )

    except KeyboardInterrupt:
        finish_run_progress(
            "cancelled",
            message="Ingest cancelled by the user.",
            message_key="run.message.cancelled",
            exit_code=130,
        )
        logger.info(
            "\n[AnalizaMovies] Interrumpido por el usuario (Ctrl+C).", always=True
        )
    except Exception as exc:
        finish_run_progress(
            "failed",
            message="Ingest finished with an error.",
            message_key="run.message.failed",
            exit_code=1,
        )
        logger.error(f"[AnalizaMovies] Error inesperado: {exc!r}", always=True)
        raise SystemExit(1)
    finally:
        logger.progress("[AnalizaMovies] Fin")


if __name__ == "__main__":
    start()
