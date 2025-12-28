from __future__ import annotations

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

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Literal

from backend import logger as logger
from backend.analiza_dlna import analyze_dlna_server
from backend.analiza_plex import analyze_all_libraries
from backend.config_base import ANALIZA_AUTO_DASHBOARD, DEBUG_MODE, SILENT_MODE
from backend.config_reports import REPORT_ALL_PATH, REPORT_FILTERED_PATH

Choice = Literal["0", "1", "2"]


def _parse_args() -> argparse.Namespace:
    """
    Flags opcionales para automatizar ejecución (sin menú).

    - Sin flags: UX actual por menú.
    - Con flags: ejecución directa.
    """
    parser = argparse.ArgumentParser(
        prog="start",
        add_help=True,
        description="Analiza Movies - CLI backend (Plex/DLNA/Streamlit)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plex", action="store_true", help="Analizar Plex directamente (sin menú)")
    mode.add_argument("--dlna", action="store_true", help="Analizar DLNA directamente (sin menú)")
    mode.add_argument("--dashboard", action="store_true", help="Abrir solo el dashboard (sin análisis)")

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="No abrir el dashboard tras el análisis (aunque esté activado por config)",
    )

    return parser.parse_args()


def _reports_available() -> bool:
    """
    Devuelve True si hay reports suficientes para habilitar el dashboard visual.

    Criterio:
    - Existe REPORT_ALL_PATH
    - Existe REPORT_FILTERED_PATH
    """
    return Path(REPORT_ALL_PATH).exists() and Path(REPORT_FILTERED_PATH).exists()


def _run_streamlit_dashboard() -> None:
    """
    Lanza el dashboard de Streamlit usando el intérprete actual.

    - Usa: python -m streamlit run frontend/dashboard.py
    - Ruta calculada relativa a la raíz del repo.
    """
    project_root = Path(__file__).resolve().parents[1]
    dashboard_path = project_root / "frontend" / "dashboard.py"

    if not dashboard_path.exists():
        logger.info(
            f"[AnalizaMovies] No se encuentra el dashboard: {str(dashboard_path)!r}",
            always=True,
        )
        return

    if not _reports_available():
        logger.info(
            "[AnalizaMovies] No hay reports suficientes (report_all.csv + report_filtered.csv) "
            "en el repositorio de reports; no se puede abrir el análisis visual.",
            always=True,
        )
        return

    logger.progress("[AnalizaMovies] Modo: Análisis visual (Streamlit)")

    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]

    if DEBUG_MODE and not SILENT_MODE:
        logger.debug_ctx("DASH", f"cmd={cmd!r}")
        logger.debug_ctx("DASH", f"REPORT_ALL_PATH={REPORT_ALL_PATH!r}")
        logger.debug_ctx("DASH", f"REPORT_FILTERED_PATH={REPORT_FILTERED_PATH!r}")

    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        logger.info("\n[AnalizaMovies] Interrumpido por el usuario (Ctrl+C).", always=True)
        return
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[AnalizaMovies] Error lanzando Streamlit: {exc!r}", always=True)
        return

    if result.returncode != 0:
        logger.info(
            f"[AnalizaMovies] Streamlit terminó con código {result.returncode}.",
            always=True,
        )
    else:
        logger.progress("[AnalizaMovies] Fin (Análisis visual)")


def _maybe_run_dashboard_after_analysis() -> None:
    """
    Ejecuta el dashboard si está habilitado por configuración
    y existen reports suficientes.
    """
    if not ANALIZA_AUTO_DASHBOARD:
        return
    _run_streamlit_dashboard()


def _ask_source() -> Choice | None:
    """
    Pregunta al usuario el origen a analizar.

    Reglas:
    - Visible siempre: interacción UI -> logger.info(..., always=True)
    - Validación defensiva: acepta "0" solo si hay reports disponibles
    - Enter cancela (devuelve None)
    - SILENT_MODE: prompts compactos
    """
    has_reports = _reports_available()

    if SILENT_MODE:
        if has_reports:
            menu = "0) Análisis visual\n1) Plex\n2) DLNA\n(Enter cancela)"
            prompt = "> "
        else:
            menu = "1) Plex\n2) DLNA\n(Enter cancela)"
            prompt = "> "
    else:
        if has_reports:
            menu = (
                "¿Qué quieres ejecutar?\n"
                "  0) Análisis visual (Streamlit)\n"
                "  1) Plex (analizar)\n"
                "  2) DLNA (analizar)\n"
                "(Pulsa Enter para cancelar)"
            )
            prompt = "Selecciona una opción (0/1/2): "
        else:
            menu = (
                "¿Qué origen quieres ejecutar?\n"
                "  1) Plex (analizar)\n"
                "  2) DLNA (analizar)\n"
                "(Pulsa Enter para cancelar)"
            )
            prompt = "Selecciona una opción (1/2): "

    valid_choices: set[str] = {"1", "2"}
    if has_reports:
        valid_choices.add("0")

    while True:
        logger.info("\n" + menu, always=True)
        raw = input(prompt).strip()

        if raw == "":
            logger.info("[AnalizaMovies] Operación cancelada.", always=True)
            return None

        if raw in valid_choices:
            return raw  # type: ignore[return-value]

        if has_reports:
            logger.info("Opción no válida (usa 0, 1 ó 2, o Enter para cancelar).", always=True)
        else:
            logger.info("Opción no válida (usa 1 ó 2, o Enter para cancelar).", always=True)


def start() -> None:
    """
    Entry-point principal (console_scripts).

    - Sin flags: comportamiento actual (menú).
    - Con flags: ejecución directa.
    """
    logger.progress("[AnalizaMovies] Inicio")

    args = _parse_args()

    if SILENT_MODE:
        logger.progress("[AnalizaMovies] SILENT_MODE=True" + (" DEBUG_MODE=True" if DEBUG_MODE else ""))
    elif DEBUG_MODE:
        logger.debug_ctx("ANALYZE", "SILENT_MODE=False DEBUG_MODE=True")

    try:
        # =========================
        # MODO FLAGS (sin menú)
        # =========================
        if args.dashboard:
            _run_streamlit_dashboard()
            return

        if args.plex:
            logger.progress("[AnalizaMovies] Modo: Plex")
            try:
                analyze_all_libraries()
            finally:
                logger.progress("[AnalizaMovies] Fin (Plex)")

            if not args.no_dashboard:
                _maybe_run_dashboard_after_analysis()
            return

        if args.dlna:
            logger.progress("[AnalizaMovies] Modo: DLNA")
            try:
                analyze_dlna_server()
            finally:
                logger.progress("[AnalizaMovies] Fin (DLNA)")

            if not args.no_dashboard:
                _maybe_run_dashboard_after_analysis()
            return

        # =========================
        # MODO INTERACTIVO (actual)
        # =========================
        choice = _ask_source()
        if choice is None:
            return

        if choice == "0":
            _run_streamlit_dashboard()
            return

        if choice == "1":
            logger.progress("[AnalizaMovies] Modo: Plex")
            try:
                analyze_all_libraries()
            finally:
                logger.progress("[AnalizaMovies] Fin (Plex)")
            _maybe_run_dashboard_after_analysis()
            return

        logger.progress("[AnalizaMovies] Modo: DLNA")
        try:
            analyze_dlna_server()
        finally:
            logger.progress("[AnalizaMovies] Fin (DLNA)")
        _maybe_run_dashboard_after_analysis()

    except KeyboardInterrupt:
        logger.info("\n[AnalizaMovies] Interrumpido por el usuario (Ctrl+C).", always=True)
    finally:
        logger.progress("[AnalizaMovies] Fin")


if __name__ == "__main__":
    start()