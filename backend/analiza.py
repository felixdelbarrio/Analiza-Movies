from __future__ import annotations

"""
backend/analiza.py

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

from typing import Literal

from backend import logger as logger
from backend.analiza_dlna import analyze_dlna_server
from backend.analiza_plex import analyze_all_libraries
from backend.config_base import DEBUG_MODE, SILENT_MODE

Choice = Literal["1", "2"]


def _ask_source() -> Choice | None:
    """
    Pregunta al usuario el origen a analizar.

    Reglas:
    - Visible siempre: interacción UI -> logger.info(..., always=True)
    - Validación defensiva: solo acepta "1" o "2"
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

    while True:
        logger.info("\n" + menu, always=True)
        raw = input(prompt).strip()

        if raw == "":
            logger.info("[AnalizaMovies] Operación cancelada.", always=True)
            return None

        if raw in ("1", "2"):
            return raw  # type: ignore[return-value]

        logger.info("Opción no válida (usa 1 ó 2, o Enter para cancelar).", always=True)


def main() -> None:
    """
    Entry-point principal.

    Responsabilidades:
    - Mostrar marco global (inicio / modo / fin).
    - Encapsular UX mínima (menú).
    - Invocar orquestadores que realizan el trabajo pesado.
    """
    logger.progress("[AnalizaMovies] Inicio")

    # Contexto de modo (compacto)
    if SILENT_MODE:
        logger.progress("[AnalizaMovies] SILENT_MODE=True" + (" DEBUG_MODE=True" if DEBUG_MODE else ""))
    elif DEBUG_MODE:
        logger.debug_ctx("ANALYZE", "SILENT_MODE=False DEBUG_MODE=True")

    try:
        choice = _ask_source()
        if choice is None:
            return

        if choice == "1":
            logger.progress("[AnalizaMovies] Modo: Plex")
            try:
                analyze_all_libraries()
            finally:
                logger.progress("[AnalizaMovies] Fin (Plex)")
            return

        logger.progress("[AnalizaMovies] Modo: DLNA")
        try:
            # DLNA: el orquestador gestiona selección/discovery si no se le pasa device
            analyze_dlna_server()
        finally:
            logger.progress("[AnalizaMovies] Fin (DLNA)")

    except KeyboardInterrupt:
        # Salida limpia en Ctrl+C
        logger.info("\n[AnalizaMovies] Interrumpido por el usuario (Ctrl+C).", always=True)
    finally:
        # Marco global fin (sin duplicar los "Fin (Plex/DLNA)")
        logger.progress("[AnalizaMovies] Fin")


if __name__ == "__main__":
    main()