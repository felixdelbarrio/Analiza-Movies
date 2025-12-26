from __future__ import annotations

"""
backend/analiza.py

Punto de entrada unificado (CLI) para análisis de películas.

Este módulo es “UI/CLI puro”:
- Presenta menús y recoge inputs del usuario.
- Decide el origen (Plex o DLNA).
- En ruta DLNA: hace *discover* y selección de servidor (interacción).
- Arranca los orquestadores (Plex/DLNA), que ya gestionan:
    * concurrencia
    * progreso por item
    * escritura de CSVs
    * resumen final

Reglas de consola (alineado con backend/logger.py)
-------------------------------------------------
- Menús, prompts y validación de input: SIEMPRE visibles
    -> usar logger.info(..., always=True)
- Estado global (inicio / modo / fin) sin “spam”:
    -> usar logger.progress(...)
- Debug contextual:
    -> usar logger.debug_ctx("ANALYZE", "...") (ya respeta DEBUG_MODE/SILENT_MODE)

Comportamiento en modos
-----------------------
- SILENT_MODE=True:
    * prompts minimalistas
    * listados DLNA cortos (solo nombre) salvo DEBUG_MODE
- DEBUG_MODE=True:
    * señales extra útiles (p.ej. nº de dispositivos DLNA detectados)
"""

from typing import Literal

from backend import logger as logger
from backend.analiza_dlna import analyze_dlna_server
from backend.analiza_plex import analyze_all_libraries
from backend.config_base import (
    DEBUG_MODE,
    SILENT_MODE,
)
from backend.dlna_discovery import DLNADevice, discover_dlna_devices

Choice = Literal["1", "2"]


# ============================================================================
# Menús / Input handling
# ============================================================================

def _ask_source() -> Choice:
    """
    Pregunta al usuario el origen a analizar.

    Reglas:
    - Visible siempre: es interacción.
    - Validación defensiva: solo acepta "1" o "2".
    - SILENT_MODE: prompt compacto.
    """
    if SILENT_MODE:
        prompt = "1) Plex\n2) DLNA\n> "
    else:
        prompt = (
            "¿Qué origen quieres ejecutar?\n"
            "  1) Plex (analizar)\n"
            "  2) DLNA (analizar)\n"
            "Selecciona una opción (1/2): "
        )

    while True:
        answer = input(prompt).strip()
        if answer in ("1", "2"):
            return answer  # type: ignore[return-value]

        # Siempre visible (aunque SILENT_MODE=True)
        logger.info("Opción no válida (usa 1 ó 2).", always=True)


def _format_device_line(dev: DLNADevice) -> str:
    """
    Formatea una línea para el listado de dispositivos DLNA.

    - SILENT_MODE: solo nombre (lista minimalista)
    - No SILENT: nombre + host:port
    """
    if SILENT_MODE:
        return dev.friendly_name
    return f"{dev.friendly_name} ({dev.host}:{dev.port})"


def _select_dlna_device() -> DLNADevice | None:
    """
    Descubre servidores DLNA/UPnP en la red y permite seleccionar uno.

    UX / logging:
    - Texto de interacción siempre visible.
    - SILENT_MODE: listado minimalista; detalles solo si DEBUG_MODE.
    - Al cancelar (Enter), devuelve None.
    """
    logger.info("\nBuscando servidores DLNA/UPnP...\n", always=True)

    devices = discover_dlna_devices()

    if not devices:
        logger.info("No se han encontrado servidores DLNA/UPnP.", always=True)
        return None

    if SILENT_MODE and DEBUG_MODE:
        logger.progress(f"[DLNA][DEBUG] Dispositivos detectados: {len(devices)}")

    logger.info("Servidores DLNA/UPnP encontrados:\n", always=True)
    for idx, dev in enumerate(devices, start=1):
        logger.info(f"  {idx}) {_format_device_line(dev)}", always=True)
        if DEBUG_MODE:
            # Contexto adicional solo en debug (útil para diagnosticar)
            if SILENT_MODE:
                logger.info(f"      {dev.host}:{dev.port}", always=True)
            logger.info(f"      LOCATION: {dev.location}", always=True)

    while True:
        raw = input(f"\nServidor (1-{len(devices)}) o Enter cancela: ").strip()

        if raw == "":
            logger.info("Cancelado.", always=True)
            return None

        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(devices):
                chosen = devices[num - 1]
                logger.info(f"OK: {chosen.friendly_name}", always=True)
                if DEBUG_MODE:
                    logger.info(f"    {chosen.host}:{chosen.port}", always=True)
                    logger.info(f"    LOCATION: {chosen.location}", always=True)
                logger.info("", always=True)
                return chosen

        logger.info(f"Opción no válida (1-{len(devices)} o Enter).", always=True)


# ============================================================================
# Entry-point
# ============================================================================

def main() -> None:
    """
    Entry-point principal.

    Responsabilidades:
    - Mostrar marco global (inicio / modo / fin).
    - Encapsular la UX (menú + selección DLNA).
    - Invocar orquestadores que realizan el trabajo pesado.
    """
    logger.progress("[AnalizaMovies] Inicio")

    # Contexto de modo (compacto)
    if SILENT_MODE:
        logger.progress("[AnalizaMovies] SILENT_MODE=True" + (" DEBUG_MODE=True" if DEBUG_MODE else ""))
    elif DEBUG_MODE:
        logger.debug_ctx("ANALYZE", "SILENT_MODE=False DEBUG_MODE=True")

    choice = _ask_source()

    if choice == "1":
        logger.progress("[AnalizaMovies] Modo: Plex")
        analyze_all_libraries()
        logger.progress("[AnalizaMovies] Fin (Plex)")
        return

    logger.progress("[AnalizaMovies] Modo: DLNA")
    device = _select_dlna_device()
    if device is None:
        logger.progress("[AnalizaMovies] Fin (DLNA cancelado)")
        return

    analyze_dlna_server(device)
    logger.progress("[AnalizaMovies] Fin (DLNA)")


if __name__ == "__main__":
    main()