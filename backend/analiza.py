from __future__ import annotations

"""
analiza.py

Punto de entrada unificado para análisis de películas.

Este módulo es “UI/CLI” puro: menús, selección de origen y arranque de orquestadores.
No debe hacer trabajo pesado de red/IO salvo el “discover” de DLNA cuando el usuario
elige esa ruta. 

Reglas de consola (alineadas con backend/logger.py):
- Menús y prompts deben ser visibles SIEMPRE (interacción):
    * usar _logger.info(..., always=True) para texto de menú y errores de input
- SILENT_MODE=True:
    * minimizar texto no esencial
    * usar _logger.progress(...) como “señal de estado” (inicio, modo, fin)
    * listados DLNA minimalistas (solo nombre); detalles solo si DEBUG_MODE
- DEBUG_MODE=True:
    * contexto extra útil sin saturar (p.ej. nº de dispositivos detectados)
    * para debug contextual, preferir _logger.debug_ctx("ANALYZE", "...") si existe

Notas:
- `analyze_all_libraries()` y `analyze_dlna_server()` ya gestionan su propio progreso
  y reportes finales. Aquí solo los invocamos y mostramos el “marco” (inicio/fin).
"""

from typing import Literal

from backend import logger as _logger
from backend.analiza_dlna import analyze_dlna_server
from backend.analiza_plex import analyze_all_libraries
from backend.config import DEBUG_MODE, SILENT_MODE
from backend.dlna_discovery import DLNADevice, discover_dlna_devices

Choice = Literal["1", "2"]


def _ask_source() -> Choice:
    """
    Pregunta al usuario el origen de datos a analizar.

    Reglas:
    - Debe ser visible siempre (es un prompt).
    - SILENT_MODE=True: prompt minimalista.
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

        # Mensaje visible siempre, independientemente de SILENT_MODE.
        if SILENT_MODE:
            _logger.info("Opción no válida (usa 1 ó 2).", always=True)
        else:
            _logger.info("Opción no válida. Introduce 1 ó 2.", always=True)


def _select_dlna_device() -> DLNADevice | None:
    """
    Descubre servidores DLNA/UPnP en la red, los lista y permite seleccionar uno.

    Reglas:
    - Listado/prompt visible SIEMPRE (interacción).
    - SILENT_MODE=True:
        * textos más cortos
        * lista minimalista: idx) friendly_name
        * host/port y LOCATION solo en DEBUG_MODE
    """
    if SILENT_MODE:
        _logger.info("\nBuscando servidores DLNA/UPnP...\n", always=True)
    else:
        _logger.info("\nBuscando servidores DLNA/UPnP en la red...\n", always=True)

    devices = discover_dlna_devices()

    if not devices:
        _logger.info("No se han encontrado servidores DLNA/UPnP.", always=True)
        return None

    if SILENT_MODE and DEBUG_MODE:
        _logger.progress(f"[DLNA][DEBUG] Dispositivos detectados: {len(devices)}")

    _logger.info("Servidores DLNA/UPnP encontrados:\n", always=True)
    for idx, dev in enumerate(devices, start=1):
        if SILENT_MODE:
            _logger.info(f"  {idx}) {dev.friendly_name}", always=True)
            if DEBUG_MODE:
                _logger.info(f"      {dev.host}:{dev.port}", always=True)
                _logger.info(f"      LOCATION: {dev.location}", always=True)
        else:
            _logger.info(f"  {idx}) {dev.friendly_name} ({dev.host}:{dev.port})", always=True)
            _logger.info(f"      LOCATION: {dev.location}", always=True)

    while True:
        raw = input(f"\nServidor (1-{len(devices)}) o Enter cancela: ").strip()

        if raw == "":
            _logger.info("Cancelado.", always=True)
            return None

        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(devices):
                chosen = devices[num - 1]
                if SILENT_MODE:
                    _logger.info(f"OK: {chosen.friendly_name}\n", always=True)
                    if DEBUG_MODE:
                        _logger.info(f"    {chosen.host}:{chosen.port}", always=True)
                        _logger.info(f"    LOCATION: {chosen.location}\n", always=True)
                else:
                    _logger.info(
                        f"OK: {chosen.friendly_name} ({chosen.host}:{chosen.port})\n",
                        always=True,
                    )
                return chosen

        _logger.info(f"Opción no válida (1-{len(devices)} o Enter).", always=True)


def main() -> None:
    """
    Punto de entrada principal.

    Responsabilidades:
    - Mostrar “marco” de ejecución (inicio, modo, fin).
    - En DLNA, descubrir/seleccionar servidor antes de analizar.

    Consola:
    - progress() para estado (siempre visible, sin timestamps).
    - info(always=True) para interacción (menús/errores/prompt context).
    """
    _logger.progress("[AnalizaMovies] Inicio")

    if SILENT_MODE:
        if DEBUG_MODE:
            _logger.progress("[AnalizaMovies][DEBUG] SILENT_MODE=True DEBUG_MODE=True")
        else:
            _logger.progress("[AnalizaMovies] SILENT_MODE=True")

    choice = _ask_source()

    if choice == "1":
        _logger.progress("[AnalizaMovies] Modo: Plex")
        analyze_all_libraries()
        _logger.progress("[AnalizaMovies] Fin (Plex)")
        return

    _logger.progress("[AnalizaMovies] Modo: DLNA")
    device = _select_dlna_device()
    if device is None:
        _logger.progress("[AnalizaMovies] Fin (DLNA cancelado)")
        return

    analyze_dlna_server(device)
    _logger.progress("[AnalizaMovies] Fin (DLNA)")


if __name__ == "__main__":
    main()