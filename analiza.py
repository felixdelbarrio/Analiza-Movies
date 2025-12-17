from __future__ import annotations

"""
Punto de entrada unificado para análisis de películas.

Al ejecutarse, pregunta al usuario si quiere analizar:
  1) Plex
  2) DLNA

Según la opción elegida, delega en:
  - backend.analiza_plex.analyze_all_libraries()
  - backend.analiza_dlna.analyze_dlna_server()

Para el caso DLNA:
  - Primero se descubren servidores DLNA/UPnP en la red.
  - Se listan numerados para que el usuario elija uno.
  - Tras la selección, se lanza el flujo de análisis DLNA.
"""

from typing import Literal, Optional

from backend.analiza_plex import analyze_all_libraries
from backend.analiza_dlna import analyze_dlna_server
from backend.dlna_discovery import DLNADevice, discover_dlna_devices


Choice = Literal["1", "2"]


def _ask_source() -> Choice:
    """Pregunta al usuario el origen de datos a analizar."""
    prompt = (
        "¿Qué origen quieres analizar?\n"
        "  1) Plex\n"
        "  2) DLNA\n"
        "Selecciona una opción (1/2): "
    )
    while True:
        answer = input(prompt).strip()
        if answer in ("1", "2"):
            return answer  # type: ignore[return-value]
        print("Opción no válida. Introduce 1 o 2.")


def _select_dlna_device() -> Optional[DLNADevice]:
    """
    Descubre servidores DLNA en la red, los lista numerados y
    permite seleccionar uno.

    Devuelve:
      - DLNADevice elegido, o
      - None si no hay dispositivos o el usuario cancela.
    """
    print("\nBuscando servidores DLNA/UPnP en la red...\n")
    devices = discover_dlna_devices()

    if not devices:
        print("No se han encontrado servidores DLNA/UPnP en la red.")
        return None

    print("Se han encontrado los siguientes servidores DLNA/UPnP:\n")
    for idx, dev in enumerate(devices, start=1):
        print(f"  {idx}) {dev.friendly_name} ({dev.host}:{dev.port})")
        print(f"      LOCATION: {dev.location}")

    while True:
        raw = input(
            f"\nSelecciona un servidor (1-{len(devices)}) "
            "o pulsa Enter para cancelar: "
        ).strip()

        if raw == "":
            print("Operación cancelada, no se seleccionó servidor DLNA.")
            return None

        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(devices):
                chosen = devices[num - 1]
                print(
                    f"\nHas seleccionado: {chosen.friendly_name} "
                    f"({chosen.host}:{chosen.port})\n"
                )
                return chosen

        print(f"Opción no válida. Introduce un número entre 1 y {len(devices)}, "
              "o Enter para cancelar.")


def main() -> None:
    """Punto de entrada principal."""
    choice = _ask_source()

    if choice == "1":
        # Análisis clásico Plex
        analyze_all_libraries()
    else:
        # Flujo DLNA: descubrimiento + selección + análisis
        device = _select_dlna_device()
        if device is None:
            # Usuario canceló o no hay servidores; simplemente salimos.
            return

        # De momento el análisis DLNA trabaja sobre directorio local;
        # más adelante puedes adaptar analyze_dlna_server para usar `device`.
        analyze_dlna_server()


if __name__ == "__main__":
    main()