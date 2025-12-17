# backend/dlna_discovery.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import socket
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger


# Dirección multicast SSDP estándar
SSDP_ADDR: Tuple[str, int] = ("239.255.255.250", 1900)

# Usamos un ST genérico para que responda el máximo número de servidores
# (incluyendo Plex u otros MediaServers que no anuncian exactamente
# "urn:schemas-upnp-org:device:MediaServer:1").
SSDP_ST: str = "ssdp:all"

# MX: segundos máximos que los servidores pueden esperar antes de responder
SSDP_MX: int = 2


@dataclass
class DLNADevice:
    friendly_name: str
    location: str
    host: str
    port: int


def _parse_ssdp_response(data: bytes) -> Dict[str, str]:
    """
    Parsea una respuesta SSDP (tipo cabeceras HTTP) a dict de headers en minúsculas.
    """
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return {}

    lines = text.split("\r\n")
    headers: Dict[str, str] = {}

    for line in lines[1:]:  # saltamos la primera línea "HTTP/1.1 200 OK"
        if not line.strip():
            continue
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    return headers


def _fetch_friendly_name(location: str) -> str:
    """
    Descarga la descripción del dispositivo UPnP (XML) desde LOCATION y
    extrae el <friendlyName>. Si falla, devuelve la LOCATION como fallback.
    """
    try:
        with urlopen(location, timeout=3) as resp:
            xml_data = resp.read()
    except Exception as exc:  # pragma: no cover (errores de red reales)
        _logger.warning(f"[DLNA] No se pudo descargar LOCATION {location}: {exc}")
        return location

    try:
        root = ET.fromstring(xml_data)
        # El friendlyName típico está en device/friendlyName
        # namespace libre o con ns; probamos ambas cosas de forma simple
        # Sin namespaces:
        for dev in root.iter("device"):
            fn = dev.findtext("friendlyName")
            if fn:
                return fn.strip()

        # Con posibles namespaces (heurística mínima)
        for elem in root.iter():
            if elem.tag.endswith("device"):
                fn = None
                for child in elem:
                    if isinstance(child.tag, str) and child.tag.endswith("friendlyName"):
                        fn = child.text
                        break
                if fn:
                    return fn.strip()
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] Error parseando XML de {location}: {exc}")

    return location


def discover_dlna_devices(
    timeout: float = 3.0,
    st: str = SSDP_ST,
    mx: int = SSDP_MX,
) -> List[DLNADevice]:
    """
    Descubre dispositivos DLNA/UPnP MediaServer en la red usando SSDP.

    Devuelve una lista de DLNADevice con:
      - friendly_name
      - location (URL completa de descripción)
      - host
      - port

    Si no se encuentra ningún dispositivo con el ST indicado y éste no es
    "ssdp:all", se hace un reintento automático con ST="ssdp:all".
    """
    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"MX: {int(mx)}\r\n"
        f"ST: {st}\r\n"
        "\r\n"
    ).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.settimeout(timeout)
        # Algunos stacks requieren permitir reuse
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.sendto(msg, SSDP_ADDR)

        start = time.time()
        locations: Dict[str, DLNADevice] = {}

        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                break

            try:
                sock.settimeout(remaining)
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            except Exception:  # pragma: no cover
                break

            headers = _parse_ssdp_response(data)
            loc = headers.get("location")
            if not loc:
                continue

            # Deduplicamos por LOCATION
            if loc in locations:
                continue

            parsed = urlparse(loc)
            host = parsed.hostname or addr[0]
            port = parsed.port or 80

            friendly = _fetch_friendly_name(loc)
            locations[loc] = DLNADevice(
                friendly_name=friendly,
                location=loc,
                host=host,
                port=port,
            )

        devices = list(locations.values())

        # Fallback: si no hay dispositivos y el ST no era ya "ssdp:all",
        # reintentamos una vez con ST genérico.
        if not devices and st != "ssdp:all":
            _logger.info(
                f"[DLNA] Ningún dispositivo con ST={st!r}, "
                "reintentando con ST='ssdp:all'."
            )
            return discover_dlna_devices(timeout=timeout, st="ssdp:all", mx=mx)

        _logger.info(f"[DLNA] Descubiertos {len(devices)} servidor(es) DLNA/UPnP.")
        return devices
    finally:
        sock.close()