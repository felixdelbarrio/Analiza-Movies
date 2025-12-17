# backend/dlna_discovery.py
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger

SSDP_ADDR = ("239.255.255.250", 1900)
SSDP_ST = "urn:schemas-upnp-org:device:MediaServer:1"
SSDP_MX = 2


@dataclass
class DLNADevice:
    friendly_name: str
    location: str
    host: str
    port: int


def _parse_ssdp_response(data: bytes) -> Dict[str, str]:
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return {}

    headers = {}
    for line in text.split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return headers


def _fetch_friendly_name(location: str) -> str:
    try:
        with urlopen(location, timeout=3) as resp:
            xml_data = resp.read()
    except Exception:
        return location

    try:
        root = ET.fromstring(xml_data)
        for dev in root.iter():
            if dev.tag.endswith("device"):
                fn = dev.findtext("friendlyName")
                if fn:
                    return fn.strip()
    except Exception:
        pass

    return location


def discover_dlna_devices(timeout: float = 3.0) -> List[DLNADevice]:
    """Descubrimiento robusto para macOS + Linux + Windows."""
    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"MX: {SSDP_MX}\r\n"
        f"ST: {SSDP_ST}\r\n"
        "\r\n"
    ).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    try:
        # Necesario en macOS para compartir el socket multicast
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except Exception:
            pass  # no soportado en algunas plataformas

        # Unirse al grupo multicast
        mreq = socket.inet_aton(SSDP_ADDR[0]) + socket.inet_aton("0.0.0.0")
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        sock.settimeout(timeout)
        sock.sendto(msg, SSDP_ADDR)

        start = time.time()
        found: Dict[str, DLNADevice] = {}

        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                break

            try:
                sock.settimeout(remaining)
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            except Exception:
                break

            headers = _parse_ssdp_response(data)
            loc = headers.get("location")
            if not loc:
                continue

            if loc in found:
                continue

            parsed = urlparse(loc)
            host = parsed.hostname or addr[0]
            port = parsed.port or 80
            friendly = _fetch_friendly_name(loc)

            found[loc] = DLNADevice(
                friendly_name=friendly,
                location=loc,
                host=host,
                port=port,
            )

        devices = list(found.values())
        _logger.info(f"[DLNA] Servidores descubiertos: {len(devices)}")
        return devices

    finally:
        sock.close()