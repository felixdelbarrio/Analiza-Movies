from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import socket
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger


SSDP_ADDR: Tuple[str, int] = ("239.255.255.250", 1900)

# Mantenemos ssdp:all para no perder servidores que anuncian mal (p.ej. Plex),
# pero filtramos luego por ContentDirectory.
SSDP_ST: str = "ssdp:all"
SSDP_MX: int = 2


@dataclass(frozen=True, slots=True)
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

    lines = text.split("\r\n")
    headers: Dict[str, str] = {}

    for line in lines[1:]:
        if not line.strip():
            continue
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    return headers


def _fetch_device_description(location: str) -> bytes | None:
    try:
        with urlopen(location, timeout=3) as resp:
            return resp.read()
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] No se pudo descargar LOCATION {location}: {exc}")
        return None


def _extract_friendly_name(xml_data: bytes, fallback: str) -> str:
    try:
        root = ET.fromstring(xml_data)
    except Exception:  # pragma: no cover
        return fallback

    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith("friendlyName"):
            if elem.text:
                name = elem.text.strip()
                if name:
                    return name
            break

    return fallback


def _has_content_directory(xml_data: bytes) -> bool:
    try:
        root = ET.fromstring(xml_data)
    except Exception:  # pragma: no cover
        return False

    for elem in root.iter():
        if not (isinstance(elem.tag, str) and elem.tag.endswith("serviceType")):
            continue
        if elem.text and "ContentDirectory" in elem.text:
            return True

    return False


def discover_dlna_devices(
    timeout: float = 3.0,
    st: str = SSDP_ST,
    mx: int = SSDP_MX,
) -> List[DLNADevice]:
    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"ST: {st}\r\n"
        f"MX: {mx}\r\n"
        "\r\n"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.settimeout(timeout)
        sock.sendto(msg.encode("utf-8"), SSDP_ADDR)

        start = time.time()
        found: Dict[str, DLNADevice] = {}

        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                break

            sock.settimeout(remaining)

            try:
                data, _ = sock.recvfrom(65507)
            except socket.timeout:
                break
            except Exception as exc:  # pragma: no cover
                _logger.warning(f"[DLNA] Error recibiendo SSDP: {exc}")
                break

            headers = _parse_ssdp_response(data)
            location = headers.get("location")
            if not location:
                continue

            parsed = urlparse(location)
            if not parsed.hostname:
                continue

            host = parsed.hostname
            port = parsed.port if parsed.port is not None else 80

            xml_data = _fetch_device_description(location)
            if xml_data is None:
                continue

            if not _has_content_directory(xml_data):
                _logger.info(
                    f"[DLNA] Ignorando dispositivo sin ContentDirectory: {location}",
                    always=True,
                )
                continue

            friendly = _extract_friendly_name(xml_data, fallback=location)

            if location not in found:
                found[location] = DLNADevice(
                    friendly_name=friendly,
                    location=location,
                    host=host,
                    port=port,
                )

        devices = list(found.values())
        _logger.info(f"[DLNA] Descubiertos {len(devices)} servidor(es) DLNA/UPnP.", always=True)
        return devices

    finally:
        try:
            sock.close()
        except Exception:
            pass