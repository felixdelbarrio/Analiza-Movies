from __future__ import annotations

"""
backend/dlna_discovery.py

Descubrimiento de dispositivos DLNA/UPnP por SSDP (M-SEARCH).

Objetivo:
- Emitir un broadcast SSDP y recoger respuestas durante `timeout`.
- Para cada respuesta con header LOCATION:
    1) Descargar el device description XML (LOCATION).
    2) Verificar que expone el servicio ContentDirectory.
    3) Extraer friendlyName.
    4) Devolver una lista de DLNADevice únicos (deduplicados por LOCATION).

Filosofía de logs (alineada con backend/logger.py):
- Nunca romper el flujo por logging.
- Mensajes de “ruido” (dispositivos ignorados, detalles de red) → solo en DEBUG_MODE.
- En SILENT_MODE:
    - evitar spam; debug contextual se emite con progress() únicamente si DEBUG_MODE=True.
- En NO SILENT:
    - debug contextual con info() si DEBUG_MODE=True.

Notas de robustez:
- Algunos servidores (p.ej. Plex) pueden anunciar cosas de forma inconsistente.
  Mantenemos ST=ssdp:all y filtramos después por ContentDirectory.
- Descargas LOCATION con timeout corto para no bloquear el discovery.
"""

from dataclasses import dataclass
import socket
import time
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger


# SSDP multicast estándar
SSDP_ADDR: Tuple[str, int] = ("239.255.255.250", 1900)

# Mantenemos ssdp:all para no perder servidores que anuncian mal (p.ej. Plex),
# pero filtramos luego por ContentDirectory.
SSDP_ST: str = "ssdp:all"
SSDP_MX: int = 2


@dataclass(frozen=True, slots=True)
class DLNADevice:
    """Modelo mínimo de un servidor DLNA descubierto."""
    friendly_name: str
    location: str
    host: str
    port: int


# ============================================================================
# Logging controlado por modos (sin depender de imports directos a config)
# ============================================================================

def _safe_get_cfg():
    """Devuelve backend.config si ya está importado (evita dependencias circulares)."""
    import sys
    return sys.modules.get("backend.config")


def _is_debug_mode() -> bool:
    cfg = _safe_get_cfg()
    if cfg is None:
        return False
    try:
        return bool(getattr(cfg, "DEBUG_MODE", False))
    except Exception:
        return False


def _is_silent_mode() -> bool:
    cfg = _safe_get_cfg()
    if cfg is None:
        return False
    try:
        return bool(getattr(cfg, "SILENT_MODE", False))
    except Exception:
        return False


def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False → no hace nada.
    - DEBUG_MODE=True:
        * SILENT_MODE=True: progress (señales mínimas).
        * SILENT_MODE=False: info normal.
    """
    if not _is_debug_mode():
        return

    text = str(msg)
    try:
        if _is_silent_mode():
            _logger.progress(f"[DLNA][DEBUG] {text}")
        else:
            _logger.info(f"[DLNA][DEBUG] {text}")
    except Exception:
        # fallback solo en modo no-silent
        if not _is_silent_mode():
            print(text)


# ============================================================================
# SSDP parsing + XML helpers
# ============================================================================

def _parse_ssdp_response(data: bytes) -> Dict[str, str]:
    """
    Parsea una respuesta SSDP (cabeceras HTTP-like) a dict lower-case.

    Devuelve {} si no puede decodificar.
    """
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return {}

    lines = text.split("\r\n")
    headers: Dict[str, str] = {}

    # ignoramos la primera línea ("HTTP/1.1 200 OK")
    for line in lines[1:]:
        if not line.strip():
            continue
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    return headers


def _fetch_device_description(location: str) -> bytes | None:
    """
    Descarga el XML de descripción del dispositivo (LOCATION).

    Timeout bajo para evitar bloquear discovery.
    """
    try:
        with urlopen(location, timeout=3) as resp:
            return resp.read()
    except Exception as exc:  # pragma: no cover
        # Warning visible solo si el logger decide (SILENT_MODE lo suele suprimir).
        _logger.warning(f"[DLNA] No se pudo descargar LOCATION {location}: {exc}")
        return None


def _extract_friendly_name(xml_data: bytes, fallback: str) -> str:
    """
    Extrae <friendlyName> del XML. Si falla, devuelve `fallback`.
    """
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
    """
    Detecta si el device description expone el servicio ContentDirectory.
    """
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


# ============================================================================
# API pública
# ============================================================================

def discover_dlna_devices(
    timeout: float = 3.0,
    st: str = SSDP_ST,
    mx: int = SSDP_MX,
) -> List[DLNADevice]:
    """
    Descubre servidores DLNA/UPnP en la red mediante SSDP.

    Args:
        timeout: segundos totales de escucha tras enviar M-SEARCH.
        st: search target SSDP (por defecto ssdp:all).
        mx: ventana MX del M-SEARCH.

    Returns:
        Lista de DLNADevice, deduplicados por LOCATION.
    """
    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"ST: {st}\r\n"
        f"MX: {mx}\r\n"
        "\r\n"
    )

    # UDP multicast SSDP
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
                data, addr = sock.recvfrom(65507)
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
                _log_debug(f"SSDP response sin hostname en LOCATION: {location!r} from={addr!r}")
                continue

            host = parsed.hostname
            port = parsed.port if parsed.port is not None else 80

            xml_data = _fetch_device_description(location)
            if xml_data is None:
                _log_debug(f"LOCATION fetch falló: {location!r}")
                continue

            if not _has_content_directory(xml_data):
                # Esto es “ruido” en la mayoría de redes -> solo en DEBUG
                _log_debug(f"Ignorando sin ContentDirectory: {location}")
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

        # Señal final: en modo normal es útil; en SILENT_MODE no debe generar ruido
        # porque el caller ya imprime su propio “Buscando… / encontrados…”.
        if _is_silent_mode():
            _log_debug(f"Descubiertos {len(devices)} servidor(es) DLNA/UPnP.")
        else:
            _logger.info(
                f"[DLNA] Descubiertos {len(devices)} servidor(es) DLNA/UPnP.",
                always=True,
            )

        return devices

    finally:
        try:
            sock.close()
        except Exception:
            pass