from __future__ import annotations

"""
backend/dlna_discovery.py

Descubrimiento de dispositivos DLNA/UPnP por SSDP (M-SEARCH).

(…docstring igual que el tuyo…)
"""

from dataclasses import dataclass
import socket
import time
from types import ModuleType
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger


# SSDP multicast estándar
SSDP_ADDR: tuple[str, int] = ("239.255.255.250", 1900)

# Defaults (se sobre-escriben si config está presente)
_DEFAULT_SSDP_ST = "ssdp:all"
_DEFAULT_SSDP_MX = 2
_DEFAULT_DISCOVERY_TIMEOUT = 3.0
_DEFAULT_DEVICE_DESC_TIMEOUT = 3.0
_DEFAULT_DEVICE_DESC_MAX_BYTES = 2_000_000

# Anti-bucle: cap de respuestas procesadas por “run” de discovery.
_DEFAULT_DISCOVERY_MAX_RESPONSES = 2048

# Anti-flood por host
_DEFAULT_DISCOVERY_MAX_RESPONSES_PER_HOST = 512

_DEFAULT_DENY_TOKENS = [
    "internetgatewaydevice",
    "wanipconnection",
    "wanpppconnection",
    "wandevice",
    "igd",
]
_DEFAULT_ALLOW_HINT_TOKENS = ["mediaserver", "contentdirectory"]


@dataclass(frozen=True, slots=True)
class DLNADevice:
    """
    Modelo mínimo de un servidor DLNA descubierto.
    """
    friendly_name: str
    location: str
    host: str
    port: int
    device_id: str | None = None


# ============================================================================
# Logging controlado por modos (sin depender de imports directos a config)
# ============================================================================

def _safe_get_cfg() -> ModuleType | None:
    """Devuelve backend.config si ya está importado (evita dependencias circulares)."""
    import sys

    mod = sys.modules.get("backend.config")
    return mod if isinstance(mod, ModuleType) else None


def _cfg_get(name: str, default: Any) -> Any:
    """Lee un atributo de backend.config (si existe) sin romper nunca."""
    cfg = _safe_get_cfg()
    if cfg is None:
        return default
    try:
        return getattr(cfg, name, default)
    except Exception:
        return default


def _is_debug_mode() -> bool:
    try:
        return bool(_cfg_get("DEBUG_MODE", False))
    except Exception:
        return False


def _is_silent_mode() -> bool:
    try:
        return bool(_cfg_get("SILENT_MODE", False))
    except Exception:
        return False


def _log_debug(msg: object) -> None:
    if not _is_debug_mode():
        return

    text = str(msg)
    try:
        if _is_silent_mode():
            _logger.progress(f"[DLNA][DEBUG] {text}")
        else:
            _logger.info(f"[DLNA][DEBUG] {text}")
    except Exception:
        if not _is_silent_mode():
            try:
                print(text)
            except Exception:
                pass


# ============================================================================
# SSDP parsing + URL/XML helpers
# ============================================================================

def _parse_ssdp_response(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return {}

    lines = text.split("\r\n")
    if not lines:
        return {}

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    return headers


def _normalize_location(location: str) -> str:
    loc = (location or "").strip()
    if not loc:
        return loc

    try:
        p = urlparse(loc)
    except Exception:
        return loc

    scheme = (p.scheme or "http").lower()
    host = (p.hostname or "").lower()
    if not host:
        return loc

    port = p.port
    netloc = host if port is None else f"{host}:{port}"
    path = p.path or "/"

    return urlunparse((scheme, netloc, path, p.params, p.query, ""))


def _headers_text_for_filter(headers: dict[str, str], location: str) -> str:
    parts = [
        location,
        headers.get("st", ""),
        headers.get("usn", ""),
        headers.get("server", ""),
        headers.get("nt", ""),
    ]
    return " | ".join(p for p in parts if p).lower()


def _should_ignore_by_tokens(text: str, deny_tokens: list[str]) -> bool:
    for tok in deny_tokens:
        if tok and tok in text:
            return True
    return False


def _has_any_allow_hint(text: str, allow_tokens: list[str]) -> bool:
    for tok in allow_tokens:
        if tok and tok in text:
            return True
    return False


def _extract_device_id_from_headers(headers: dict[str, str]) -> str | None:
    usn = (headers.get("usn") or "").strip()
    if not usn:
        return None

    low = usn.lower()
    if "uuid:" in low:
        try:
            start = low.index("uuid:")
            end = low.find("::", start)
            token = usn[start:end] if end != -1 else usn[start:]
            token = token.strip()
            return token or usn
        except Exception:
            return usn

    return usn


def _fetch_device_description(location: str, *, timeout_s: float, max_bytes: int) -> bytes | None:
    try:
        req = Request(
            location,
            headers={"Accept": "text/xml, application/xml;q=0.9, */*;q=0.1"},
        )

        with urlopen(req, timeout=timeout_s) as resp:
            try:
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    cl_i = int(cl)
                    if cl_i > max_bytes:
                        _logger.warning(
                            f"[DLNA] LOCATION demasiado grande (Content-Length={cl_i} > {max_bytes}) -> ignore: {location}",
                            always=True,
                        )
                        return None
            except Exception:
                pass

            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                _logger.warning(
                    f"[DLNA] LOCATION excede max_bytes ({max_bytes}) -> ignore: {location}",
                    always=True,
                )
                return None
            return data

    except Exception as exc:  # pragma: no cover
        _logger.warning(f"[DLNA] No se pudo descargar LOCATION {location}: {exc!r}", always=True)
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


# ============================================================================
# API pública
# ============================================================================

def discover_dlna_devices(
    timeout: float | None = None,
    st: str | None = None,
    mx: int | None = None,
) -> list[DLNADevice]:
    timeout_s: float = float(timeout) if timeout is not None else float(
        _cfg_get("DLNA_DISCOVERY_TIMEOUT_SECONDS", _DEFAULT_DISCOVERY_TIMEOUT)
    )
    st_s: str = str(st) if st is not None else str(_cfg_get("DLNA_DISCOVERY_ST", _DEFAULT_SSDP_ST))
    mx_i: int = int(mx) if mx is not None else int(_cfg_get("DLNA_DISCOVERY_MX", _DEFAULT_SSDP_MX))

    device_desc_timeout_s: float = float(_cfg_get("DLNA_DEVICE_DESC_TIMEOUT_SECONDS", _DEFAULT_DEVICE_DESC_TIMEOUT))
    device_desc_max_bytes: int = int(_cfg_get("DLNA_DEVICE_DESC_MAX_BYTES", _DEFAULT_DEVICE_DESC_MAX_BYTES))

    deny_tokens: list[str] = list(_cfg_get("DLNA_DISCOVERY_DENY_TOKENS", _DEFAULT_DENY_TOKENS))
    allow_hint_tokens: list[str] = list(_cfg_get("DLNA_DISCOVERY_ALLOW_HINT_TOKENS", _DEFAULT_ALLOW_HINT_TOKENS))

    max_responses: int = int(_cfg_get("DLNA_DISCOVERY_MAX_RESPONSES", _DEFAULT_DISCOVERY_MAX_RESPONSES))
    max_responses_per_host: int = int(
        _cfg_get("DLNA_DISCOVERY_MAX_RESPONSES_PER_HOST", _DEFAULT_DISCOVERY_MAX_RESPONSES_PER_HOST)
    )

    if timeout_s < 0.2:
        timeout_s = 0.2
    if mx_i < 1:
        mx_i = 1
    if mx_i > 10:
        mx_i = 10
    if device_desc_timeout_s < 0.2:
        device_desc_timeout_s = 0.2
    if device_desc_max_bytes < 16_384:
        device_desc_max_bytes = 16_384
    if max_responses < 64:
        max_responses = 64
    if max_responses_per_host < 32:
        max_responses_per_host = 32

    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"ST: {st_s}\r\n"
        f"MX: {mx_i}\r\n"
        "\r\n"
    )

    _log_debug(
        "Discovery start "
        f"timeout={timeout_s}s st={st_s!r} mx={mx_i} "
        f"desc_timeout={device_desc_timeout_s}s max_bytes={device_desc_max_bytes} "
        f"max_responses={max_responses} max_responses_per_host={max_responses_per_host}"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    except Exception:
        pass

    try:
        sock.settimeout(timeout_s)
        sock.sendto(msg.encode("utf-8"), SSDP_ADDR)

        start = time.monotonic()

        found_by_location: dict[str, DLNADevice] = {}
        found_by_device_id: dict[str, DLNADevice] = {}

        bad_locations: set[str] = set()
        responses_by_host: dict[str, int] = {}

        processed = 0

        while True:
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                break

            if processed >= max_responses:
                _logger.warning(
                    f"[DLNA] Discovery alcanzó max_responses={max_responses}; cortando para evitar bucle/flood.",
                    always=True,
                )
                break

            sock.settimeout(remaining)

            try:
                data, addr = sock.recvfrom(65507)
            except socket.timeout:
                break
            except Exception as exc:  # pragma: no cover
                _logger.warning(f"[DLNA] Error recibiendo SSDP: {exc!r}", always=True)
                break

            processed += 1

            src_ip = addr[0] if isinstance(addr, tuple) and addr else "?"
            responses_by_host[src_ip] = responses_by_host.get(src_ip, 0) + 1
            if responses_by_host[src_ip] > max_responses_per_host:
                if responses_by_host[src_ip] == max_responses_per_host + 1:
                    _logger.warning(
                        f"[DLNA] Host {src_ip} supera max_responses_per_host={max_responses_per_host}; "
                        "ignorando más respuestas de este host para evitar flood.",
                        always=True,
                    )
                continue

            headers = _parse_ssdp_response(data)
            location_raw = headers.get("location")
            if not location_raw:
                continue

            normalized_location = _normalize_location(location_raw)
            if not normalized_location:
                continue

            if normalized_location in bad_locations:
                continue

            parsed = urlparse(normalized_location)
            if not parsed.hostname:
                _log_debug(
                    f"SSDP response sin hostname en LOCATION: raw={location_raw!r} normalized={normalized_location!r} from={addr!r}"
                )
                bad_locations.add(normalized_location)
                continue

            hdr_blob = _headers_text_for_filter(headers, normalized_location)
            if _should_ignore_by_tokens(hdr_blob, deny_tokens):
                _log_debug(f"Ignorado por deny-tokens (ruido UPnP): {normalized_location} from={src_ip}")
                bad_locations.add(normalized_location)
                continue

            if _has_any_allow_hint(hdr_blob, allow_hint_tokens):
                _log_debug(f"Allow-hint detectado: {normalized_location}")

            host = parsed.hostname
            port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)

            device_id = _extract_device_id_from_headers(headers)
            if device_id and device_id in found_by_device_id:
                continue

            if normalized_location in found_by_location:
                continue

            xml_data = _fetch_device_description(
                normalized_location,
                timeout_s=device_desc_timeout_s,
                max_bytes=device_desc_max_bytes,
            )
            if xml_data is None:
                bad_locations.add(normalized_location)
                _log_debug(f"LOCATION fetch falló o fue limitado: {normalized_location!r}")
                continue

            if not _has_content_directory(xml_data):
                bad_locations.add(normalized_location)
                _log_debug(f"Ignorando sin ContentDirectory: {normalized_location}")
                continue

            friendly = _extract_friendly_name(xml_data, fallback=normalized_location)

            dev = DLNADevice(
                friendly_name=friendly,
                location=normalized_location,
                host=host,
                port=port,
                device_id=device_id,
            )

            found_by_location[normalized_location] = dev
            if device_id:
                found_by_device_id[device_id] = dev

        devices = list(found_by_location.values())

        if _is_silent_mode():
            _log_debug(f"Descubiertos {len(devices)} servidor(es) DLNA/UPnP.")
        else:
            _logger.info(f"[DLNA] Descubiertos {len(devices)} servidor(es) DLNA/UPnP.", always=True)

        return devices

    finally:
        try:
            sock.close()
        except Exception:
            pass