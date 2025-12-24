from __future__ import annotations

"""
backend/dlna_discovery.py

Descubrimiento de dispositivos DLNA/UPnP por SSDP (M-SEARCH).

Problemas reales observados (especialmente con Plex u otros MediaServers)
------------------------------------------------------------------------
1) Bucle "infinito" o runs eternos:
   - Causa típica A: usar time.time() y el reloj del sistema se ajusta hacia atrás
     (NTP, VM, sleep/wake). El cálculo de remaining no llega a 0.
     ✅ Fix: usar time.monotonic().

   - Causa típica B: flood de respuestas SSDP (mismo host o múltiples interfaces).
     ✅ Fix: fuse global `DLNA_DISCOVERY_MAX_RESPONSES`.

2) Re-análisis múltiple del mismo servidor (mismo “device”) con LOCATION distintos:
   - Muchos devices (y Plex en algunas setups) pueden responder con variaciones
     de LOCATION (query, path alternativo, puertos, etc.)
   - Deduplicar solo por LOCATION no siempre es suficiente.
   ✅ Fix: deduplicar también por identidad UPnP (USN/UDN/UUID) cuando está disponible.

3) Reintentos inútiles sobre LOCATION que ya falló en este mismo run:
   - Timeout / XML inválido / no ContentDirectory / payload enorme.
   ✅ Fix: negative-cache in-memory por run.

Filosofía de logs (alineada con backend/logger.py)
--------------------------------------------------
- Nunca romper el flujo por logging.
- Ruido / filtros / detalles → solo en DEBUG_MODE.
- Señales importantes → warning/info con always=True cuando aplica.
"""

from dataclasses import dataclass
import socket
import time
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from backend import logger as _logger


# SSDP multicast estándar
SSDP_ADDR: Tuple[str, int] = ("239.255.255.250", 1900)

# Defaults (se sobre-escriben si config está presente)
_DEFAULT_SSDP_ST = "ssdp:all"
_DEFAULT_SSDP_MX = 2
_DEFAULT_DISCOVERY_TIMEOUT = 3.0
_DEFAULT_DEVICE_DESC_TIMEOUT = 3.0
_DEFAULT_DEVICE_DESC_MAX_BYTES = 2_000_000

# Anti-bucle: cap de respuestas procesadas por “run” de discovery.
_DEFAULT_DISCOVERY_MAX_RESPONSES = 2048

# Anti-flood por host (si un solo IP responde sin parar).
# Si no quieres este comportamiento, puedes setearlo muy alto en config.
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

    Nota:
    - `location` aquí es la LOCATION normalizada, usada para GET y para dedup por URL.
    - `device_id` es un identificador estable si se pudo extraer (USN/UDN/UUID).
      Ayuda a deduplicar aunque cambie LOCATION.
    """
    friendly_name: str
    location: str
    host: str
    port: int
    device_id: str | None = None


# ============================================================================
# Logging controlado por modos (sin depender de imports directos a config)
# ============================================================================

def _safe_get_cfg():
    """Devuelve backend.config si ya está importado (evita dependencias circulares)."""
    import sys
    return sys.modules.get("backend.config")


def _cfg_get(name: str, default):
    """
    Lee un atributo de backend.config (si existe) sin romper nunca.
    """
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
    """
    Debug contextual:
    - DEBUG_MODE=False → no-op
    - DEBUG_MODE=True:
        * SILENT_MODE=True  -> progress
        * SILENT_MODE=False -> info
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
        if not _is_silent_mode():
            try:
                print(text)
            except Exception:
                pass


# ============================================================================
# SSDP parsing + URL/XML helpers
# ============================================================================

def _parse_ssdp_response(data: bytes) -> Dict[str, str]:
    """
    Parsea una respuesta SSDP (cabeceras HTTP-like) a dict lower-case.
    Devuelve {} si no parece parseable.
    """
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return {}

    lines = text.split("\r\n")
    if not lines:
        return {}

    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    return headers


def _normalize_location(location: str) -> str:
    """
    Normaliza LOCATION para deduplicación consistente.

    Reglas:
    - scheme y host a lower
    - elimina fragment (#...)
    - conserva query (por compatibilidad; si tu red lo necesita, se puede ignorar)
    - path vacío -> "/"
    - si hay port explícito, se conserva; si no, no se añade
    """
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
    fragment = ""

    return urlunparse((scheme, netloc, path, p.params, p.query, fragment))


def _headers_text_for_filter(headers: Dict[str, str], location: str) -> str:
    """
    Construye un “blob” textual para filtros deny/allow.
    """
    parts = [
        location,
        headers.get("st", ""),
        headers.get("usn", ""),
        headers.get("server", ""),
        headers.get("nt", ""),
    ]
    return " | ".join(p for p in parts if p).lower()


def _should_ignore_by_tokens(text: str, deny_tokens: List[str]) -> bool:
    if not deny_tokens:
        return False
    for tok in deny_tokens:
        if tok and tok in text:
            return True
    return False


def _has_any_allow_hint(text: str, allow_tokens: List[str]) -> bool:
    if not allow_tokens:
        return False
    for tok in allow_tokens:
        if tok and tok in text:
            return True
    return False


def _extract_device_id_from_headers(headers: Dict[str, str]) -> str | None:
    """
    Intenta obtener un identificador estable del device a partir de headers SSDP.

    Usualmente:
      USN: uuid:XXXXXXXX-....::urn:schemas-upnp-org:device:MediaServer:1

    Estrategia:
    - Preferimos USN si existe, y extraemos la parte "uuid:..."
    - Si no, usamos USN completo (normalizado) como fallback.

    Esto es clave para deduplicar cuando LOCATION varía.
    """
    usn = (headers.get("usn") or "").strip()
    if not usn:
        return None

    low = usn.lower()

    # Caso típico: "uuid:....::..."
    # Nos quedamos con el tramo uuid:...
    if "uuid:" in low:
        try:
            # buscamos desde "uuid:" hasta antes de "::" (si existe)
            start = low.index("uuid:")
            end = low.find("::", start)
            token = usn[start:end] if end != -1 else usn[start:]
            token = token.strip()
            return token or usn
        except Exception:
            return usn

    return usn


def _fetch_device_description(location: str, *, timeout_s: float, max_bytes: int) -> bytes | None:
    """
    Descarga el XML de descripción del dispositivo (LOCATION) con límites fuertes.
    """
    try:
        req = Request(
            location,
            headers={"Accept": "text/xml, application/xml;q=0.9, */*;q=0.1"},
        )

        with urlopen(req, timeout=timeout_s) as resp:
            # Content-Length defensivo
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


# ============================================================================
# API pública
# ============================================================================

def discover_dlna_devices(
    timeout: float | None = None,
    st: str | None = None,
    mx: int | None = None,
) -> List[DLNADevice]:
    """
    Descubre servidores DLNA/UPnP en la red mediante SSDP.

    Devuelve una lista deduplicada:
    - por LOCATION normalizada (evita repeticiones obvias)
    - y por device_id (USN/UUID) cuando existe (evita repeticiones “Plex-style”)

    Además:
    - usa monotonic() para evitar bucles por ajustes de reloj
    - aplica fuse global de respuestas y fuse por host (anti-flood)
    - negative-cache por run para no reintentar LOCATION fallidas
    """
    timeout_s: float = float(timeout) if timeout is not None else float(
        _cfg_get("DLNA_DISCOVERY_TIMEOUT_SECONDS", _DEFAULT_DISCOVERY_TIMEOUT)
    )
    st_s: str = str(st) if st is not None else str(_cfg_get("DLNA_DISCOVERY_ST", _DEFAULT_SSDP_ST))
    mx_i: int = int(mx) if mx is not None else int(_cfg_get("DLNA_DISCOVERY_MX", _DEFAULT_SSDP_MX))

    device_desc_timeout_s: float = float(_cfg_get("DLNA_DEVICE_DESC_TIMEOUT_SECONDS", _DEFAULT_DEVICE_DESC_TIMEOUT))
    device_desc_max_bytes: int = int(_cfg_get("DLNA_DEVICE_DESC_MAX_BYTES", _DEFAULT_DEVICE_DESC_MAX_BYTES))

    deny_tokens: List[str] = list(_cfg_get("DLNA_DISCOVERY_DENY_TOKENS", _DEFAULT_DENY_TOKENS))
    allow_hint_tokens: List[str] = list(_cfg_get("DLNA_DISCOVERY_ALLOW_HINT_TOKENS", _DEFAULT_ALLOW_HINT_TOKENS))

    max_responses: int = int(_cfg_get("DLNA_DISCOVERY_MAX_RESPONSES", _DEFAULT_DISCOVERY_MAX_RESPONSES))
    max_responses_per_host: int = int(
        _cfg_get("DLNA_DISCOVERY_MAX_RESPONSES_PER_HOST", _DEFAULT_DISCOVERY_MAX_RESPONSES_PER_HOST)
    )

    # Defensive caps locales
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

        # Dedup “fuerte”
        found_by_location: Dict[str, DLNADevice] = {}
        found_by_device_id: Dict[str, DLNADevice] = {}

        # Negative-cache por run:
        # - locations que ya fallaron / no son DLNA / no ContentDirectory.
        bad_locations: set[str] = set()

        # Anti-flood por IP
        responses_by_host: Dict[str, int] = {}

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
                _logger.warning(f"[DLNA] Error recibiendo SSDP: {exc}")
                break

            processed += 1

            src_ip = addr[0] if isinstance(addr, tuple) and addr else "?"
            responses_by_host[src_ip] = responses_by_host.get(src_ip, 0) + 1
            if responses_by_host[src_ip] > max_responses_per_host:
                # No cortamos el run entero: solo dejamos de procesar a este host.
                # Esto evita que un Plex “monopolice” discovery en redes ruidosas.
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

            # Negative-cache: si ya sabemos que esta URL no sirve en este run, no insistimos.
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

            # ✅ Dedup por device id (USN/UUID) además de por LOCATION
            device_id = _extract_device_id_from_headers(headers)
            if device_id and device_id in found_by_device_id:
                # Ya tenemos este dispositivo, aunque cambie LOCATION.
                # Si quieres, aquí podrías “actualizar” location si te interesa, pero normalmente no.
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