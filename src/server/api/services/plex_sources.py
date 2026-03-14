from __future__ import annotations

import hashlib
import socket
import urllib.parse
import uuid
import webbrowser
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from xml.etree import ElementTree

import requests

from shared.runtime_profiles import (
    PROJECT_DIR,
    SourceProfile,
    build_profile_from_discovery,
)

_PRODUCT_NAME = "Analiza Movies"
_PIN_URL = "https://plex.tv/api/v2/pins"
_RESOURCES_URL = "https://clients.plex.tv/api/v2/resources"
_REQUEST_TIMEOUT_S = 10.0
_LOCAL_SCAN_TIMEOUT_S = 0.35
_LOCAL_SCAN_WORKERS = 32
_LOCAL_SCAN_PORT = 32400


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client_identifier() -> str:
    digest = hashlib.sha256(str(PROJECT_DIR).encode("utf-8")).hexdigest()
    return digest[:24]


def _plex_headers(
    *,
    accept: str = "application/json",
    token: str | None = None,
) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "X-Plex-Product": _PRODUCT_NAME,
        "X-Plex-Client-Identifier": _client_identifier(),
    }
    clean_token = (token or "").strip()
    if clean_token:
        headers["X-Plex-Token"] = clean_token
    return headers


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _local_ipv4_candidates() -> list[str]:
    hosts: set[str] = set()
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        for info in infos:
            addr = info[4][0]
            if isinstance(addr, str) and addr and not addr.startswith("127."):
                hosts.add(addr)
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        addr = sock.getsockname()[0]
        sock.close()
        if addr and not addr.startswith("127."):
            hosts.add(addr)
    except Exception:
        pass

    return sorted(hosts)


def _local_scan_targets() -> list[str]:
    targets: set[str] = set()
    for host in _local_ipv4_candidates():
        parts = host.split(".")
        if len(parts) != 4:
            continue
        prefix = ".".join(parts[:3])
        for suffix in range(1, 255):
            targets.add(f"{prefix}.{suffix}")
    return sorted(targets)


def _scan_local_plex_host(host: str) -> dict[str, Any] | None:
    url = f"http://{host}:{_LOCAL_SCAN_PORT}/identity"
    try:
        response = requests.get(url, timeout=_LOCAL_SCAN_TIMEOUT_S)
        response.raise_for_status()
    except Exception:
        return None

    try:
        root = ElementTree.fromstring(response.text)
    except Exception:
        return None

    machine_identifier = root.attrib.get("machineIdentifier") or None
    version = root.attrib.get("version") or None
    return {
        "name": f"Plex {host}",
        "source_type": "plex",
        "host": host,
        "port": _LOCAL_SCAN_PORT,
        "base_url": f"http://{host}",
        "machine_identifier": machine_identifier,
        "plex_token": None,
        "discovery": "local_scan",
        "local": True,
        "relay": False,
        "version": version,
    }


def _clean_port(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return fallback
    return fallback


def _local_scan_plex_servers() -> list[dict[str, Any]]:
    targets = _local_scan_targets()
    if not targets:
        return []

    found: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=_LOCAL_SCAN_WORKERS) as executor:
        futures = {
            executor.submit(_scan_local_plex_host, host): host for host in targets
        }
        for future in as_completed(futures):
            payload = future.result()
            if payload is not None:
                found.append(payload)

    found.sort(key=lambda item: ((item.get("name") or ""), (item.get("host") or "")))
    return found


def _preferred_connection(connections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not connections:
        return None

    def _rank(item: dict[str, Any]) -> tuple[int, int, str]:
        local = _boolish(item.get("local"))
        relay = _boolish(item.get("relay"))
        uri = str(item.get("uri") or "")
        return (0 if local else 1, 1 if relay else 0, uri)

    return sorted(connections, key=_rank)[0]


def _parse_resource_devices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    media = payload.get("MediaContainer")
    if isinstance(media, dict):
        devices = media.get("Device")
    else:
        devices = payload.get("Device")
    if isinstance(devices, list):
        return [item for item in devices if isinstance(item, dict)]
    if isinstance(devices, dict):
        return [devices]
    return []


def _discover_plex_from_account(user_token: str) -> list[dict[str, Any]]:
    response = requests.get(
        _RESOURCES_URL,
        params={
            "includeHttps": 1,
            "includeRelay": 1,
            "includeIPv6": 1,
        },
        headers=_plex_headers(token=user_token),
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()

    out: list[dict[str, Any]] = []
    for device in _parse_resource_devices(payload):
        provides_raw = str(device.get("provides") or "")
        provides = {chunk.strip().lower() for chunk in provides_raw.split(",") if chunk}
        if "server" not in provides:
            continue

        connections_obj = device.get("Connection")
        if isinstance(connections_obj, list):
            connections = [item for item in connections_obj if isinstance(item, dict)]
        elif isinstance(connections_obj, dict):
            connections = [connections_obj]
        else:
            connections = []

        preferred = _preferred_connection(connections)
        if preferred is None:
            continue

        uri = str(preferred.get("uri") or "").strip()
        parsed = urllib.parse.urlparse(uri)
        address = str(preferred.get("address") or "").strip() or (parsed.hostname or "")
        port_num = _clean_port(preferred.get("port"), parsed.port or _LOCAL_SCAN_PORT)
        scheme = (
            parsed.scheme or str(preferred.get("protocol") or "http").strip() or "http"
        )
        host = address or parsed.hostname or ""
        if not host:
            continue

        base_url = f"{scheme}://{parsed.hostname or host}"
        out.append(
            {
                "name": str(device.get("name") or host),
                "source_type": "plex",
                "host": host,
                "port": port_num,
                "base_url": base_url,
                "machine_identifier": str(device.get("clientIdentifier") or "").strip()
                or None,
                "plex_token": str(device.get("accessToken") or "").strip()
                or user_token,
                "discovery": "plex_account",
                "local": _boolish(preferred.get("local")),
                "relay": _boolish(preferred.get("relay")),
                "uri": uri or None,
            }
        )

    out.sort(key=lambda item: (0 if item.get("local") else 1, item.get("name") or ""))
    return out


def merge_plex_servers(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            machine_identifier = str(item.get("machine_identifier") or "").strip()
            host = str(item.get("host") or "").strip()
            port = item.get("port")
            key = machine_identifier or f"{host}:{port}"
            if not key:
                continue
            current = merged.get(key, {})
            updated = dict(current)
            for key_name, value in item.items():
                if value in (None, "", []):
                    continue
                if key_name in {"local"}:
                    updated[key_name] = bool(value) or bool(updated.get(key_name))
                else:
                    updated[key_name] = value
            merged[key] = updated

    return sorted(
        merged.values(),
        key=lambda item: (0 if item.get("local") else 1, str(item.get("name") or "")),
    )


def discover_plex_servers(user_token: str | None = None) -> list[dict[str, Any]]:
    local_servers = _local_scan_plex_servers()
    if not (user_token or "").strip():
        return sanitize_plex_servers(local_servers)

    try:
        account_servers = _discover_plex_from_account(str(user_token).strip())
    except Exception:
        account_servers = []

    return sanitize_plex_servers(merge_plex_servers(account_servers, local_servers))


@dataclass(slots=True)
class PlexAuthSession:
    session_id: str
    pin_id: int
    code: str
    auth_url: str
    created_at: str
    status: str = "pending"
    user_token: str | None = None
    error: str | None = None
    servers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["servers"] = self.servers or []
        return payload


_AUTH_LOCK = Lock()
_AUTH_SESSIONS: dict[str, PlexAuthSession] = {}
_PROFILE_TOKEN_LOCK = Lock()
_PROFILE_TOKENS: dict[str, str] = {}


def remember_profile_token(profile_id: str | None, token: str | None) -> None:
    clean_profile_id = str(profile_id or "").strip()
    clean_token = str(token or "").strip()
    if not clean_profile_id or not clean_token:
        return
    with _PROFILE_TOKEN_LOCK:
        _PROFILE_TOKENS[clean_profile_id] = clean_token


def resolve_profile_token(profile: SourceProfile) -> str | None:
    direct_token = str(profile.plex_token or "").strip()
    if direct_token:
        return direct_token
    with _PROFILE_TOKEN_LOCK:
        stored_token = _PROFILE_TOKENS.get(profile.id)
    return None if stored_token is None else stored_token.strip() or None


def _register_server_token(server: dict[str, Any]) -> None:
    token = str(server.get("plex_token") or "").strip()
    if not token:
        return
    profile = build_profile_from_discovery(
        source_type="plex",
        name=str(server.get("name") or "Origen Plex").strip() or "Origen Plex",
        host=server.get("host"),
        port=server.get("port"),
        base_url=server.get("base_url"),
        location=server.get("location"),
        device_id=server.get("device_id"),
        machine_identifier=server.get("machine_identifier"),
        profile_id=server.get("id"),
    )
    remember_profile_token(profile.id, token)


def sanitize_plex_servers(servers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in servers:
        server = dict(item)
        _register_server_token(server)
        server["plex_token"] = None
        sanitized.append(server)
    return sanitized


def start_plex_auth_session(*, open_browser: bool = True) -> dict[str, Any]:
    response = requests.post(
        _PIN_URL,
        params={"strong": "true"},
        headers=_plex_headers(),
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()

    pin_id = int(payload["id"])
    code = str(payload["code"]).strip()
    auth_url = "https://app.plex.tv/auth#?" + urllib.parse.urlencode(
        {
            "clientID": _client_identifier(),
            "code": code,
            "forwardUrl": "https://app.plex.tv/desktop",
            "context[device][product]": _PRODUCT_NAME,
        }
    )

    session = PlexAuthSession(
        session_id=uuid.uuid4().hex,
        pin_id=pin_id,
        code=code,
        auth_url=auth_url,
        created_at=_now_iso(),
    )

    with _AUTH_LOCK:
        _AUTH_SESSIONS[session.session_id] = session

    browser_opened = False
    if open_browser:
        try:
            browser_opened = bool(webbrowser.open(auth_url))
        except Exception:
            browser_opened = False

    payload_out = session.to_dict()
    payload_out["browser_opened"] = browser_opened
    return payload_out


def poll_plex_auth_session(session_id: str) -> dict[str, Any]:
    with _AUTH_LOCK:
        session = _AUTH_SESSIONS.get(session_id)

    if session is None:
        raise KeyError(session_id)

    if session.status == "complete":
        return session.to_dict()

    response = requests.get(
        f"{_PIN_URL}/{session.pin_id}",
        headers=_plex_headers(),
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()

    token = str(payload.get("authToken") or "").strip() or None
    if token is None:
        return session.to_dict()

    session.status = "complete"
    session.user_token = token
    session.servers = discover_plex_servers(token)

    with _AUTH_LOCK:
        _AUTH_SESSIONS[session_id] = session

    return session.to_dict()


def get_plex_auth_session(session_id: str) -> dict[str, Any] | None:
    with _AUTH_LOCK:
        session = _AUTH_SESSIONS.get(session_id)
    return None if session is None else session.to_dict()
