from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest


@dataclass(slots=True)
class FakeHTTPResponse:
    payload: bytes

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None


@dataclass(slots=True)
class URLCall:
    url: object
    timeout: float | None


class URLOpenMock:
    """
    Minimal urlopen mock with programmable routing.

    Records calls and returns FakeHTTPResponse(payload) per route.
    """

    def __init__(self, router: Callable[[object], bytes]) -> None:
        self._router = router
        self.calls: list[URLCall] = []

    def __call__(self, url: object, timeout: float | None = None) -> FakeHTTPResponse:
        self.calls.append(URLCall(url=url, timeout=timeout))
        payload = self._router(url)
        return FakeHTTPResponse(payload=payload)


def build_soap_envelope(result_xml: str, *, total_matches: int | None) -> bytes:
    total = (
        f"<TotalMatches>{total_matches}</TotalMatches>"
        if total_matches is not None
        else ""
    )
    return (
        "<?xml version='1.0'?>"
        "<s:Envelope xmlns:s='http://schemas.xmlsoap.org/soap/envelope/'>"
        "<s:Body>"
        "<u:BrowseResponse xmlns:u='urn:schemas-upnp-org:service:ContentDirectory:1'>"
        f"<Result>{result_xml}</Result>"
        f"{total}"
        "</u:BrowseResponse>"
        "</s:Body>"
        "</s:Envelope>"
    ).encode("utf-8")


@pytest.fixture()
def didl_video_item_xml() -> str:
    return (
        '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" '
        'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/">'
        '<item id="i1">'
        "<dc:title>My Movie</dc:title>"
        "<dc:date>1999-01-01</dc:date>"
        '<res protocolInfo="http-get:*:video/mp4:*" size="123">http://example/video.mp4</res>'
        "<upnp:class>object.item.videoItem.movie</upnp:class>"
        "</item>"
        "</DIDL-Lite>"
    )


@pytest.fixture()
def didl_container_and_item_xml(didl_video_item_xml: str) -> str:
    inner_item = didl_video_item_xml[
        didl_video_item_xml.find("<item") : didl_video_item_xml.rfind("</item>") + 7
    ]
    return (
        '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" '
        'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/">'
        '<container id="c1"><dc:title>Movies</dc:title></container>'
        f"{inner_item}"
        "</DIDL-Lite>"
    )


@pytest.fixture()
def device_description_xml() -> bytes:
    return (
        "<?xml version='1.0'?>"
        "<root xmlns='urn:schemas-upnp-org:device-1-0'>"
        "<device>"
        "<friendlyName>Test DLNA</friendlyName>"
        "<serviceList>"
        "<service>"
        "<serviceType>urn:schemas-upnp-org:service:ContentDirectory:1</serviceType>"
        "<controlURL>/ctl/ContentDir</controlURL>"
        "</service>"
        "</serviceList>"
        "</device>"
        "</root>"
    ).encode("utf-8")
