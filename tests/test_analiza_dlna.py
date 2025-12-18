from __future__ import annotations

from typing import Callable

import pytest
import xml.etree.ElementTree as ET

import backend.analiza_dlna as dlna
from backend.dlna_discovery import DLNADevice
from tests.conftest import URLOpenMock, build_soap_envelope


def _device(*, friendly_name: str = "Test") -> DLNADevice:
    return DLNADevice(
        host="192.168.1.2",
        port=8200,
        friendly_name=friendly_name,
        location="http://192.168.1.2:8200/desc.xml",
    )


class TestUnitHelpers:
    def test_xml_text_none(self) -> None:
        assert dlna._xml_text(None) is None

    def test_xml_text_strips_and_none_for_empty(self) -> None:
        e = ET.Element("x")
        e.text = "   "
        assert dlna._xml_text(e) is None

        e.text = "  ABC  "
        assert dlna._xml_text(e) == "ABC"

    def test_is_plex_server_case_insensitive(self) -> None:
        assert dlna._is_plex_server(_device(friendly_name="Plex Media Server")) is True
        assert dlna._is_plex_server(_device(friendly_name="PLEX MEDIA SERVER")) is True
        assert dlna._is_plex_server(_device(friendly_name="Jellyfin")) is False

    def test_is_likely_video_root_title_positive_negative_mix(self) -> None:
        assert dlna._is_likely_video_root_title("Vídeos") is True
        assert dlna._is_likely_video_root_title("Videos") is True
        assert dlna._is_likely_video_root_title("Photos") is False
        assert dlna._is_likely_video_root_title("Video & Photos") is False
        assert dlna._is_likely_video_root_title("   ") is False

    def test_folder_browse_container_score(self) -> None:
        assert dlna._folder_browse_container_score("By Folder") == 100
        assert dlna._folder_browse_container_score("Browse Folders") == 100
        assert dlna._folder_browse_container_score("Examinar carpetas") == 100
        assert dlna._folder_browse_container_score("Movies") == 0
        assert dlna._folder_browse_container_score("   ") == 0

    def test_is_plex_virtual_container_title_defaults_true_for_empty(self) -> None:
        assert dlna._is_plex_virtual_container_title("") is True
        assert dlna._is_plex_virtual_container_title("   ") is True

    def test_is_plex_virtual_container_title_tokens(self) -> None:
        assert dlna._is_plex_virtual_container_title("Recently Added") is True
        assert dlna._is_plex_virtual_container_title("By Genre") is True
        assert dlna._is_plex_virtual_container_title("Movies") is False

    def test_parse_multi_selection_valid_and_unique_preserve_order(self) -> None:
        assert dlna._parse_multi_selection("1,2,3", 3) == [1, 2, 3]
        assert dlna._parse_multi_selection(" 1 , 2 ", 5) == [1, 2]
        assert dlna._parse_multi_selection("1,2,2,3", 3) == [1, 2, 3]

    def test_parse_multi_selection_invalid(self) -> None:
        assert dlna._parse_multi_selection("", 3) is None
        assert dlna._parse_multi_selection(" , , ", 3) is None
        assert dlna._parse_multi_selection("a,1", 3) is None
        assert dlna._parse_multi_selection("0", 3) is None
        assert dlna._parse_multi_selection("4", 3) is None

    def test_extract_year_from_date(self) -> None:
        assert dlna._extract_year_from_date("1999-01-01") == 1999
        assert dlna._extract_year_from_date("1999") == 1999
        assert dlna._extract_year_from_date("1899-01-01") is None
        assert dlna._extract_year_from_date("2101-01-01") is None
        assert dlna._extract_year_from_date("abcd") is None

    def test_is_video_item_by_class_or_protocolinfo(self) -> None:
        item = ET.Element("item")
        cls = ET.SubElement(item, "upnp:class")
        cls.text = "object.item.videoItem.movie"
        assert dlna._is_video_item(item) is True

        item2 = ET.Element("item")
        res = ET.SubElement(item2, "res")
        res.attrib["protocolInfo"] = "http-get:*:video/mp4:*"
        assert dlna._is_video_item(item2) is True

        item3 = ET.Element("item")
        ET.SubElement(item3, "dc:title").text = "No"
        assert dlna._is_video_item(item3) is False

    def test_extract_container_title_and_id(self) -> None:
        c = ET.Element("container")
        c.attrib["id"] = "c1"
        ET.SubElement(c, "dc:title").text = "Movies"
        out = dlna._extract_container_title_and_id(c)
        assert out is not None
        assert out.object_id == "c1"
        assert out.title == "Movies"

        c2 = ET.Element("container")
        ET.SubElement(c2, "dc:title").text = "Movies"
        assert dlna._extract_container_title_and_id(c2) is None

        c3 = ET.Element("container")
        c3.attrib["id"] = "c1"
        assert dlna._extract_container_title_and_id(c3) is None

    def test_extract_video_item_parses_fields(self) -> None:
        item = ET.Element("item")
        ET.SubElement(item, "dc:title").text = "My Movie"
        ET.SubElement(item, "dc:date").text = "1999-01-01"
        res = ET.SubElement(item, "res")
        res.text = "http://example/video.mp4"
        res.attrib["size"] = "123"

        out = dlna._extract_video_item(item)
        assert out is not None
        assert out.title == "My Movie"
        assert out.year == 1999
        assert out.resource_url == "http://example/video.mp4"
        assert out.size_bytes == 123

    def test_extract_video_item_requires_title_and_res(self) -> None:
        item = ET.Element("item")
        ET.SubElement(item, "dc:date").text = "1999-01-01"
        assert dlna._extract_video_item(item) is None

        item2 = ET.Element("item")
        ET.SubElement(item2, "dc:title").text = "X"
        assert dlna._extract_video_item(item2) is None

    def test_extract_video_item_size_non_numeric_is_none(self) -> None:
        item = ET.Element("item")
        ET.SubElement(item, "dc:title").text = "My Movie"
        res = ET.SubElement(item, "res")
        res.text = "http://example/video.mp4"
        res.attrib["size"] = "12a"

        out = dlna._extract_video_item(item)
        assert out is not None
        assert out.size_bytes is None


class TestSoap:
    def test_fetch_xml_root_ok(
        self, monkeypatch: pytest.MonkeyPatch, device_description_xml: bytes
    ) -> None:
        mock = URLOpenMock(lambda _url: device_description_xml)
        monkeypatch.setattr(dlna, "urlopen", mock)

        root = dlna._fetch_xml_root("http://device/desc.xml")
        assert root is not None
        assert isinstance(root, ET.Element)
        assert mock.calls and mock.calls[0].timeout == 5.0

    def test_fetch_xml_root_download_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raising(_url: object, timeout: float | None = None) -> object:  # pragma: no cover
            raise OSError("boom")

        monkeypatch.setattr(dlna, "urlopen", raising)
        assert dlna._fetch_xml_root("http://device/desc.xml") is None

    def test_find_content_directory_endpoints(
        self, monkeypatch: pytest.MonkeyPatch, device_description_xml: bytes
    ) -> None:
        mock = URLOpenMock(lambda _url: device_description_xml)
        monkeypatch.setattr(dlna, "urlopen", mock)

        endpoints = dlna._find_content_directory_endpoints("http://192.168.1.2:8200/desc.xml")
        assert endpoints is not None
        control_url, service_type = endpoints
        assert service_type == "urn:schemas-upnp-org:service:ContentDirectory:1"
        assert control_url == "http://192.168.1.2:8200/ctl/ContentDir"

    def test_soap_browse_direct_children_parses_didl(
        self, monkeypatch: pytest.MonkeyPatch, didl_container_and_item_xml: str
    ) -> None:
        soap_bytes = build_soap_envelope(didl_container_and_item_xml, total_matches=2)

        mock = URLOpenMock(lambda _url: soap_bytes)
        monkeypatch.setattr(dlna, "urlopen", mock)

        out = dlna._soap_browse_direct_children(
            "http://device/ctl",
            "urn:schemas-upnp-org:service:ContentDirectory:1",
            "0",
            0,
            10,
        )
        assert out is not None
        children, total = out
        assert total == 2
        assert len(children) == 2

        req_obj = mock.calls[0].url
        assert hasattr(req_obj, "headers")
        headers = getattr(req_obj, "headers")
        assert any(k.lower() == "soapaction" for k in headers)

    def test_soap_browse_direct_children_fallback_unescape(
        self, monkeypatch: pytest.MonkeyPatch, didl_container_and_item_xml: str
    ) -> None:
        escaped = (
            didl_container_and_item_xml.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
        soap_bytes = build_soap_envelope(escaped, total_matches=None)

        mock = URLOpenMock(lambda _url: soap_bytes)
        monkeypatch.setattr(dlna, "urlopen", mock)

        out = dlna._soap_browse_direct_children(
            "http://device/ctl",
            "urn:schemas-upnp-org:service:ContentDirectory:1",
            "0",
            0,
            10,
        )
        assert out is not None
        children, total = out
        assert len(children) == 2
        assert total == 2

    def test_soap_browse_direct_children_no_result_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        soap = (
            "<?xml version='1.0'?>"
            "<s:Envelope xmlns:s='http://schemas.xmlsoap.org/soap/envelope/'>"
            "<s:Body><u:BrowseResponse xmlns:u='urn:schemas-upnp-org:service:ContentDirectory:1'>"
            "<TotalMatches>0</TotalMatches>"
            "</u:BrowseResponse></s:Body></s:Envelope>"
        ).encode("utf-8")

        mock = URLOpenMock(lambda _url: soap)
        monkeypatch.setattr(dlna, "urlopen", mock)

        out = dlna._soap_browse_direct_children(
            "http://device/ctl",
            "urn:schemas-upnp-org:service:ContentDirectory:1",
            "0",
            0,
            10,
        )
        assert out is None


class TestNavigation:
    def test_list_video_root_containers_filters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = _device()
        roots = [
            dlna._DlnaContainer(object_id="1", title="Music"),
            dlna._DlnaContainer(object_id="2", title="Videos"),
            dlna._DlnaContainer(object_id="3", title="Photos"),
            dlna._DlnaContainer(object_id="4", title="Vídeos"),
        ]

        monkeypatch.setattr(dlna, "_list_root_containers", lambda _d: (roots, ("ctl", "st")))
        out = dlna._list_video_root_containers(dev)
        assert [c.object_id for c in out] == ["2", "4"]

    def test_auto_descend_folder_browse_descends_best_scored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dev = _device()
        start = dlna._DlnaContainer(object_id="root", title="Videos")

        tree: dict[str, list[dlna._DlnaContainer]] = {
            "root": [
                dlna._DlnaContainer(object_id="a", title="Movies"),
                dlna._DlnaContainer(object_id="b", title="By Folder"),
            ],
            "b": [
                dlna._DlnaContainer(object_id="c", title="Browse Folders"),
            ],
            "c": [
                dlna._DlnaContainer(object_id="d", title="Not a folder view"),
            ],
        }

        monkeypatch.setattr(dlna, "_list_child_containers", lambda _d, oid: tree.get(oid, []))

        out = dlna._auto_descend_folder_browse(dev, start)
        assert out.object_id == "c"

    def test_auto_descend_folder_browse_caps_three_levels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dev = _device()
        start = dlna._DlnaContainer(object_id="0", title="Videos")

        def children(_d: DLNADevice, oid: str) -> list[dlna._DlnaContainer]:
            if oid == "0":
                return [dlna._DlnaContainer(object_id="1", title="By Folder")]
            if oid == "1":
                return [dlna._DlnaContainer(object_id="2", title="By Folder")]
            if oid == "2":
                return [dlna._DlnaContainer(object_id="3", title="By Folder")]
            if oid == "3":
                return [dlna._DlnaContainer(object_id="4", title="By Folder")]
            return []

        monkeypatch.setattr(dlna, "_list_child_containers", children)
        out = dlna._auto_descend_folder_browse(dev, start)
        assert out.object_id == "3"


class TestItemsRecursion:
    def test_iter_video_items_recursive_paginates_and_descends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dev = _device()

        monkeypatch.setattr(
            dlna,
            "_find_content_directory_endpoints",
            lambda _loc: ("http://device/ctl", "urn:schemas-upnp-org:service:ContentDirectory:1"),
        )

        calls: list[tuple[str, int, int]] = []

        def browse(
            _ctl: str, _st: str, object_id: str, start: int, count: int
        ) -> tuple[list[ET.Element], int] | None:
            calls.append((object_id, start, count))

            if object_id == "A":
                if start == 0:
                    return [ET.Element("container", {"id": "B"})], 450

                items_out: list[ET.Element] = []
                n = min(200, 450 - start)
                for i in range(n):
                    it = ET.Element("item")
                    ET.SubElement(it, "upnp:class").text = "object.item.videoItem.movie"
                    ET.SubElement(it, "dc:title").text = f"Movie {start + i}"
                    res = ET.SubElement(it, "res")
                    res.text = f"http://example/{start + i}.mp4"
                    items_out.append(it)
                return items_out, 450

            if object_id == "B":
                it = ET.Element("item")
                ET.SubElement(it, "upnp:class").text = "object.item.videoItem.movie"
                ET.SubElement(it, "dc:title").text = "Sub Movie"
                res = ET.SubElement(it, "res")
                res.text = "http://example/sub.mp4"
                return [it], 1

            return [], 0

        monkeypatch.setattr(dlna, "_soap_browse_direct_children", browse)

        out = dlna._iter_video_items_recursive(dev, "A")
        assert len(out) == 451
        assert any(x.title == "Sub Movie" for x in out)

        assert ("A", 0, 200) in calls
        assert ("A", 200, 200) in calls
        assert ("A", 400, 200) in calls
        assert ("B", 0, 200) in calls

    def test_iter_video_items_recursive_breaks_on_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = _device()

        monkeypatch.setattr(
            dlna,
            "_find_content_directory_endpoints",
            lambda _loc: ("http://device/ctl", "urn:schemas-upnp-org:service:ContentDirectory:1"),
        )

        def browse(
            _ctl: str, _st: str, _oid: str, _start: int, _count: int
        ) -> tuple[list[ET.Element], int] | None:
            return None

        monkeypatch.setattr(dlna, "_soap_browse_direct_children", browse)
        assert dlna._iter_video_items_recursive(dev, "A") == []


class TestCliFlows:
    def test_ask_dlna_device_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        devs = [DLNADevice(host="h", port=1, friendly_name="A", location="loc")]
        monkeypatch.setattr(dlna, "discover_dlna_devices", lambda: devs)
        monkeypatch.setattr(dlna, "input", lambda _prompt="": "")
        assert dlna._ask_dlna_device() is None

    def test_ask_dlna_device_invalid_then_select(self, monkeypatch: pytest.MonkeyPatch) -> None:
        devs = [
            DLNADevice(host="h", port=1, friendly_name="A", location="loc"),
            DLNADevice(host="h", port=2, friendly_name="B", location="loc2"),
        ]
        monkeypatch.setattr(dlna, "discover_dlna_devices", lambda: devs)

        answers = iter(["x", "9", "2"])
        monkeypatch.setattr(dlna, "input", lambda _prompt="": next(answers))
        chosen = dlna._ask_dlna_device()
        assert chosen is not None
        assert chosen.friendly_name == "B"

    def test_select_folders_non_plex_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base = dlna._DlnaContainer(object_id="root", title="Videos")
        dev = DLNADevice(host="h", port=1, friendly_name="A", location="loc")

        monkeypatch.setattr(dlna, "input", lambda _prompt="": "0")
        assert dlna._select_folders_non_plex(base, dev) == [base]

    def test_select_folders_non_plex_choose_subset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base = dlna._DlnaContainer(object_id="root", title="Videos")
        dev = DLNADevice(host="h", port=1, friendly_name="A", location="loc")

        folders = [
            dlna._DlnaContainer(object_id="1", title="Movies"),
            dlna._DlnaContainer(object_id="2", title="Kids"),
            dlna._DlnaContainer(object_id="3", title="IgnoreMe"),
        ]

        monkeypatch.setattr(dlna, "EXCLUDE_DLNA_LIBRARIES", {"IgnoreMe"})
        monkeypatch.setattr(dlna, "_list_child_containers", lambda _d, _oid: folders)

        answers = iter(["1", "1,2"])
        monkeypatch.setattr(dlna, "input", lambda _prompt="": next(answers))

        selected = dlna._select_folders_non_plex(base, dev)
        assert selected is not None
        assert [c.title for c in selected] == ["Movies", "Kids"]

    def test_select_folders_plex_filters_virtual(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base = dlna._DlnaContainer(object_id="root", title="Videos")
        dev = DLNADevice(host="h", port=1, friendly_name="Plex Media Server", location="loc")

        folders = [
            dlna._DlnaContainer(object_id="1", title="Movies"),
            dlna._DlnaContainer(object_id="2", title="Recently Added"),
        ]

        monkeypatch.setattr(dlna, "EXCLUDE_DLNA_LIBRARIES", set())
        monkeypatch.setattr(dlna, "_list_child_containers", lambda _d, _oid: folders)

        answers = iter(["1", "1"])
        monkeypatch.setattr(dlna, "input", lambda _prompt="": next(answers))

        selected = dlna._select_folders_plex(base, dev)
        assert selected is not None
        assert [c.title for c in selected] == ["Movies"]


class TestE2EPipeline:
    def test_analyze_dlna_server_pipeline_writes_csv_and_filters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dev = DLNADevice(host="h", port=1, friendly_name="Non Plex", location="loc")

        root = dlna._DlnaContainer(object_id="root", title="Videos")
        library_container = dlna._DlnaContainer(object_id="lib1", title="Movies")

        monkeypatch.setattr(dlna, "_list_video_root_containers", lambda _d: [root])
        monkeypatch.setattr(dlna, "_auto_descend_folder_browse", lambda _d, c: c)
        monkeypatch.setattr(dlna, "_select_folders_non_plex", lambda _b, _d: [library_container])
        monkeypatch.setattr(dlna, "_is_plex_server", lambda _d: False)

        items = [
            dlna._DlnaVideoItem(title="Movie A", resource_url="http://x/a.mp4", size_bytes=10, year=2001),
            dlna._DlnaVideoItem(title="Movie B", resource_url="http://x/b.mp4", size_bytes=None, year=None),
        ]
        monkeypatch.setattr(dlna, "_iter_video_items_recursive", lambda _d, _oid: items)

        def fake_get_movie_record(*, title: str, year: int | None, imdb_id_hint: str | None) -> dict[str, object]:
            return {
                "Poster": f"http://poster/{title}",
                "Website": f"http://trailer/{title}",
                "imdbID": "tt1234567",
                "__wiki": {"wikidata_id": "Q1", "wikipedia_title": "Some Title"},
            }

        monkeypatch.setattr(dlna, "get_movie_record", fake_get_movie_record)

        def fake_analyze_input_movie(
            movie_input: object,
            fetch_omdb: Callable[[str, int | None], dict[str, object]],
        ) -> dict[str, object]:
            title = getattr(movie_input, "title")
            year = getattr(movie_input, "year")
            library = getattr(movie_input, "library")
            file_size_bytes = getattr(movie_input, "file_size_bytes")

            _ = fetch_omdb(title, year)

            decision = "DELETE" if title == "Movie A" else "KEEP"
            return {
                "title": title,
                "year": year,
                "library": library,
                "file_size_bytes": file_size_bytes,
                "decision": decision,
            }

        monkeypatch.setattr(dlna, "analyze_input_movie", fake_analyze_input_movie)

        sort_called: list[bool] = []

        def fake_sort(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            sort_called.append(True)
            return rows

        monkeypatch.setattr(dlna, "sort_filtered_rows", fake_sort)

        written: dict[str, object] = {}

        def capture_all(path: str, rows: list[dict[str, object]]) -> None:
            written["all_path"] = path
            written["all_rows"] = rows

        def capture_filtered(path: str, rows: list[dict[str, object]]) -> None:
            written["filtered_path"] = path
            written["filtered_rows"] = rows

        def capture_sugg(path: str, rows: list[dict[str, object]]) -> None:
            written["sugg_path"] = path
            written["sugg_rows"] = rows

        monkeypatch.setattr(dlna, "write_all_csv", capture_all)
        monkeypatch.setattr(dlna, "write_filtered_csv", capture_filtered)
        monkeypatch.setattr(dlna, "write_suggestions_csv", capture_sugg)

        dlna.analyze_dlna_server(device=dev)

        all_rows_obj = written.get("all_rows")
        assert isinstance(all_rows_obj, list)
        assert len(all_rows_obj) == 2

        row0 = all_rows_obj[0]
        assert row0["library"] == "Movies"
        assert row0["poster_url"] is not None
        assert row0["trailer_url"] is not None
        assert row0["imdb_id"] is not None
        assert row0["omdb_json"] is not None
        assert row0["wikidata_id"] is not None
        assert row0["wikipedia_title"] is not None
        assert row0["file"] in {"http://x/a.mp4", "http://x/b.mp4"}
        assert "file_size" in row0

        filtered_rows_obj = written.get("filtered_rows")
        assert isinstance(filtered_rows_obj, list)
        assert len(filtered_rows_obj) == 1
        assert filtered_rows_obj[0]["decision"] == "DELETE"
        assert sort_called == [True]

        sugg_rows_obj = written.get("sugg_rows")
        assert isinstance(sugg_rows_obj, list)
        assert sugg_rows_obj == []