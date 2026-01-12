from backend.analiza_dlna import analyze_dlna_server
from backend.dlna_client import DlnaContainer, DlnaVideoItem, TraversalLimits
from backend.dlna_discovery import DLNADevice


class DummyWriter:
    def __init__(self) -> None:
        self.rows = []

    def write_row(self, row):
        self.rows.append(dict(row))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyMetrics:
    def __init__(self) -> None:
        self.counters = {}

    def incr(self, key: str, n: int = 1) -> None:
        self.counters[key] = int(self.counters.get(key, 0)) + int(n)

    def observe_ms(self, key: str, ms: float) -> None:
        return

    def add_error(self, subsystem: str, action: str, *, endpoint: str | None, detail: str) -> None:
        return

    def snapshot(self):
        return {"counters": dict(self.counters)}


class FakeDLNAClient:
    def __init__(self, *, device, root, containers, items_by_object_id) -> None:
        self._device = device
        self._root = root
        self._containers = containers
        self._items_by_object_id = items_by_object_id
        self.select_device_calls = 0
        self.select_containers_calls = 0
        self.scan_calls = []

    def ask_user_to_select_device(self):
        self.select_device_calls += 1
        return self._device

    def ask_user_to_select_video_containers(self, device):
        self.select_containers_calls += 1
        return self._root, self._containers

    def iter_video_items_recursive(self, device, object_id, limits):
        self.scan_calls.append(object_id)
        return list(self._items_by_object_id.get(object_id, [])), {}

    def extract_ext_from_resource_url(self, resource_url: str) -> str:
        if resource_url.endswith(".mkv"):
            return ".mkv"
        return ""


def _patch_common(monkeypatch, *, silent: bool) -> None:
    import backend.analiza_dlna as analiza_dlna

    monkeypatch.setattr(analiza_dlna, "SILENT_MODE", silent)
    monkeypatch.setattr(analiza_dlna, "DEBUG_MODE", False)
    monkeypatch.setattr(analiza_dlna, "PLEX_ANALYZE_WORKERS", 1)
    monkeypatch.setattr(analiza_dlna, "OMDB_HTTP_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(analiza_dlna, "OMDB_HTTP_MIN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(analiza_dlna, "METRICS", DummyMetrics())
    monkeypatch.setattr(analiza_dlna, "build_traversal_limits", lambda: TraversalLimits(1, 1, 1, 1, 1))
    monkeypatch.setattr(analiza_dlna, "flush_external_caches", lambda: None)
    monkeypatch.setattr(analiza_dlna, "reset_omdb_metrics", lambda: None)
    monkeypatch.setattr(analiza_dlna, "get_omdb_metrics_snapshot", lambda: {})


def test_analyze_dlna_server_returns_when_no_device(monkeypatch):
    import backend.analiza_dlna as analiza_dlna

    _patch_common(monkeypatch, silent=False)

    fake_client = FakeDLNAClient(device=None, root=None, containers=[], items_by_object_id={})
    monkeypatch.setattr(analiza_dlna, "DLNAClient", lambda: fake_client)

    def _fail_writer(*args, **kwargs):
        raise AssertionError("writers should not be opened when no device is selected")

    monkeypatch.setattr(analiza_dlna, "open_all_csv_writer", _fail_writer)
    monkeypatch.setattr(analiza_dlna, "open_suggestions_csv_writer", _fail_writer)

    analyze_dlna_server(device=None)
    assert fake_client.select_device_calls == 1
    assert fake_client.select_containers_calls == 0


def test_analyze_dlna_server_interactive_writes_filtered_rows(monkeypatch):
    import backend.analiza_dlna as analiza_dlna

    _patch_common(monkeypatch, silent=False)

    device = DLNADevice(
        friendly_name="TestServer",
        location="http://dlna.test/device.xml",
        host="127.0.0.1",
        port=8200,
    )
    root = DlnaContainer(object_id="root", title="Video")
    containers = [DlnaContainer(object_id="c1", title="Movies")]

    items_by_object_id = {
        "c1": [
            DlnaVideoItem(
                title="Volume 1",
                resource_url="http://dlna.test/Best.Movie.1999.mkv",
                size_bytes=123,
                item_id="item-1",
            ),
            DlnaVideoItem(
                title="Another Movie (2001)",
                resource_url="http://dlna.test/Another.Movie.2001.mkv",
                size_bytes=456,
                item_id="item-2",
            ),
        ]
    }

    fake_client = FakeDLNAClient(
        device=device,
        root=root,
        containers=containers,
        items_by_object_id=items_by_object_id,
    )
    monkeypatch.setattr(analiza_dlna, "DLNAClient", lambda: fake_client)

    all_writer = DummyWriter()
    sugg_writer = DummyWriter()
    filtered_writer = DummyWriter()

    monkeypatch.setattr(analiza_dlna, "open_all_csv_writer", lambda _path: all_writer)
    monkeypatch.setattr(analiza_dlna, "open_suggestions_csv_writer", lambda _path: sugg_writer)
    monkeypatch.setattr(analiza_dlna, "open_filtered_csv_writer_only_if_rows", lambda _path: filtered_writer)
    monkeypatch.setattr(analiza_dlna, "sort_filtered_rows", lambda rows: list(rows))

    seen_inputs = []

    def _fake_analyze_movie(movie_input, source_movie=None):
        seen_inputs.append(movie_input)
        decision = "DELETE" if "Best Movie" in movie_input.title else "KEEP"
        row = {"decision": decision, "title": movie_input.title}
        return row, None, []

    monkeypatch.setattr(analiza_dlna, "analyze_movie", _fake_analyze_movie)

    analyze_dlna_server(device=None)

    assert len(all_writer.rows) == 2
    assert len(sugg_writer.rows) == 0
    assert len(filtered_writer.rows) == 1

    titles = [mi.title for mi in seen_inputs]
    assert any(title.startswith("Best Movie") for title in titles)
    assert any(title.startswith("Another Movie") for title in titles)

    years = {mi.title: mi.year for mi in seen_inputs}
    assert years[next(title for title in years if title.startswith("Best Movie"))] == 1999
    assert years[next(title for title in years if title.startswith("Another Movie"))] == 2001


def test_analyze_dlna_server_silent_dedupe(monkeypatch):
    import backend.analiza_dlna as analiza_dlna

    _patch_common(monkeypatch, silent=True)

    device = DLNADevice(
        friendly_name="TestServer",
        location="http://dlna.test/device.xml",
        host="127.0.0.1",
        port=8200,
    )
    root = DlnaContainer(object_id="root", title="Video")
    containers = [
        DlnaContainer(object_id="c1", title="Movies A"),
        DlnaContainer(object_id="c2", title="Movies B"),
    ]

    shared_item = DlnaVideoItem(
        title="Shared Movie (2010)",
        resource_url="http://dlna.test/Shared.Movie.2010.mkv",
        size_bytes=100,
        item_id="dup-1",
    )

    items_by_object_id = {
        "c1": [shared_item],
        "c2": [shared_item],
    }

    fake_client = FakeDLNAClient(
        device=device,
        root=root,
        containers=containers,
        items_by_object_id=items_by_object_id,
    )
    monkeypatch.setattr(analiza_dlna, "DLNAClient", lambda: fake_client)

    all_writer = DummyWriter()
    sugg_writer = DummyWriter()

    monkeypatch.setattr(analiza_dlna, "open_all_csv_writer", lambda _path: all_writer)
    monkeypatch.setattr(analiza_dlna, "open_suggestions_csv_writer", lambda _path: sugg_writer)

    def _fail_filtered_writer(_path):
        raise AssertionError("filtered writer should not be called when no filtered rows")

    monkeypatch.setattr(analiza_dlna, "open_filtered_csv_writer_only_if_rows", _fail_filtered_writer)

    analyze_calls = []

    def _fake_analyze_movie(movie_input, source_movie=None):
        analyze_calls.append(movie_input)
        row = {"decision": "KEEP", "title": movie_input.title}
        return row, None, []

    monkeypatch.setattr(analiza_dlna, "analyze_movie", _fake_analyze_movie)

    analyze_dlna_server(device=None)

    assert len(analyze_calls) == 1
    assert len(all_writer.rows) == 1
    assert all_writer.rows[0]["decision"] == "KEEP"
    assert len(sugg_writer.rows) == 0
