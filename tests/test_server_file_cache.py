import json

import pandas as pd
from pandas.api.types import is_string_dtype

from server.api.caching.file_cache import FileCache
from server.api.settings import Settings


def _settings() -> Settings:
    return Settings(
        log_level="INFO",
        cors_origins_raw="*",
        cors_allow_credentials=False,
        gzip_min_size=0,
        file_cache_max_entries=4,
        file_cache_ttl_seconds=0.0,
        file_read_max_attempts=1,
        file_read_retry_sleep_s=0.0,
    )


def test_file_cache_load_json_refreshes_on_change(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    cache = FileCache(_settings())
    first = cache.load_json(path)
    assert first == {"a": 1}

    path.write_text(json.dumps({"a": 2}), encoding="utf-8")
    second = cache.load_json(path)
    assert second == {"a": 2}


def test_file_cache_load_csv_casts_text_columns(tmp_path):
    path = tmp_path / "report.csv"
    df = pd.DataFrame([{"poster_url": "http://x", "title": "Movie"}])
    df.to_csv(path, index=False)

    cache = FileCache(_settings())
    out = cache.load_csv(path, text_columns=["poster_url"])

    assert is_string_dtype(out["poster_url"].dtype)
