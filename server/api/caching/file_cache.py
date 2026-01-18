# LRU+TTL+locks + load_json_cached/load_csv_cached + retries
from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable, TypeVar, cast

import pandas as pd
from pandas._typing import DtypeArg

from server.api.services import metrics
from server.api.settings import Settings

T = TypeVar("T")


@dataclass(frozen=True)
class CachedEntry:
    """
    Entrada cacheada por fichero.

    - mtime_ns: detecta cambios del fichero.
    - loaded_at_monotonic: permite TTL sin depender del reloj del sistema.
    - data: JSON (objeto Python) o DataFrame.
    """

    mtime_ns: int
    loaded_at_monotonic: float
    data: object


class FileCache:
    """
    Caché LRU + TTL opcional + locks por key.
    Pensado para servidores con acceso concurrente.
    """

    def __init__(self, settings: Settings) -> None:
        self._max_entries = max(1, settings.file_cache_max_entries)
        self._ttl_seconds = max(0.0, settings.file_cache_ttl_seconds)
        self._read_max_attempts = max(1, settings.file_read_max_attempts)
        self._read_retry_sleep_s = max(0.0, settings.file_read_retry_sleep_s)

        self._cache_lock = RLock()
        self._key_locks: dict[str, RLock] = {}

        self._json_cache: "OrderedDict[str, CachedEntry]" = OrderedDict()
        self._csv_cache: "OrderedDict[str, CachedEntry]" = OrderedDict()

    def _get_key_lock(self, key: str) -> RLock:
        with self._cache_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = RLock()
                self._key_locks[key] = lock
            return lock

    @staticmethod
    def _mtime_ns(path: Path) -> int:
        return path.stat().st_mtime_ns

    def _is_fresh(self, entry: CachedEntry, mtime_ns: int) -> bool:
        if entry.mtime_ns != mtime_ns:
            return False
        if self._ttl_seconds <= 0.0:
            return True
        age = time.monotonic() - entry.loaded_at_monotonic
        return age <= self._ttl_seconds

    def _lru_get(
        self, cache: "OrderedDict[str, CachedEntry]", key: str
    ) -> CachedEntry | None:
        with self._cache_lock:
            entry = cache.get(key)
            if entry is None:
                return None
            cache.move_to_end(key)
            return entry

    def _lru_put(
        self, cache: "OrderedDict[str, CachedEntry]", key: str, entry: CachedEntry
    ) -> None:
        with self._cache_lock:
            cache[key] = entry
            cache.move_to_end(key)
            while len(cache) > self._max_entries:
                cache.popitem(last=False)
                metrics.inc("cache_evictions_total", 1)

    def _read_with_retries(self, read_fn: Callable[[], T]) -> T:
        last_exc: Exception | None = None
        for attempt in range(self._read_max_attempts):
            try:
                return read_fn()
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self._read_max_attempts:
                    raise
                metrics.inc("cache_read_retries_total", 1)
                time.sleep(self._read_retry_sleep_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("read retries: unreachable")

    def load_json(self, path: Path) -> object:
        key = str(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        lock = self._get_key_lock(key)
        with lock:
            m = self._mtime_ns(path)
            cached = self._lru_get(self._json_cache, key)
            if cached is not None and self._is_fresh(cached, m):
                metrics.inc("cache_json_hit_total", 1)
                return cached.data

            metrics.inc("cache_json_miss_total", 1)

            def _read() -> object:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)

            data = self._read_with_retries(_read)

            metrics.inc("cache_refresh_total", 1)
            entry = CachedEntry(
                mtime_ns=m, loaded_at_monotonic=time.monotonic(), data=data
            )
            self._lru_put(self._json_cache, key, entry)
            return data

    def load_csv(self, path: Path, *, text_columns: list[str]) -> pd.DataFrame:
        key = str(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        lock = self._get_key_lock(key)
        with lock:
            m = self._mtime_ns(path)
            cached = self._lru_get(self._csv_cache, key)
            if (
                cached is not None
                and self._is_fresh(cached, m)
                and isinstance(cached.data, pd.DataFrame)
            ):
                metrics.inc("cache_csv_hit_total", 1)
                return cached.data

            metrics.inc("cache_csv_miss_total", 1)

            dtype_map: dict[str, DtypeArg] = {c: "string" for c in text_columns}
            dtype_arg = cast(DtypeArg, dtype_map)

            def _read() -> pd.DataFrame:
                return pd.read_csv(path, dtype=dtype_arg, encoding="utf-8")

            df = self._read_with_retries(_read)

            for col in text_columns:
                if col in df.columns:
                    try:
                        df[col] = df[col].astype("string")
                    except Exception:
                        df[col] = df[col].astype(str)

            metrics.inc("cache_refresh_total", 1)
            entry = CachedEntry(
                mtime_ns=m, loaded_at_monotonic=time.monotonic(), data=df
            )
            self._lru_put(self._csv_cache, key, entry)
            return df
