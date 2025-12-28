from __future__ import annotations

from threading import RLock

_LOCK = RLock()
_METRICS: dict[str, int] = {
    "http_requests_total": 0,
    "http_errors_5xx_total": 0,
    "cache_json_hit_total": 0,
    "cache_json_miss_total": 0,
    "cache_csv_hit_total": 0,
    "cache_csv_miss_total": 0,
    "cache_evictions_total": 0,
    "cache_refresh_total": 0,
    "cache_read_retries_total": 0,
}


def inc(name: str, value: int = 1) -> None:
    with _LOCK:
        _METRICS[name] = _METRICS.get(name, 0) + value


def render_prometheus() -> str:
    with _LOCK:
        lines: list[str] = []
        for k, v in sorted(_METRICS.items()):
            lines.append(f"# TYPE {k} counter")
            lines.append(f"{k} {v}")
        return "\n".join(lines) + "\n"