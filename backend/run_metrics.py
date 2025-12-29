from __future__ import annotations

"""
backend/run_metrics.py

Métricas agregadas del run (thread-safe) para Plex/DLNA/OMDb/Wiki.

Objetivo:
- Contar eventos relevantes (calls, errors, retries, cache hits, etc.)
- Poder imprimir un resumen final CONSISTENTE (sin depender de variables locales).
- Ser seguro en ThreadPool.

Uso:
    from backend.run_metrics import METRICS

    METRICS.incr("dlna.browse.calls")
    METRICS.incr("dlna.browse.errors.http_500")
    METRICS.observe_ms("dlna.browse.latency_ms", elapsed_ms)
    METRICS.add_error("dlna", "browse", endpoint=url, detail="HTTP 500")

    summary = METRICS.snapshot()
"""

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ErrorEvent:
    ts: float
    subsystem: str   # "dlna" | "plex" | "omdb" | "wiki"
    action: str      # "browse" | "attr_read" | "query" | ...
    endpoint: str | None
    detail: str


class RunMetrics:
    """
    Contadores + observaciones básicas.

    Diseño:
    - counters: dict[str, int]
    - timings_ms: dict[str, {"count": int, "sum": float, "min": float, "max": float, "avg": float}]
    - errors: lista acotada para diagnóstico (no infinito)
    """

    def __init__(self, *, max_error_events: int = 2000) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._timings: dict[str, dict[str, float]] = {}
        self._errors: list[ErrorEvent] = []
        self._max_error_events = max(0, int(max_error_events))

    def incr(self, key: str, n: int = 1) -> None:
        if not key:
            return
        with self._lock:
            self._counters[key] = int(self._counters.get(key, 0)) + int(n)

    def set_if_absent(self, key: str, value: int) -> None:
        if not key:
            return
        with self._lock:
            if key not in self._counters:
                self._counters[key] = int(value)

    def observe_ms(self, key: str, ms: float) -> None:
        if not key:
            return
        v = float(ms)
        with self._lock:
            t = self._timings.get(key)
            if t is None:
                self._timings[key] = {
                    "count": 1.0,
                    "sum": v,
                    "min": v,
                    "max": v,
                }
                return
            t["count"] = float(t.get("count", 0.0)) + 1.0
            t["sum"] = float(t.get("sum", 0.0)) + v
            t["min"] = min(float(t.get("min", v)), v)
            t["max"] = max(float(t.get("max", v)), v)

    def add_error(self, subsystem: str, action: str, *, endpoint: str | None, detail: str) -> None:
        ev = ErrorEvent(
            ts=time.time(),
            subsystem=str(subsystem),
            action=str(action),
            endpoint=endpoint,
            detail=str(detail)[:800],
        )
        with self._lock:
            if self._max_error_events <= 0:
                return
            if len(self._errors) >= self._max_error_events:
                # strategy: drop oldest
                self._errors.pop(0)
            self._errors.append(ev)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            timings = {k: dict(v) for k, v in self._timings.items()}
            errors = list(self._errors)

        # Añadimos derivados útiles
        derived: dict[str, Any] = {}
        derived["errors.total"] = len(errors)
        derived["errors.by_subsystem"] = {}
        for e in errors:
            by = derived["errors.by_subsystem"]
            by[e.subsystem] = int(by.get(e.subsystem, 0)) + 1

        # Timing: añadimos avg
        for t in timings.values():
            cnt = max(1.0, float(t.get("count", 0.0)))
            t["avg"] = float(t.get("sum", 0.0)) / cnt

        return {"counters": counters, "timings_ms": timings, "errors": errors, "derived": derived}


# -----------------------------------------------------------------------------
# Compat shims (para módulos que esperan inc/observe_seconds)
# -----------------------------------------------------------------------------

def inc(name: str, *, value: int = 1) -> None:
    """Compat: incrementa contador."""
    try:
        METRICS.incr(name, n=int(value))
    except Exception:
        return


def counter_inc(name: str, *, value: int = 1) -> None:
    """Alias compat."""
    inc(name, value=value)


def observe_seconds(name: str, *, seconds: float) -> None:
    """Compat: observa timing en segundos (se guarda en ms)."""
    try:
        METRICS.observe_ms(name, float(seconds) * 1000.0)
    except Exception:
        return


def timing(name: str, *, seconds: float) -> None:
    """Alias compat."""
    observe_seconds(name, seconds=seconds)


# Singleton del run (módulo)
METRICS = RunMetrics()