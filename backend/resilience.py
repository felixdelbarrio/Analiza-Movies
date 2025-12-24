from __future__ import annotations

"""
backend/resilience.py

Retry con backoff+jitter + circuit breaker simple (por endpoint/key).
Pensado para DLNA SOAP Browse, y también para accesos Plex “sensibles”.

Principios:
- Backoff exponencial con jitter para evitar thundering herd.
- Circuit breaker:
    - CLOSED (normal)
    - OPEN (bloquea temporalmente)
    - HALF_OPEN (prueba 1 request; si ok => CLOSED; si falla => OPEN)
- Thread-safe.

No impone logging: devuelve estados para que el caller use backend/logger.py.
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float = 0.0
    state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
    last_error: str = ""


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        open_seconds: float = 20.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, CircuitState] = {}
        self._failure_threshold = max(1, int(failure_threshold))
        self._open_seconds = max(0.1, float(open_seconds))
        self._half_open_max_calls = max(1, int(half_open_max_calls))
        self._half_open_inflight: dict[str, int] = {}

    def allow(self, key: str) -> tuple[bool, str]:
        """
        Decide si se permite la llamada.
        Returns: (allowed, reason)
        """
        now = time.monotonic()
        with self._lock:
            st = self._states.get(key)
            if st is None:
                self._states[key] = CircuitState()
                return True, "closed:new"

            if st.state == "CLOSED":
                return True, "closed"

            if st.state == "OPEN":
                if (now - st.opened_at) >= self._open_seconds:
                    st.state = "HALF_OPEN"
                    self._half_open_inflight[key] = 0
                    return True, "half_open:cooldown_elapsed"
                return False, "open"

            # HALF_OPEN
            inflight = int(self._half_open_inflight.get(key, 0))
            if inflight >= self._half_open_max_calls:
                return False, "half_open:quota_reached"
            self._half_open_inflight[key] = inflight + 1
            return True, "half_open:probe"

    def on_success(self, key: str) -> None:
        with self._lock:
            st = self._states.get(key)
            if st is None:
                self._states[key] = CircuitState()
                return
            st.failures = 0
            st.last_error = ""
            st.opened_at = 0.0
            st.state = "CLOSED"
            self._half_open_inflight.pop(key, None)

    def on_failure(self, key: str, *, error: str) -> None:
        now = time.monotonic()
        with self._lock:
            st = self._states.get(key)
            if st is None:
                st = CircuitState()
                self._states[key] = st

            st.failures += 1
            st.last_error = str(error)[:500]

            if st.state == "HALF_OPEN":
                # fallo en probe => OPEN directamente
                st.state = "OPEN"
                st.opened_at = now
                self._half_open_inflight.pop(key, None)
                return

            if st.failures >= self._failure_threshold:
                st.state = "OPEN"
                st.opened_at = now
                self._half_open_inflight.pop(key, None)

    def debug_state(self, key: str) -> CircuitState | None:
        with self._lock:
            st = self._states.get(key)
            return None if st is None else CircuitState(**st.__dict__)


def backoff_sleep(attempt: int, *, base: float = 0.35, cap: float = 6.0, jitter: float = 0.35) -> None:
    """
    Exponential backoff: base * 2^attempt, cap y jitter.
    jitter=0.35 => +-35% aleatorio.
    """
    a = max(0, int(attempt))
    delay = min(cap, base * (2 ** a))
    if jitter > 0:
        j = 1.0 + random.uniform(-jitter, jitter)
        delay = max(0.0, delay * j)
    time.sleep(delay)


def call_with_resilience(
    *,
    breaker: CircuitBreaker,
    key: str,
    fn: Callable[[], T],
    should_retry: Callable[[BaseException], bool],
    max_retries: int = 2,
) -> tuple[T | None, str]:
    """
    Ejecuta fn() con:
    - circuit breaker
    - retries con backoff

    Returns: (result_or_none, status)
    status:
      - "ok"
      - "circuit_open"
      - "error"
    """
    allowed, reason = breaker.allow(key)
    if not allowed:
        return None, f"circuit_open:{reason}"

    last_exc: BaseException | None = None
    for attempt in range(0, max(0, int(max_retries)) + 1):
        try:
            out = fn()
            breaker.on_success(key)
            return out, "ok"
        except BaseException as exc:
            last_exc = exc
            breaker.on_failure(key, error=repr(exc))

            if attempt >= max_retries or not should_retry(exc):
                break

            backoff_sleep(attempt)

    return None, f"error:{repr(last_exc) if last_exc else 'unknown'}"