import backend.resilience as res


def test_circuit_breaker_transitions(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(res.time, "monotonic", lambda: now[0])

    breaker = res.CircuitBreaker(
        failure_threshold=2, open_seconds=1.0, half_open_max_calls=1
    )

    allowed, reason = breaker.allow("svc")
    assert allowed is True
    assert reason == "closed:new"

    breaker.on_failure("svc", error="err-1")
    allowed, reason = breaker.allow("svc")
    assert allowed is True
    assert reason == "closed"

    breaker.on_failure("svc", error="err-2")
    allowed, reason = breaker.allow("svc")
    assert allowed is False
    assert reason == "open"

    now[0] = 2.0
    allowed, reason = breaker.allow("svc")
    assert allowed is True
    assert reason.startswith("half_open")

    allowed, reason = breaker.allow("svc")
    assert allowed is True
    assert "probe" in reason

    allowed, reason = breaker.allow("svc")
    assert allowed is False
    assert "quota_reached" in reason

    breaker.on_success("svc")
    allowed, reason = breaker.allow("svc")
    assert allowed is True
    assert reason == "closed"


def test_call_with_resilience_retries(monkeypatch):
    breaker = res.CircuitBreaker(failure_threshold=5, open_seconds=1.0)
    monkeypatch.setattr(res, "backoff_sleep", lambda attempt: None)

    calls = []

    def fn():
        calls.append("call")
        if len(calls) < 3:
            raise ValueError("nope")
        return "ok"

    out, status = res.call_with_resilience(
        breaker=breaker,
        key="svc",
        fn=fn,
        should_retry=lambda exc: isinstance(exc, ValueError),
        max_retries=2,
    )

    assert out == "ok"
    assert status == "ok"
    assert len(calls) == 3


def test_call_with_resilience_no_retry(monkeypatch):
    breaker = res.CircuitBreaker(failure_threshold=5, open_seconds=1.0)
    monkeypatch.setattr(res, "backoff_sleep", lambda attempt: None)

    def fn():
        raise RuntimeError("boom")

    out, status = res.call_with_resilience(
        breaker=breaker,
        key="svc",
        fn=fn,
        should_retry=lambda exc: False,
        max_retries=2,
    )

    assert out is None
    assert status.startswith("error:")
