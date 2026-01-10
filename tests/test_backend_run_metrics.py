from backend.run_metrics import RunMetrics


def test_run_metrics_counters_and_timings():
    metrics = RunMetrics(max_error_events=1)

    metrics.incr("a")
    metrics.incr("a", n=2)
    metrics.set_if_absent("b", 3)
    metrics.set_if_absent("b", 5)

    metrics.observe_ms("latency", 10)
    metrics.observe_ms("latency", 30)

    metrics.add_error("omdb", "query", endpoint="x", detail="bad")
    metrics.add_error("omdb", "query", endpoint="y", detail="worse")

    snap = metrics.snapshot()

    assert snap["counters"]["a"] == 3
    assert snap["counters"]["b"] == 3
    assert snap["timings_ms"]["latency"]["count"] == 2.0
    assert snap["timings_ms"]["latency"]["avg"] == 20.0

    assert snap["derived"]["errors.total"] == 1
    assert snap["derived"]["errors.by_subsystem"]["omdb"] == 1
