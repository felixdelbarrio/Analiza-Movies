import backend.logger as logger


def test_filter_log_kwargs_accepts_supported_keys():
    err = ValueError("boom")
    out = logger._filter_log_kwargs(
        {
            "exc_info": err,
            "stack_info": True,
            "stacklevel": 2,
            "extra": {"a": 1},
            "bad": "nope",
        }
    )

    assert "exc_info" in out and out["exc_info"] is err
    assert out["stack_info"] is True
    assert out["stacklevel"] == 2
    assert out["extra"] == {"a": 1}
    assert "bad" not in out


def test_filter_log_kwargs_ignores_invalid_types():
    out = logger._filter_log_kwargs(
        {"exc_info": "no", "stack_info": "no", "stacklevel": "no", "extra": "no"}
    )
    assert out == {}


def test_append_bounded_log_respects_limit(monkeypatch):
    monkeypatch.setattr(logger, "logs_limit", lambda: 2)
    logs = []

    logger.append_bounded_log(logs, "line-1")
    logger.append_bounded_log(logs, "line-2", tag="tag")
    logger.append_bounded_log(logs, "line-3")

    assert logs[0] == "line-1"
    assert logs[1].startswith("[tag] ")
    assert logs[-1] == logger._LOGS_TRUNCATED_SENTINEL


def test_truncate_line_marks_truncated():
    out = logger.truncate_line("x" * 50, max_chars=10)
    assert "truncated" in out
    assert len(out) <= 50
