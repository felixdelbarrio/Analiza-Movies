import backend.config_base as cfg


def test_clean_env_raw():
    assert cfg._clean_env_raw(None) is None
    assert cfg._clean_env_raw("  ") is None
    assert cfg._clean_env_raw("'value'") == "value"
    assert cfg._clean_env_raw('"value"') == "value"
    assert cfg._clean_env_raw("  value ") == "value"


def test_get_env_parsers(monkeypatch):
    monkeypatch.delenv("TEST_STR", raising=False)
    assert cfg._get_env_str("TEST_STR", "default") == "default"

    monkeypatch.setenv("TEST_STR", "  hello ")
    assert cfg._get_env_str("TEST_STR", "default") == "hello"

    monkeypatch.setenv("TEST_INT", "10")
    assert cfg._get_env_int("TEST_INT", 1) == 10
    monkeypatch.setenv("TEST_INT", "bad")
    assert cfg._get_env_int("TEST_INT", 1) == 1

    monkeypatch.setenv("TEST_FLOAT", "3.5")
    assert cfg._get_env_float("TEST_FLOAT", 1.0) == 3.5
    monkeypatch.setenv("TEST_FLOAT", "bad")
    assert cfg._get_env_float("TEST_FLOAT", 1.0) == 1.0

    monkeypatch.setenv("TEST_BOOL", "yes")
    assert cfg._get_env_bool("TEST_BOOL", False) is True
    monkeypatch.setenv("TEST_BOOL", "no")
    assert cfg._get_env_bool("TEST_BOOL", True) is False
    monkeypatch.setenv("TEST_BOOL", "maybe")
    assert cfg._get_env_bool("TEST_BOOL", True) is True


def test_get_env_enum_and_caps(monkeypatch):
    monkeypatch.setenv("TEST_ENUM", "Value")
    assert (
        cfg._get_env_enum_str(
            "TEST_ENUM",
            default="default",
            allowed={"value", "other"},
            normalize=True,
        )
        == "value"
    )

    monkeypatch.setenv("TEST_ENUM", "invalid")
    assert (
        cfg._get_env_enum_str(
            "TEST_ENUM",
            default="default",
            allowed={"value", "other"},
            normalize=True,
        )
        == "default"
    )

    assert cfg._cap_int("CAP", 1, min_v=3, max_v=5) == 3
    assert cfg._cap_int("CAP", 10, min_v=3, max_v=5) == 5
    assert cfg._cap_int("CAP", 4, min_v=3, max_v=5) == 4

    assert cfg._cap_float_min("CAPF", 0.1, min_v=0.5) == 0.5
    assert cfg._cap_float_min("CAPF", 0.6, min_v=0.5) == 0.6


def test_parse_env_kv_map_json_and_fallback():
    raw_json = '{"a": "1", "b": 2}'
    assert cfg._parse_env_kv_map(raw_json) == {"a": "1", "b": "2"}

    raw_fallback = "a:1, b: 2, :bad, c:, d:4"
    assert cfg._parse_env_kv_map(raw_fallback) == {"a": "1", "b": "2", "d": "4"}

    assert cfg._parse_env_kv_map("nope") == {}
