from __future__ import annotations

from pathlib import Path

import pytest

from frontend import config_front_base


def test_save_front_grid_colorize_writes_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env.front"
    env_path.write_text("OTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(config_front_base, "_ENV_FRONT_PATH", env_path)

    config_front_base.save_front_grid_colorize(True)
    assert "FRONT_GRID_COLORIZE=true" in env_path.read_text(encoding="utf-8")

    config_front_base.save_front_grid_colorize(False)
    assert "FRONT_GRID_COLORIZE=false" in env_path.read_text(encoding="utf-8")
