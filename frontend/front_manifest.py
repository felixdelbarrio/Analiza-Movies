from __future__ import annotations

"""
frontend/front_manifest.py

Lectura opcional de manifest (si existe).
Si no existe, el front puede seguir por configuración estática (config_front_artifacts).

Este diseño permite evolucionar a manifest sin acoplarse al backend:
- el backend (si quiere) escribe reports/manifest.json
- el front lo lee si está, y si no, usa defaults.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from frontend.config_front_base import REPORTS_DIR


@dataclass(frozen=True)
class FrontManifest:
    schema: int
    generated_at: str | None
    artifacts: dict[str, str]

    def get_path(self, key: str, *, project_root: Path) -> Path | None:
        raw = self.artifacts.get(key)
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else (project_root / p)


def load_manifest_if_present(*, manifest_path: Path | None = None) -> FrontManifest | None:
    p = manifest_path or (REPORTS_DIR / "manifest.json")
    if not p.exists():
        return None

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(obj, Mapping):
        return None

    schema = obj.get("schema")
    if not isinstance(schema, int):
        return None

    gen = obj.get("generated_at")
    generated_at = gen if isinstance(gen, str) and gen.strip() else None

    artifacts_obj = obj.get("artifacts")
    if not isinstance(artifacts_obj, Mapping):
        return None

    artifacts: dict[str, str] = {}
    for k, v in artifacts_obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            artifacts[k.strip()] = v.strip()

    return FrontManifest(schema=schema, generated_at=generated_at, artifacts=artifacts)