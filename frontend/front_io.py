from __future__ import annotations

"""
frontend/front_io.py

Carga datos desde reports/ y data/ para el FRONT (0 imports backend).

- Si existe manifest.json (recomendado), lo usa como fuente de verdad.
- Si no, busca por candidatos en reports/.
- Expone providers listos para front_stats (RecordsProvider).

El "record" para stats debe ser Mapping con 'imdbRating' o 'imdb_rating'.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from frontend.config_front_base import FRONT_DEBUG_MODE, FRONT_REPORTS_DIR, FRONT_DATA_DIR
from frontend.config_front_io import (
    FRONT_USE_MANIFEST,
    FRONT_REPORTS_MANIFEST_NAME,
    FRONT_REPORT_CSV_CANDIDATES,
    FRONT_REPORT_JSON_CANDIDATES,
)

Record = Mapping[str, Any]


def _dbg(msg: object) -> None:
    if FRONT_DEBUG_MODE:
        try:
            print(f"[FRONT][IO][DEBUG] {msg}")
        except Exception:
            pass


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def _manifest_path() -> Path:
    return (FRONT_REPORTS_DIR / FRONT_REPORTS_MANIFEST_NAME).resolve()


def read_reports_manifest() -> dict[str, Any] | None:
    p = _manifest_path()
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _dbg(f"manifest read failed: {exc!r}")
        return None


def _manifest_pick_path(manifest: Mapping[str, Any], key: str) -> Path | None:
    """
    Convención recomendada para manifest:
      {
        "schema": 1,
        "generated_at": "...",
        "artifacts": {
          "report_csv": "reports/all_movies.csv",
          "omdb_records_json": "data/omdb_cache_export.json"
        }
      }
    Acepta paths relativos al project root, o absolutos.
    """
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    v = artifacts.get(key)
    if not isinstance(v, str) or not v.strip():
        return None
    p = Path(v.strip())
    if p.is_absolute():
        return p
    # relativo al root = parent de frontend = project root
    project_root = FRONT_REPORTS_DIR.parent
    return (project_root / p).resolve()


# ---------------------------------------------------------------------
# Discovery por candidatos (sin manifest)
# ---------------------------------------------------------------------

def _first_existing_in_reports(names: list[str]) -> Path | None:
    for name in names:
        p = (FRONT_REPORTS_DIR / name).resolve()
        if p.exists() and p.is_file():
            return p
    return None


def _first_existing_in_data(names: list[str]) -> Path | None:
    for name in names:
        p = (FRONT_DATA_DIR / name).resolve()
        if p.exists() and p.is_file():
            return p
    return None


# ---------------------------------------------------------------------
# Lectores
# ---------------------------------------------------------------------

def load_report_df() -> pd.DataFrame | None:
    """
    Devuelve el DataFrame principal del report (CSV) si existe.
    Prioridad:
      1) manifest.artifacts.report_csv
      2) candidatos en reports/
    """
    p: Path | None = None

    if FRONT_USE_MANIFEST:
        m = read_reports_manifest()
        if m:
            p = _manifest_pick_path(m, "report_csv")

    if p is None:
        p = _first_existing_in_reports(list(FRONT_REPORT_CSV_CANDIDATES))

    if p is None:
        _dbg("No report CSV found.")
        return None

    try:
        _dbg(f"Loading report CSV: {p}")
        return pd.read_csv(p)
    except Exception as exc:
        _dbg(f"read_csv failed: {exc!r}")
        return None


def load_records_json() -> list[dict[str, Any]] | None:
    """
    Carga un JSON de records (list[dict]) útil para stats.
    Prioridad:
      1) manifest.artifacts.omdb_records_json (o similar)
      2) candidatos en reports/
      3) candidatos en data/
    """
    p: Path | None = None

    if FRONT_USE_MANIFEST:
        m = read_reports_manifest()
        if m:
            p = _manifest_pick_path(m, "omdb_records_json")

    if p is None:
        p = _first_existing_in_reports(list(FRONT_REPORT_JSON_CANDIDATES))
    if p is None:
        p = _first_existing_in_data(list(FRONT_REPORT_JSON_CANDIDATES))

    if p is None:
        _dbg("No records JSON found.")
        return None

    try:
        _dbg(f"Loading records JSON: {p}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # soporte opcional: {"records":[...]}
            recs = obj.get("records")
            if isinstance(recs, list):
                return [x for x in recs if isinstance(x, dict)]
        return None
    except Exception as exc:
        _dbg(f"records json load failed: {exc!r}")
        return None


# ---------------------------------------------------------------------
# Providers listos para front_stats
# ---------------------------------------------------------------------

def records_provider_from_json() -> Iterable[Record]:
    """
    Provider para front_stats.get_global_imdb_mean_from_cache(provider).
    """
    recs = load_records_json()
    if not recs:
        return []
    return recs


def records_provider_from_report_df() -> Iterable[Record]:
    """
    Provider alternativo: saca records desde el CSV principal.
    Crea dicts con imdb_rating si existe.
    """
    df = load_report_df()
    if df is None or df.empty:
        return []

    # Normaliza columna: acepta 'imdb_rating' o 'omdb_imdb_rating'
    col = None
    if "imdb_rating" in df.columns:
        col = "imdb_rating"
    elif "omdb_imdb_rating" in df.columns:
        col = "omdb_imdb_rating"

    if col is None:
        return []

    out: list[dict[str, Any]] = []
    for v in df[col].tolist():
        out.append({"imdb_rating": v})
    return out