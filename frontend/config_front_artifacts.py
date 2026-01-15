"""
frontend/config_front_artifacts.py

Contrato de artefactos que el front consume.

- No importa backend.
- Los nombres por defecto reflejan los ficheros actuales:
  data/: omdb_cache.json, wiki_cache.json
  reports/: metadata_fix.csv, report_all.csv, report_filtered.csv

Puedes sobreescribir nombres desde .env.front si en el futuro cambian.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from frontend.config_front_base import DATA_DIR, REPORTS_DIR, _get_env_str

# -------------------------
# DATA artifacts (json)
# -------------------------

OMDB_CACHE_FILENAME: Final[str] = (
    _get_env_str("FRONT_OMDB_CACHE_FILENAME", "omdb_cache.json") or "omdb_cache.json"
)
WIKI_CACHE_FILENAME: Final[str] = (
    _get_env_str("FRONT_WIKI_CACHE_FILENAME", "wiki_cache.json") or "wiki_cache.json"
)

OMDB_CACHE_PATH: Final[Path] = DATA_DIR / OMDB_CACHE_FILENAME
WIKI_CACHE_PATH: Final[Path] = DATA_DIR / WIKI_CACHE_FILENAME

# -------------------------
# REPORTS artifacts (csv)
# -------------------------

METADATA_FIX_FILENAME: Final[str] = (
    _get_env_str("FRONT_METADATA_FIX_FILENAME", "metadata_fix.csv")
    or "metadata_fix.csv"
)
REPORT_ALL_FILENAME: Final[str] = (
    _get_env_str("FRONT_REPORT_ALL_FILENAME", "report_all.csv") or "report_all.csv"
)
REPORT_FILTERED_FILENAME: Final[str] = (
    _get_env_str("FRONT_REPORT_FILTERED_FILENAME", "report_filtered.csv")
    or "report_filtered.csv"
)

METADATA_FIX_PATH: Final[Path] = REPORTS_DIR / METADATA_FIX_FILENAME
REPORT_ALL_PATH: Final[Path] = REPORTS_DIR / REPORT_ALL_FILENAME
REPORT_FILTERED_PATH: Final[Path] = REPORTS_DIR / REPORT_FILTERED_FILENAME
