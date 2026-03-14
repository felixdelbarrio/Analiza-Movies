from __future__ import annotations

from typing import Final

from backend.config_base import REPORTS_DIR_PATH

REPORT_ALL_FILENAME: Final[str] = "report_all.csv"
REPORT_FILTERED_FILENAME: Final[str] = "report_filtered.csv"
METADATA_FIX_FILENAME: Final[str] = "metadata_fix.csv"

REPORT_ALL_PATH: Final[str] = str(REPORTS_DIR_PATH / REPORT_ALL_FILENAME)
REPORT_FILTERED_PATH: Final[str] = str(REPORTS_DIR_PATH / REPORT_FILTERED_FILENAME)
METADATA_FIX_PATH: Final[str] = str(REPORTS_DIR_PATH / METADATA_FIX_FILENAME)
