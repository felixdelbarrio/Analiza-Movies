from __future__ import annotations

"""
frontend/front_logger.py

Logger ultra simple para frontend:
- No depende del backend
- En Streamlit, lo más fiable suele ser no spamear stdout.
- Si FRONT_DEBUG=True, permite warnings en consola.
"""

from frontend.config_front_base import FRONT_DEBUG


def log_warning(msg: str) -> None:
    if FRONT_DEBUG:
        # stdout/stderr (visible al ejecutar streamlit)
        print(str(msg))


def log_info(msg: str) -> None:
    if FRONT_DEBUG:
        print(str(msg))