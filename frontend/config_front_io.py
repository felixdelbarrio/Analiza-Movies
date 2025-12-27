from __future__ import annotations

"""
frontend/config_front_io.py

Convenciones de I/O para el FRONT (lectura de outputs del backend),
sin importar backend.

Idea:
- El front consume archivos en reports/ y data/ por nombre "contractual".
- Si existe manifest, se prioriza (mejor).
- Si no, se usa fallback por patrones de nombres (sin API).
"""

from typing import Final

from frontend.config_front_base import _get_env_str, _get_env_bool

# Preferir manifest si existe (recomendado)
FRONT_USE_MANIFEST: Final[bool] = _get_env_bool("FRONT_USE_MANIFEST", True)

# Nombre del manifest en reports/
FRONT_REPORTS_MANIFEST_NAME: Final[str] = _get_env_str("FRONT_REPORTS_MANIFEST_NAME", "manifest.json")

# Nombres “convencionales” (ajusta a tus outputs reales)
# Ejemplo: si tu backend genera reports/all_movies.csv o reports/report.csv etc.
FRONT_REPORT_CSV_CANDIDATES: Final[list[str]] = [
    s.strip()
    for s in _get_env_str("FRONT_REPORT_CSV_CANDIDATES", "all_movies.csv,report.csv,metadata.csv").split(",")
    if s.strip()
]

# Si quieres leer también JSON (p.ej. cache export, resumen, etc.)
FRONT_REPORT_JSON_CANDIDATES: Final[list[str]] = [
    s.strip()
    for s in _get_env_str("FRONT_REPORT_JSON_CANDIDATES", "summary.json,omdb_cache_export.json").split(",")
    if s.strip()
]