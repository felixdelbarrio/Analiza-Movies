from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from backend import logger as _logger


# ============================================================
#                   UTILIDADES INTERNAS CSV
# ============================================================


def _collect_fieldnames(rows: Iterable[Mapping[str, object]]) -> list[str]:
    """
    Devuelve la unión de todas las claves presentes en todas las filas.

    Esto evita inconsistencias si algunas filas tienen keys extra
    o si faltan keys en la primera fila.
    """
    fieldset: set[str] = set()
    for r in rows:
        fieldset.update(str(k) for k in r.keys())
    return sorted(fieldset)


def _write_dict_rows_csv(
    path: str,
    rows: Iterable[Mapping[str, object]],
    *,
    default_fieldnames: list[str] | None = None,
    empty_message: str | None = None,
    kind_label: str = "CSV",
) -> None:
    """
    Escritura atómica de CSV desde un iterable de dict/Mapping.

    - Si hay filas → usa unión de sus claves como cabecera.
    - Si no hay filas → usa default_fieldnames, o no escribe nada.
    """
    pathp = Path(path)
    dirpath = pathp.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    rows_list = list(rows)  # para poder iterar varias veces

    if rows_list:
        fieldnames = _collect_fieldnames(rows_list)
    else:
        if default_fieldnames is None:
            if empty_message:
                _logger.info(empty_message)
            else:
                _logger.info(f"No hay filas para escribir en {kind_label}.")
            return
        fieldnames = list(default_fieldnames)

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(dirpath),
            newline="",
        ) as tf:
            writer = csv.DictWriter(tf, fieldnames=fieldnames)
            writer.writeheader()
            if rows_list:
                writer.writerows(rows_list)
            temp_name = tf.name

        os.replace(temp_name, str(pathp))
        _logger.info(f"{kind_label} escrito en {path}")
    except Exception as exc:
        _logger.error(f"Error escribiendo {kind_label} en {path}: {exc}")


# ============================================================
#                    FUNCIONES PÚBLICAS CSV
# ============================================================


def write_all_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Escribe el CSV completo con todas las películas analizadas."""
    _write_dict_rows_csv(
        path,
        rows,
        empty_message="No hay filas para escribir en report_all.csv",
        kind_label="CSV completo",
    )


def write_filtered_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Escribe el CSV filtrado con DELETE/MAYBE."""
    _write_dict_rows_csv(
        path,
        rows,
        empty_message="No hay filas filtradas para escribir en report_filtered.csv",
        kind_label="CSV filtrado",
    )


_STANDARD_SUGGESTION_FIELDS: Final[list[str]] = [
    "plex_guid",
    "library",
    "plex_title",
    "plex_year",
    "omdb_title",
    "omdb_year",
    "imdb_rating",
    "imdb_votes",
    "suggestions_json",
]


def write_suggestions_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Escribe el CSV con sugerencias de metadata, incluso vacío."""
    rows_list = list(rows)

    if not rows_list:
        _logger.info(
            "No hay sugerencias de metadata para escribir. "
            "Se crea un CSV vacío con solo cabeceras."
        )

    _write_dict_rows_csv(
        path,
        rows_list,
        default_fieldnames=_STANDARD_SUGGESTION_FIELDS,
        kind_label="CSV de sugerencias",
    )


# ============================================================
#                INFORME HTML INTERACTIVO
# ============================================================

# reporting.py está en backend/, así que el root del proyecto es parent de ese dir.
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
TEMPLATE_PATH: Final[Path] = (
    _PROJECT_ROOT / "frontend" / "templates" / "filtered_report.html"
)


def write_interactive_html(
    path: str,
    rows: Iterable[Mapping[str, object]],
    *,
    title: str = "Plex Movies Cleaner — Informe interactivo",
    subtitle: str = "Vista rápida de las películas marcadas como DELETE / MAYBE.",
) -> None:
    """
    Genera un HTML usando una plantilla externa en
    `frontend/templates/filtered_report.html`.

    La plantilla debe contener los placeholders:
      - __TITLE__
      - __SUBTITLE__
      - __ROWS_JSON__
    """
    rows_list = list(rows)

    processed_rows: list[dict[str, object]] = [
        {
            "poster_url": r.get("poster_url"),
            "library": r.get("library"),
            "title": r.get("title"),
            "year": r.get("year"),
            "imdb_rating": r.get("imdb_rating"),
            "rt_score": r.get("rt_score"),
            "imdb_votes": r.get("imdb_votes"),
            "decision": r.get("decision"),
            "reason": r.get("reason"),
            "misidentified_hint": r.get("misidentified_hint"),
            "file": r.get("file"),
        }
        for r in rows_list
    ]

    rows_json = json.dumps(processed_rows, ensure_ascii=False)
    # Escape defensivo para evitar que una cadena contenga </script
    rows_json_safe = rows_json.replace("</script", "<\\/script")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Plantilla HTML no encontrada en {TEMPLATE_PATH}. "
            "Asegúrate de crear frontend/templates/filtered_report.html"
        )

    try:
        html_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        _logger.error(f"No se pudo leer la plantilla HTML {TEMPLATE_PATH}: {exc}")
        raise

    html = (
        html_template
        .replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__ROWS_JSON__", rows_json_safe)
    )

    # Escritura atómica del HTML
    pathp = Path(path)
    dirpath = pathp.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(dirpath),
            newline="",
        ) as tf:
            tf.write(html)
            temp_name = tf.name

        os.replace(temp_name, str(pathp))
        _logger.info(f"Informe HTML interactivo escrito en {path}")
    except Exception as exc:
        _logger.error(f"Error escribiendo informe HTML en {path}: {exc}")