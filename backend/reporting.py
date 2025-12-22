from __future__ import annotations

"""
backend/reporting.py

Responsabilidades:
- Escritura de CSVs de forma ATÓMICA.
- Soporte STREAMING (fila a fila) para grandes catálogos.
- Mantener compatibilidad con APIs legacy (write_all_csv, etc.).
- Evitar explosiones de memoria en SILENT_MODE.

Filosofía:
- Nunca dejar CSVs corruptos.
- Nunca perder datos por interrupciones.
- Logs mínimos y claros.
"""

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from backend import logger as _logger


# ============================================================================
# FIELDNAMES ESTÁNDAR
# ============================================================================

_STANDARD_REPORT_FIELDS: Final[list[str]] = [
    "library",
    "title",
    "year",
    "plex_rating",
    "imdb_rating",
    "imdb_votes",
    "rt_score",
    "metacritic_score",
    "decision",
    "reason",
    "misidentified_hint",
    "file",
    "file_size",
    "rating_key",
    "guid",
    "imdb_id",
    "poster_url",
    "trailer_url",
    "thumb",
    "omdb_json",
    "wikidata_id",
    "wikipedia_title",
]

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


# ============================================================================
# CSV ATÓMICO EN STREAMING
# ============================================================================

class CSVAtomicWriter:
    """
    Writer incremental con commit atómico.

    - Escribe en fichero temporal.
    - write_row() permite streaming.
    - close() hace os.replace() (atómico).

    commit_if_empty:
      - True  → crea CSV aunque solo tenga cabecera
      - False → si no hay filas, NO genera fichero
    """

    def __init__(
        self,
        path: str,
        *,
        fieldnames: list[str],
        kind_label: str = "CSV",
        commit_if_empty: bool = True,
    ) -> None:
        self._path = str(path)
        self._pathp = Path(path)
        self._dir = self._pathp.parent
        self._fieldnames = list(fieldnames)
        self._kind_label = kind_label
        self._commit_if_empty = commit_if_empty

        self._tmp_name: str | None = None
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._rows_written: int = 0
        self._is_open = False

    def __enter__(self) -> CSVAtomicWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def rows_written(self) -> int:
        return self._rows_written

    # --------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return

        self._dir.mkdir(parents=True, exist_ok=True)

        tf = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self._dir),
            newline="",
        )

        self._fh = tf
        self._tmp_name = tf.name
        self._writer = csv.DictWriter(
            tf,
            fieldnames=self._fieldnames,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._is_open = True

    # --------------------------------------------------------

    def write_row(self, row: Mapping[str, object]) -> None:
        if not self._is_open:
            self.open()

        try:
            assert self._writer is not None
            self._writer.writerow(dict(row))
            self._rows_written += 1
        except Exception as exc:
            _logger.error(f"Error escribiendo fila en {self._kind_label}: {exc}", always=True)

    # --------------------------------------------------------

    def close(self) -> None:
        if not self._is_open:
            return

        tmp_name = self._tmp_name
        fh = self._fh
        rows_written = self._rows_written
        commit_if_empty = self._commit_if_empty

        self._is_open = False
        self._tmp_name = None
        self._fh = None
        self._writer = None

        try:
            if fh is not None:
                try:
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except Exception:
                        pass
                finally:
                    fh.close()

            if not tmp_name:
                return

            if rows_written == 0 and not commit_if_empty:
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
                _logger.info(f"{self._kind_label} vacío: no se genera fichero ({self._path})")
                return

            os.replace(tmp_name, str(self._pathp))
            _logger.info(f"{self._kind_label} escrito en {self._path}")

        except Exception as exc:
            _logger.error(f"Error cerrando {self._kind_label} en {self._path}: {exc}", always=True)
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass


# ============================================================================
# FACTORIES (STREAMING)
# ============================================================================

def open_all_csv_writer(path: str) -> CSVAtomicWriter:
    return CSVAtomicWriter(
        path,
        fieldnames=_STANDARD_REPORT_FIELDS,
        kind_label="CSV completo",
        commit_if_empty=True,
    )


def open_filtered_csv_writer_only_if_rows(path: str) -> CSVAtomicWriter:
    return CSVAtomicWriter(
        path,
        fieldnames=_STANDARD_REPORT_FIELDS,
        kind_label="CSV filtrado",
        commit_if_empty=False,
    )


def open_suggestions_csv_writer(path: str) -> CSVAtomicWriter:
    return CSVAtomicWriter(
        path,
        fieldnames=_STANDARD_SUGGESTION_FIELDS,
        kind_label="CSV de sugerencias",
        commit_if_empty=True,
    )


# ============================================================================
# API LEGACY (compatibilidad)
# ============================================================================

def write_all_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    with open_all_csv_writer(path) as w:
        for r in rows:
            w.write_row(r)


def write_filtered_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    with open_filtered_csv_writer_only_if_rows(path) as w:
        for r in rows:
            w.write_row(r)


def write_suggestions_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    with open_suggestions_csv_writer(path) as w:
        for r in rows:
            w.write_row(r)