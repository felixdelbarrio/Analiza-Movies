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

Mejoras aplicadas en esta revisión:
- Opción A: alinear _STANDARD_SUGGESTION_FIELDS con metadata_fix.py (campos nuevos)
  y mantener imdb_rating/imdb_votes.
- FIX: añadir columna "action" al CSV de sugerencias (metadata_fix.py la emite).
- Robustez: abortar commit si ocurre una excepción dentro del context manager
  (no reemplaza el CSV final; deja el anterior intacto).
- Diagnóstico ligero: aviso (debug) si llegan keys extra no contempladas en fieldnames,
  SIN romper compatibilidad (extrasaction="ignore" se mantiene).
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, Optional, TextIO

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

# ✅ Opción A: sugerencias alineadas con metadata_fix.py “nuevo”
# ✅ FIX: incluir "action" (metadata_fix.py la emite)
_STANDARD_SUGGESTION_FIELDS: Final[list[str]] = [
    "plex_guid",
    "library",
    "context_lang",
    "plex_original_title",
    "plex_imdb_id",
    "omdb_imdb_id",
    "plex_title",
    "plex_year",
    "omdb_title",
    "omdb_year",
    "imdb_rating",
    "imdb_votes",
    "action",
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

    Comportamiento ante excepción (en context manager):
      - Si hay excepción dentro del `with`, NO se comitea el CSV final
        (se borra el tmp). Así evitamos “CSV truncado pero válido”.
    """

    def __init__(
        self,
        path: str,
        *,
        fieldnames: list[str],
        kind_label: str = "CSV",
        commit_if_empty: bool = True,
        warn_on_extras: bool = True,
    ) -> None:
        self._path: str = str(path)
        self._pathp: Path = Path(path)
        self._dir: Path = self._pathp.parent
        self._fieldnames: list[str] = list(fieldnames)
        self._fieldnames_set: set[str] = set(fieldnames)
        self._kind_label: str = str(kind_label)
        self._commit_if_empty: bool = bool(commit_if_empty)
        self._warn_on_extras: bool = bool(warn_on_extras)

        self._tmp_name: str | None = None
        # Tipamos como TextIO real usando mkstemp + os.fdopen.
        self._fh: Optional[TextIO] = None
        self._writer: csv.DictWriter | None = None

        self._rows_written: int = 0
        self._is_open: bool = False

        # Avisos de claves extra (evita spam)
        self._warned_extra_keys: set[str] = set()

    def __enter__(self) -> "CSVAtomicWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Si hubo excepción, abortamos el commit (no reemplazamos el CSV final)
        if exc_type is not None:
            self.abort()
            return
        self.close()

    @property
    def rows_written(self) -> int:
        return self._rows_written

    # --------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return

        self._dir.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            prefix=".tmp_", suffix=".csv", dir=str(self._dir), text=True
        )
        self._tmp_name = tmp_name

        try:
            fh = os.fdopen(fd, "w", encoding="utf-8", newline="")
        except Exception:
            # si falla, cerramos el fd y limpiamos el fichero
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(tmp_name)
            except Exception:
                pass
            self._tmp_name = None
            raise

        self._fh = fh
        self._writer = csv.DictWriter(
            fh,
            fieldnames=self._fieldnames,
            extrasaction="ignore",  # compat + forward fields
        )
        self._writer.writeheader()
        self._is_open = True

    # --------------------------------------------------------

    def _maybe_warn_extras(self, row: Mapping[str, object]) -> None:
        if not self._warn_on_extras:
            return
        try:
            keys = row.keys()
        except Exception:
            return

        for k in keys:
            if k not in self._fieldnames_set and k not in self._warned_extra_keys:
                self._warned_extra_keys.add(k)
                _logger.debug(
                    f"{self._kind_label}: clave extra ignorada en CSV (no está en fieldnames): {k!r}"
                )

    def write_row(self, row: Mapping[str, object]) -> None:
        if not self._is_open:
            self.open()

        writer = self._writer
        if writer is None:
            _logger.error(f"{self._kind_label}: writer no inicializado", always=True)
            return

        self._maybe_warn_extras(row)

        try:
            writer.writerow(dict(row))
            self._rows_written += 1
        except Exception as exc:
            _logger.error(
                f"Error escribiendo fila en {self._kind_label}: {exc!r}", always=True
            )

    # --------------------------------------------------------

    def abort(self) -> None:
        """
        Aborta el commit final: cierra el file handle y borra el temporal.
        Mantiene el CSV final anterior intacto.
        """
        if not self._is_open:
            return

        tmp_name = self._tmp_name
        fh = self._fh

        self._is_open = False
        self._tmp_name = None
        self._fh = None
        self._writer = None

        try:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
        finally:
            _logger.info(
                f"{self._kind_label} abortado: no se comitea fichero ({self._path})"
            )

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
                    try:
                        fh.close()
                    except Exception:
                        pass

            if not tmp_name:
                return

            if rows_written == 0 and not commit_if_empty:
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
                _logger.info(
                    f"{self._kind_label} vacío: no se genera fichero ({self._path})"
                )
                return

            os.replace(tmp_name, str(self._pathp))
            _logger.info(f"{self._kind_label} escrito en {self._path}")

        except Exception as exc:
            _logger.error(
                f"Error cerrando {self._kind_label} en {self._path}: {exc!r}",
                always=True,
            )
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
        warn_on_extras=True,
    )


def open_filtered_csv_writer_only_if_rows(path: str) -> CSVAtomicWriter:
    return CSVAtomicWriter(
        path,
        fieldnames=_STANDARD_REPORT_FIELDS,
        kind_label="CSV filtrado",
        commit_if_empty=False,
        warn_on_extras=True,
    )


def open_suggestions_csv_writer(path: str) -> CSVAtomicWriter:
    return CSVAtomicWriter(
        path,
        fieldnames=_STANDARD_SUGGESTION_FIELDS,
        kind_label="CSV de sugerencias",
        commit_if_empty=True,
        warn_on_extras=True,
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
