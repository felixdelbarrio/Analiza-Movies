from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from backend import logger as _logger


def delete_files_from_rows(
    rows: object,
    delete_dry_run: bool,
) -> tuple[int, int, list[str]]:
    """
    Borra físicamente archivos según las rutas de la columna 'file'.

    Parámetros:
      - rows: DataFrame con al menos columnas 'file' y 'title', o bien
              cualquier iterable de objetos tipo dict/Series con esas claves.
      - delete_dry_run:
          * True  -> solo log, NO borra nada.
          * False -> borra de verdad.

    Devuelve:
      - num_ok: número de borrados correctos.
      - num_error: número de errores al borrar.
      - logs: lista de mensajes de log.
    """
    num_ok = 0
    num_error = 0
    logs: list[str] = []

    # Soportamos tanto pandas.DataFrame (rows.iterrows()) como un iterable genérico
    iterator: Iterable[object] | None = None

    try:
        import pandas as _pd  # import local para no requerir pandas al importar el módulo

        if isinstance(rows, _pd.DataFrame):
            iterator = (row for _, row in rows.iterrows())
    except Exception:
        iterator = None

    if iterator is None:
        if isinstance(rows, Iterable):
            iterator = rows
        else:
            # Coincide con el comportamiento previo: iter(rows) habría lanzado TypeError igualmente
            raise TypeError("rows must be a pandas.DataFrame or an iterable of row-like objects")

    for row in iterator:
        # row puede ser una pandas.Series (tiene .get) o un dict-like
        file_path: object | None
        title: object | None

        # file
        try:
            if hasattr(row, "get"):
                file_path = row.get("file")  # type: ignore[call-arg]
            else:
                file_path = row["file"]  # type: ignore[index]
        except Exception:
            file_path = None

        # title
        try:
            if hasattr(row, "get"):
                title = row.get("title")  # type: ignore[call-arg]
            else:
                # Puede fallar si row no soporta indexado por clave; lo capturamos abajo.
                title = row["title"]  # type: ignore[index]
        except Exception:
            title = None

        title_str = str(title) if title is not None else "<no title>"

        if not file_path:
            msg = f"[SKIP] {title_str} -> sin ruta de archivo"
            _logger.info(msg)
            logs.append(msg)
            continue

        try:
            p = Path(str(file_path)).resolve()
        except Exception as exc:
            msg = f"[SKIP] {title_str} -> ruta inválida: {file_path} ({exc})"
            _logger.warning(msg)
            logs.append(msg)
            continue

        if not p.exists():
            msg = f"[SKIP] {title_str} -> archivo no existe: {p}"
            _logger.info(msg)
            logs.append(msg)
            continue

        if not p.is_file():
            msg = f"[SKIP] {title_str} -> no es un fichero: {p}"
            _logger.warning(msg)
            logs.append(msg)
            continue

        if delete_dry_run:
            msg = f"[DRY RUN] {title_str} -> NO se borra: {p}"
            _logger.info(msg)
            logs.append(msg)
            continue

        try:
            p.unlink()
            msg = f"[OK] BORRADO {title_str} -> {p}"
            _logger.info(msg)
            logs.append(msg)
            num_ok += 1
        except Exception as exc:
            msg = f"[ERROR] {title_str}: {exc}"
            _logger.error(msg)
            logs.append(msg)
            num_error += 1

    return num_ok, num_error, logs