from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from frontend import front_logger as _logger

from frontend.config_front_artifacts import REPORT_FILTERED_PATH

from frontend.config_front_base import (
    DELETE_DRY_RUN,
    DELETE_REQUIRE_CONFIRM,
    SILENT_MODE,
)

# ============================================================================
# Helpers
# ============================================================================


def _as_rows_iter(rows: object) -> Iterable[Mapping[str, Any]]:
    """
    Normaliza el input a un iterable de mappings.

    Acepta:
      - list[dict]
      - iterable de dict-like
      - pandas.DataFrame (si está disponible)
    """
    if rows is None:
        return []

    # Pandas DataFrame -> records
    try:
        import pandas as pd  # type: ignore[import-not-found]

        if isinstance(rows, pd.DataFrame):
            return rows.to_dict(orient="records")
    except Exception:
        pass

    if isinstance(rows, Mapping):
        return [rows]

    if isinstance(rows, (list, tuple)):
        return rows

    if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
        return rows  # type: ignore[return-value]

    return []


def _safe_path_from_row(row: Mapping[str, Any]) -> Path | None:
    """
    Extrae y normaliza la ruta del fichero desde una fila.

    Espera columna:
      - file (ruta)
    """
    raw = row.get("file")
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    # Nota: En DLNA, "library/title.ext" NO será ruta real; eso se filtrará por exists().
    try:
        p = Path(s).expanduser()
    except Exception:
        return None

    return p


def _is_probably_safe_file(path: Path) -> bool:
    """
    Validación conservadora:
    - Debe existir
    - Debe ser fichero (no directorio)
    """
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


# ============================================================================
# API principal usada por el dashboard
# ============================================================================


def delete_files_from_rows(
    rows: object,
    delete_dry_run: bool,
) -> tuple[int, int, list[str]]:
    """
    Borra físicamente archivos según las filas seleccionadas.

    Input:
      - rows: list[dict] / iterable dict-like / pandas.DataFrame
      - delete_dry_run: si True, NO borra; solo simula.

    Output:
      - (ok, err, logs)

    Notas:
    - Esta función NO pregunta confirmación (eso es responsabilidad del caller/UI).
    - Filtra filas que no apunten a ficheros reales.
    """
    logs: list[str] = []
    ok = 0
    err = 0

    it = _as_rows_iter(rows)

    for i, row_obj in enumerate(it, start=1):
        if not isinstance(row_obj, Mapping):
            err += 1
            logs.append(f"[DELETE] row#{i}: formato inválido (no mapping): {type(row_obj)}")
            continue

        path = _safe_path_from_row(row_obj)
        title = str(row_obj.get("title") or "").strip()

        if path is None:
            err += 1
            logs.append(f"[DELETE] row#{i}: sin 'file' válido (title={title!r})")
            continue

        # Resolver (sin obligar: algunos paths pueden ser relativos)
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path

        if not _is_probably_safe_file(resolved):
            # No lo contamos como error duro: normalmente serán filas DLNA o paths no montados.
            logs.append(f"[DELETE] row#{i}: skip (no existe/no es fichero): {resolved} (title={title!r})")
            continue

        if delete_dry_run:
            ok += 1
            logs.append(f"[DELETE][DRY] would delete: {resolved} (title={title!r})")
            continue

        try:
            os.remove(resolved)
            ok += 1
            logs.append(f"[DELETE] deleted: {resolved} (title={title!r})")
        except Exception as exc:
            err += 1
            logs.append(f"[DELETE] ERROR deleting {resolved} (title={title!r}): {exc!r}")

    return ok, err, logs


# ============================================================================
# CLI helper (no dashboard): borrar desde report_filtered.csv
# ============================================================================


def _read_filtered_rows(csv_path: str) -> list[dict[str, Any]]:
    """
    Lee el CSV filtrado y devuelve lista de dicts.

    - Intenta pandas si está disponible.
    - Fallback a csv.DictReader si pandas no existe.
    """
    p = Path(csv_path)
    if not p.exists():
        return []

    # 1) pandas (si existe)
    try:
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        return df.to_dict(orient="records")
    except Exception:
        pass

    # 2) fallback estándar
    import csv

    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(dict(r))
    return out


def _normalize_rows_for_delete(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reduce filas a claves mínimas:
      - file
      - title
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "file": r.get("file", ""),
                "title": r.get("title", ""),
            }
        )
    return out


def _count_existing_files(rows: Iterable[dict[str, Any]]) -> int:
    """
    Cuenta cuántas filas apuntan a ficheros existentes.
    """
    n = 0
    for r in rows:
        raw = str(r.get("file") or "").strip()
        if not raw:
            continue
        try:
            p = Path(raw).expanduser().resolve()
        except Exception:
            continue
        if _is_probably_safe_file(p):
            n += 1
    return n


def run_delete_from_report_filtered(
    *,
    csv_path: str = REPORT_FILTERED_PATH,
    delete_dry_run: bool | None = None,
    require_confirm: bool | None = None,
) -> None:
    """
    Ejecuta borrado (o dry-run) a partir del CSV filtrado.

    - Prompt SIEMPRE visible (progress + input).
    - Si no hay filas, informa y sale.
    - delete_dry_run/require_confirm por defecto vienen de config.py.
    """
    dry = DELETE_DRY_RUN if delete_dry_run is None else bool(delete_dry_run)
    confirm = DELETE_REQUIRE_CONFIRM if require_confirm is None else bool(require_confirm)

    _logger.progress("[DELETE] Inicio")

    rows_raw = _read_filtered_rows(csv_path)
    if not rows_raw:
        _logger.progress(f"[DELETE] No hay filas en '{csv_path}' (o no existe).")
        return

    rows = _normalize_rows_for_delete(rows_raw)
    existing = _count_existing_files(rows)

    mode = "DRY_RUN" if dry else "REAL_DELETE"
    _logger.progress(f"[DELETE] Fuente: {csv_path}")
    _logger.progress(f"[DELETE] Filas: {len(rows)} | ficheros existentes: {existing} | modo={mode}")

    if existing == 0:
        _logger.progress("[DELETE] Nada que borrar (no se detectan ficheros reales en disco).")
        return

    if confirm:
        _logger.progress(
            "[DELETE] Confirmación requerida.\n"
            "  - Escribe 'DELETE' para continuar\n"
            "  - Enter para cancelar"
        )
        ans = input("> ").strip()
        if ans != "DELETE":
            _logger.progress("[DELETE] Cancelado por el usuario.")
            return

    num_ok, num_err, logs = delete_files_from_rows(rows, delete_dry_run=dry)

    # En SILENT_MODE evitamos spam
    if not SILENT_MODE:
        for line in logs:
            _logger.info(line)

    _logger.progress(f"[DELETE] Fin | ok={num_ok} err={num_err} dry_run={dry}")