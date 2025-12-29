from __future__ import annotations

import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from frontend.config_front_artifacts import REPORT_FILTERED_PATH
from frontend.config_front_base import DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM, SILENT_MODE


class LoggerLike(Protocol):
    def progress(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...


Row = Mapping[str, Any]
Rows = Sequence[Row]


def _get_logger() -> LoggerLike:
    import frontend.front_logger as front_logger

    return cast(LoggerLike, front_logger)


# ============================================================================
# Helpers
# ============================================================================


def _to_str_key_dict(src: Mapping[Hashable, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in src.items():
        out[str(k)] = v
    return out


def _as_rows_iter(rows: object) -> list[dict[str, Any]]:
    """
    Normaliza el input a una lista de dict[str, Any].

    Acepta:
      - list/tuple de dict-like
      - iterable de dict-like
      - pandas.DataFrame / pandas.Series (si está disponible)
      - Mapping (una sola fila)
    """
    if rows is None:
        return []

    # Pandas DataFrame/Series -> records
    try:
        import pandas as pd  # type: ignore[import-not-found]

        if isinstance(rows, pd.DataFrame):
            records = rows.to_dict(orient="records")
            out_df: list[dict[str, Any]] = []
            for r in records:
                if isinstance(r, Mapping):
                    out_df.append(_to_str_key_dict(cast(Mapping[Hashable, Any], r)))
                else:
                    out_df.append({"value": r})
            return out_df

        if isinstance(rows, pd.Series):
            series_dict = rows.to_dict()
            if isinstance(series_dict, Mapping):
                return [_to_str_key_dict(cast(Mapping[Hashable, Any], series_dict))]
            return [{"value": series_dict}]
    except Exception:
        pass

    if isinstance(rows, Mapping):
        return [_to_str_key_dict(cast(Mapping[Hashable, Any], rows))]

    if isinstance(rows, (list, tuple)):
        out_list: list[dict[str, Any]] = []
        for x in rows:
            if isinstance(x, Mapping):
                out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], x)))
            else:
                out_list.append({"value": x})
        return out_list

    if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
        out_it: list[dict[str, Any]] = []
        for x in rows:
            if isinstance(x, Mapping):
                out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], x)))
            else:
                out_it.append({"value": x})
        return out_it

    return []


def _safe_path_from_row(row: Row) -> Path | None:
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

    try:
        return Path(s).expanduser()
    except Exception:
        return None


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
        path = _safe_path_from_row(row_obj)
        title = str(row_obj.get("title") or "").strip()

        if path is None:
            err += 1
            logs.append(f"[DELETE] row#{i}: sin 'file' válido (title={title!r})")
            continue

        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path

        if not _is_probably_safe_file(resolved):
            logs.append(
                f"[DELETE] row#{i}: skip (no existe/no es fichero): {resolved} (title={title!r})"
            )
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


def _read_filtered_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Lee el CSV filtrado y devuelve lista de dicts.

    - Intenta pandas si está disponible.
    - Fallback a csv.DictReader si pandas no existe.
    """
    p = csv_path if isinstance(csv_path, Path) else Path(str(csv_path))
    if not p.exists():
        return []

    # 1) pandas (si existe)
    try:
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        records = df.to_dict(orient="records")
        out_df: list[dict[str, Any]] = []
        for r in records:
            if isinstance(r, Mapping):
                out_df.append(_to_str_key_dict(cast(Mapping[Hashable, Any], r)))
            else:
                out_df.append({"value": r})
        return out_df
    except Exception:
        pass

    # 2) fallback estándar
    import csv

    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(_to_str_key_dict(cast(Mapping[Hashable, Any], r)))
    return out


def _normalize_rows_for_delete(rows: Iterable[Row]) -> list[dict[str, Any]]:
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


def _count_existing_files(rows: Iterable[Row]) -> int:
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
    csv_path: str | Path = REPORT_FILTERED_PATH,
    delete_dry_run: bool | None = None,
    require_confirm: bool | None = None,
) -> None:
    """
    Ejecuta borrado (o dry-run) a partir del CSV filtrado.

    - Prompt SIEMPRE visible (progress + input).
    - Si no hay filas, informa y sale.
    - delete_dry_run/require_confirm por defecto vienen de config.py.
    """
    logger = _get_logger()

    dry = DELETE_DRY_RUN if delete_dry_run is None else bool(delete_dry_run)
    confirm = DELETE_REQUIRE_CONFIRM if require_confirm is None else bool(require_confirm)

    logger.progress("[DELETE] Inicio")

    rows_raw = _read_filtered_rows(csv_path)
    csv_path_str = str(csv_path)

    if not rows_raw:
        logger.progress(f"[DELETE] No hay filas en '{csv_path_str}' (o no existe).")
        return

    rows = _normalize_rows_for_delete(rows_raw)
    existing = _count_existing_files(rows)

    mode = "DRY_RUN" if dry else "REAL_DELETE"
    logger.progress(f"[DELETE] Fuente: {csv_path_str}")
    logger.progress(f"[DELETE] Filas: {len(rows)} | ficheros existentes: {existing} | modo={mode}")

    if existing == 0:
        logger.progress("[DELETE] Nada que borrar (no se detectan ficheros reales en disco).")
        return

    if confirm:
        logger.progress(
            "[DELETE] Confirmación requerida.\n"
            "  - Escribe 'DELETE' para continuar\n"
            "  - Enter para cancelar"
        )
        ans = input("> ").strip()
        if ans != "DELETE":
            logger.progress("[DELETE] Cancelado por el usuario.")
            return

    num_ok, num_err, logs = delete_files_from_rows(rows, delete_dry_run=dry)

    if not SILENT_MODE:
        for line in logs:
            logger.info(line)

    logger.progress(f"[DELETE] Fin | ok={num_ok} err={num_err} dry_run={dry}")