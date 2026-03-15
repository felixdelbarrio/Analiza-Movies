export interface CsvColumn<T> {
  header: string;
  value: (row: T) => unknown;
}

function stringifyCell(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function escapeCell(value: string) {
  return `"${value.replaceAll('"', '""')}"`;
}

export function downloadCsv<T>(
  filename: string,
  columns: CsvColumn<T>[],
  rows: T[],
  delimiter = ";"
) {
  const headerRow = columns.map((column) => escapeCell(column.header)).join(delimiter);
  const bodyRows = rows.map((row) =>
    columns
      .map((column) => escapeCell(stringifyCell(column.value(row))))
      .join(delimiter)
  );
  const payload = `\uFEFF${[headerRow, ...bodyRows].join("\n")}`;
  const blob = new Blob([payload], {
    type: "text/csv;charset=utf-8;"
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
