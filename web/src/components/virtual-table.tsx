import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";

interface Column<T> {
  key: string;
  label: string;
  width?: string;
  render: (row: T) => string | number | null | undefined;
}

interface VirtualTableProps<T> {
  rows: T[];
  columns: Column<T>[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  maxHeight?: number;
}

export function VirtualTable<T>({
  rows,
  columns,
  selectedIndex,
  onSelect,
  maxHeight = 620
}: VirtualTableProps<T>) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const gridTemplateColumns = columns.map((column) => column.width ?? "1fr").join(" ");
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 52,
    overscan: 10
  });

  if (!rows.length) {
    return <div className="virtual-table__empty">No hay filas para mostrar con los filtros actuales.</div>;
  }

  return (
    <div className="virtual-table">
      <div className="virtual-table__header" style={{ gridTemplateColumns }}>
        {columns.map((column) => (
          <span key={column.key}>
            {column.label}
          </span>
        ))}
      </div>
      <div className="virtual-table__body" ref={parentRef} style={{ maxHeight }}>
        <div
          className="virtual-table__canvas"
          style={{ height: `${rowVirtualizer.getTotalSize()}px` }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const row = rows[virtualRow.index];
            const selected = virtualRow.index === selectedIndex;
            return (
              <button
                key={virtualRow.key}
                className={`virtual-table__row${selected ? " selected" : ""}`}
                data-columns={columns.length}
                style={{
                  transform: `translateY(${virtualRow.start}px)`,
                  gridTemplateColumns
                }}
                onClick={() => onSelect(virtualRow.index)}
                type="button"
              >
                {columns.map((column) => (
                  <span key={column.key}>
                    {column.render(row) ?? "—"}
                  </span>
                ))}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
