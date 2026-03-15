import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef, type KeyboardEvent, type ReactNode } from "react";

import { useI18n } from "../i18n/provider";

export interface VirtualTableColumn<T> {
  key: string;
  label: string;
  width?: string;
  align?: "left" | "center" | "right";
  sortable?: boolean;
  render: (row: T) => ReactNode;
}

export interface VirtualTableSortState {
  key: string;
  direction: "asc" | "desc";
}

interface VirtualTableProps<T> {
  rows: T[];
  columns: VirtualTableColumn<T>[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  maxHeight?: number;
  fillHeight?: boolean;
  sortState?: VirtualTableSortState | null;
  onSortChange?: (sortState: VirtualTableSortState) => void;
  rowTone?: (row: T) => "neutral" | "keep" | "maybe" | "delete";
}

export function VirtualTable<T>({
  rows,
  columns,
  selectedIndex,
  onSelect,
  maxHeight = 620,
  fillHeight = false,
  sortState = null,
  onSortChange,
  rowTone
}: VirtualTableProps<T>) {
  const { t } = useI18n();
  const parentRef = useRef<HTMLDivElement | null>(null);
  const gridTemplateColumns = columns.map((column) => column.width ?? "1fr").join(" ");
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 64,
    overscan: 10
  });

  if (!rows.length) {
    return <div className="virtual-table__empty">{t("table.empty")}</div>;
  }

  function handleRowKeyDown(event: KeyboardEvent<HTMLDivElement>, index: number) {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    onSelect(index);
  }

  return (
    <div className={`virtual-table${fillHeight ? " virtual-table--fill" : ""}`}>
      <div className="virtual-table__header" style={{ gridTemplateColumns }}>
        {columns.map((column) => (
          column.sortable && onSortChange ? (
            <button
              key={column.key}
              className={`virtual-table__sort${
                sortState?.key === column.key ? " is-active" : ""
              }`}
              onClick={() =>
                onSortChange({
                  key: column.key,
                  direction:
                    sortState?.key === column.key && sortState.direction === "asc"
                      ? "desc"
                      : "asc"
                })
              }
              type="button"
            >
              <span>{column.label}</span>
              {sortState?.key === column.key ? (
                sortState.direction === "asc" ? (
                  <ArrowUp size={14} />
                ) : (
                  <ArrowDown size={14} />
                )
              ) : (
                <ArrowUpDown size={14} />
              )}
            </button>
          ) : (
            <span key={column.key} className="virtual-table__header-label">
              {column.label}
            </span>
          )
        ))}
      </div>
      <div
        className="virtual-table__body"
        ref={parentRef}
        style={fillHeight ? undefined : { maxHeight }}
      >
        <div
          className="virtual-table__canvas"
          style={{ height: `${rowVirtualizer.getTotalSize()}px` }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const row = rows[virtualRow.index];
            const selected = virtualRow.index === selectedIndex;
            const tone = rowTone?.(row) ?? "neutral";
            return (
              <div
                key={virtualRow.key}
                className={`virtual-table__row${selected ? " selected" : ""}`}
                data-columns={columns.length}
                data-tone={tone}
                onClick={() => onSelect(virtualRow.index)}
                onKeyDown={(event) => handleRowKeyDown(event, virtualRow.index)}
                role="button"
                style={{
                  transform: `translateY(${virtualRow.start}px)`,
                  gridTemplateColumns
                }}
                tabIndex={0}
              >
                {columns.map((column) => (
                  <span
                    key={column.key}
                    className={`virtual-table__cell virtual-table__cell--${column.align ?? "left"}`}
                  >
                    {column.render(row) ?? t("app.empty_dash")}
                  </span>
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
