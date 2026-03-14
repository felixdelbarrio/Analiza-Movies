import { useState } from "react";
import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "../app/app-context";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { findDuplicates } from "../lib/data";
import type { ReportRow } from "../lib/types";

export function DuplicatesPage() {
  const { reportAll } = useOutletContext<AppOutletContext>();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const duplicateRows = findDuplicates(reportAll);
  const row = duplicateRows[selectedIndex] ?? duplicateRows[0] ?? null;

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Duplicados"
        title="Conflictos por IMDb ID"
        description="Una vista enfocada en títulos repetidos, útil para detectar redundancia entre bibliotecas y ajustar la curación."
      />

      <div className="split-layout">
        <SectionCard
          title={`Duplicados detectados (${duplicateRows.length})`}
          eyebrow="Control"
          className="split-layout__main"
        >
          <VirtualTable<ReportRow>
            columns={[
              { key: "title", label: "Título", width: "40%", render: (current) => String(current.title || "—") },
              { key: "library", label: "Biblioteca", width: "26%", render: (current) => String(current.library || "—") },
              { key: "imdb", label: "IMDb ID", width: "18%", render: (current) => String(current.imdb_id || "—") },
              { key: "dup", label: "Veces", width: "16%", render: (current) => String(current.dup_count || "—") }
            ]}
            onSelect={setSelectedIndex}
            rows={duplicateRows}
            selectedIndex={selectedIndex}
          />
        </SectionCard>

        <div className="split-layout__aside">
          <MovieDetailPanel row={row} />
        </div>
      </div>
    </div>
  );
}
