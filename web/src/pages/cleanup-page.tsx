import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "../app/app-context";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { runDeleteAction } from "../lib/api";
import type { ReportRow } from "../lib/types";

export function CleanupPage() {
  const { reportFiltered } = useOutletContext<AppOutletContext>();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [dryRun, setDryRun] = useState(true);
  const selectedRow = reportFiltered[selectedIndex] ?? reportFiltered[0] ?? null;

  const deleteMutation = useMutation({
    mutationFn: async () => {
      const rows = selectedRow ? [selectedRow] : [];
      return runDeleteAction(rows, dryRun);
    }
  });

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Limpieza"
        title="Ejecución segura sobre candidatos DELETE/MAYBE"
        description="La acción destructiva sale del navegador y pasa por la API, con simulación previa y trazabilidad del resultado."
        actions={
          <label className="toggle-pill">
            <input
              checked={dryRun}
              onChange={(event) => setDryRun(event.target.checked)}
              type="checkbox"
            />
            <span>{dryRun ? "Modo simulación" : "Borrado real"}</span>
          </label>
        }
      />

      <div className="split-layout">
        <SectionCard
          title={`Cola de limpieza (${reportFiltered.length})`}
          eyebrow="Acción"
          className="split-layout__main"
          actions={
            <button
              className="primary-button"
              disabled={!selectedRow || deleteMutation.isPending}
              onClick={() => deleteMutation.mutate()}
              type="button"
            >
              {dryRun ? "Simular borrado" : "Borrar selección"}
            </button>
          }
        >
          <VirtualTable<ReportRow>
            columns={[
              { key: "title", label: "Título", width: "36%", render: (row) => String(row.title || "—") },
              { key: "library", label: "Biblioteca", width: "24%", render: (row) => String(row.library || "—") },
              { key: "decision", label: "Decisión", width: "14%", render: (row) => String(row.decision || "—") },
              { key: "imdb", label: "IMDb", width: "10%", render: (row) => String(row.imdb_rating || "—") },
              {
                key: "size",
                label: "GB",
                width: "8%",
                render: (row) => (row.file_size_gb ? Number(row.file_size_gb).toFixed(1) : "—")
              },
              { key: "file", label: "Ruta", width: "28%", render: (row) => String(row.file || "—") }
            ]}
            onSelect={setSelectedIndex}
            rows={reportFiltered}
            selectedIndex={selectedIndex}
          />
        </SectionCard>

        <div className="split-layout__aside">
          <MovieDetailPanel row={selectedRow} />
          <SectionCard title="Resultado" eyebrow="Logs">
            {deleteMutation.data ? (
              <div className="log-stack">
                <p>
                  OK: <strong>{deleteMutation.data.ok}</strong> · ERR: <strong>{deleteMutation.data.err}</strong>
                </p>
                <pre>{deleteMutation.data.logs.join("\n")}</pre>
              </div>
            ) : (
              <p>Aquí aparecerá el resultado de la simulación o del borrado real.</p>
            )}
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
