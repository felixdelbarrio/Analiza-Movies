import { startTransition, useDeferredValue, useState } from "react";
import { Search } from "lucide-react";
import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "../app/app-context";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { uniqueValues } from "../lib/data";
import type { ReportRow } from "../lib/types";

export function LibraryPage() {
  const { reportAll } = useOutletContext<AppOutletContext>();
  const [search, setSearch] = useState("");
  const [libraries, setLibraries] = useState<string[]>([]);
  const [decisions, setDecisions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const deferredSearch = useDeferredValue(search);

  const filteredRows = reportAll.filter((row) => {
    const matchesLibrary =
      !libraries.length ||
      (typeof row.library === "string" && libraries.includes(row.library));
    const decision = String(row.decision || "").toUpperCase();
    const matchesDecision = !decisions.length || decisions.includes(decision);
    const searchValue = deferredSearch.trim().toLowerCase();
    const matchesSearch =
      !searchValue ||
      [row.title, row.library, row.imdb_id, row.file]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(searchValue));
    return matchesLibrary && matchesDecision && matchesSearch;
  });

  const row = filteredRows[selectedIndex] ?? filteredRows[0] ?? null;

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Biblioteca completa"
        title="Exploración con enfoque cinematográfico"
        description="Filtra, compara y abre fichas con contexto editorial sin renunciar a velocidad ni densidad informativa."
      />

      <SectionCard title="Filtros" eyebrow="Exploración">
        <div className="filter-grid">
          <label className="search-field">
            <Search size={16} />
            <input
              onChange={(event) =>
                startTransition(() => {
                  setSearch(event.target.value);
                  setSelectedIndex(0);
                })
              }
              placeholder="Busca por título, biblioteca, IMDb o ruta"
              value={search}
            />
          </label>

          <label>
            <span>Biblioteca</span>
            <select
              multiple
              onChange={(event) => {
                const values = Array.from(event.target.selectedOptions).map((option) => option.value);
                setLibraries(values);
                setSelectedIndex(0);
              }}
              value={libraries}
            >
              {uniqueValues(reportAll, "library").map((library) => (
                <option key={library} value={library}>
                  {library}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Decisión</span>
            <select
              multiple
              onChange={(event) => {
                const values = Array.from(event.target.selectedOptions).map((option) => option.value);
                setDecisions(values);
                setSelectedIndex(0);
              }}
              value={decisions}
            >
              {["KEEP", "MAYBE", "DELETE", "UNKNOWN"].map((decision) => (
                <option key={decision} value={decision}>
                  {decision}
                </option>
              ))}
            </select>
          </label>
        </div>
      </SectionCard>

      <div className="split-layout">
        <SectionCard title={`Catálogo (${filteredRows.length})`} eyebrow="Listado" className="split-layout__main">
          <VirtualTable<ReportRow>
            columns={[
              { key: "title", label: "Título", width: "36%", render: (current) => String(current.title || "—") },
              { key: "year", label: "Año", width: "10%", render: (current) => String(current.year || "—") },
              { key: "library", label: "Biblioteca", width: "22%", render: (current) => String(current.library || "—") },
              { key: "decision", label: "Decisión", width: "14%", render: (current) => String(current.decision || "—") },
              { key: "imdb", label: "IMDb", width: "10%", render: (current) => String(current.imdb_rating || "—") },
              {
                key: "size",
                label: "GB",
                width: "8%",
                render: (current) =>
                  current.file_size_gb ? Number(current.file_size_gb).toFixed(1) : "—"
              }
            ]}
            onSelect={setSelectedIndex}
            rows={filteredRows}
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
