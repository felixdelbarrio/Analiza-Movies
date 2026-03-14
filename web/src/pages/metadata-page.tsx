import { useDeferredValue, useState } from "react";
import { Search } from "lucide-react";
import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "../app/app-context";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { filterMetadata } from "../lib/data";
import type { MetadataRow } from "../lib/types";

export function MetadataPage() {
  const { metadataRows } = useOutletContext<AppOutletContext>();
  const [search, setSearch] = useState("");
  const [libraries, setLibraries] = useState<string[]>([]);
  const [actions, setActions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const deferredSearch = useDeferredValue(search);

  const libraryValues = Array.from(
    new Set(
      metadataRows
        .map((row) => row.library)
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    )
  ).sort((left, right) => left.localeCompare(right, "es"));
  const actionValues = Array.from(
    new Set(
      metadataRows
        .map((row) => row.action)
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    )
  ).sort((left, right) => left.localeCompare(right, "es"));

  const filtered = filterMetadata(metadataRows, {
    library: libraries,
    action: actions,
    search: deferredSearch
  });
  const selected = filtered[selectedIndex] ?? filtered[0] ?? null;

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Metadata"
        title="Corrección editorial asistida"
        description="Filtra sugerencias y revisa inconsistencias antes de tocar títulos o identificadores en origen."
      />

      <SectionCard title="Filtros de sugerencias" eyebrow="Control">
        <div className="filter-grid">
          <label className="search-field">
            <Search size={16} />
            <input
              onChange={(event) => {
                setSearch(event.target.value);
                setSelectedIndex(0);
              }}
              placeholder="Buscar en sugerencias"
              value={search}
            />
          </label>
          <label>
            <span>Biblioteca</span>
            <select
              multiple
              onChange={(event) => {
                setLibraries(Array.from(event.target.selectedOptions).map((option) => option.value));
                setSelectedIndex(0);
              }}
              value={libraries}
            >
              {libraryValues.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Acción</span>
            <select
              multiple
              onChange={(event) => {
                setActions(Array.from(event.target.selectedOptions).map((option) => option.value));
                setSelectedIndex(0);
              }}
              value={actions}
            >
              {actionValues.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        </div>
      </SectionCard>

      <div className="split-layout">
        <SectionCard title={`Sugerencias (${filtered.length})`} eyebrow="Listado" className="split-layout__main">
          <VirtualTable<MetadataRow>
            columns={[
              { key: "plex_title", label: "Título Plex", width: "28%", render: (row) => String(row.plex_title || "—") },
              { key: "omdb_title", label: "Título OMDb", width: "28%", render: (row) => String(row.omdb_title || "—") },
              { key: "library", label: "Biblioteca", width: "20%", render: (row) => String(row.library || "—") },
              { key: "action", label: "Acción", width: "14%", render: (row) => String(row.action || "—") },
              { key: "rating", label: "IMDb", width: "10%", render: (row) => String(row.imdb_rating || "—") }
            ]}
            onSelect={setSelectedIndex}
            rows={filtered}
            selectedIndex={selectedIndex}
          />
        </SectionCard>

        <SectionCard title="Ficha" eyebrow="Detalle" className="split-layout__aside">
          {selected ? (
            <dl className="detail-definition-list">
              {Object.entries(selected).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{String(value ?? "—")}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p>No hay sugerencias con los filtros actuales.</p>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
