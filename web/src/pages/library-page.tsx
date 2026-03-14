import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useState
} from "react";
import { Search } from "lucide-react";

import { useAppContext } from "../app/use-app-context";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { useReportAll } from "../hooks/use-dashboard-data";
import { translateDecision } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { uniqueValues } from "../lib/data";
import type { ReportRow } from "../lib/types";

export function LibraryPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [search, setSearch] = useState("");
  const [libraries, setLibraries] = useState<string[]>([]);
  const [decisions, setDecisions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const deferredSearch = useDeferredValue(search);

  const libraryOptions = useMemo(() => uniqueValues(reportAll, "library", locale), [locale, reportAll]);
  const filteredRows = useMemo(
    () =>
      reportAll.filter((row) => {
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
      }),
    [decisions, deferredSearch, libraries, reportAll]
  );

  useEffect(() => {
    if (selectedIndex >= filteredRows.length) {
      setSelectedIndex(0);
    }
  }, [filteredRows.length, selectedIndex]);

  const row = filteredRows[selectedIndex] ?? filteredRows[0] ?? null;

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("library.hero.eyebrow")}
        title={t("library.hero.title")}
        description={t("library.hero.description")}
      />

      {reportAllQuery.isLoading ? (
        <PageState
          eyebrow={t("stage.connecting")}
          title={t("library.loading.title")}
          message={t("library.loading.message")}
        />
      ) : null}

      {reportAllQuery.error ? (
        <PageState
          action={
            <button
              className="primary-button"
              onClick={() => reportAllQuery.refetch()}
              type="button"
            >
              {t("app.action.retry")}
            </button>
          }
          eyebrow={t("stage.failed")}
          title={t("library.error.title")}
          message={t("library.error.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && !reportAll.length ? (
        <PageState
          eyebrow={t("library.empty.eyebrow")}
          title={t("library.empty.title")}
          message={t("library.empty.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && reportAll.length ? (
        <>
          <SectionCard title={t("library.filters.title")} eyebrow={t("library.filters.eyebrow")}>
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
                  placeholder={t("library.search.placeholder")}
                  value={search}
                />
              </label>

              <label>
                <span>{t("library.filter.library")}</span>
                <select
                  multiple
                  onChange={(event) => {
                    const values = Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    );
                    setLibraries(values);
                    setSelectedIndex(0);
                  }}
                  value={libraries}
                >
                  {libraryOptions.map((library) => (
                    <option key={library} value={library}>
                      {library}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>{t("library.filter.decision")}</span>
                <select
                  multiple
                  onChange={(event) => {
                    const values = Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    );
                    setDecisions(values);
                    setSelectedIndex(0);
                  }}
                  value={decisions}
                >
                  {["KEEP", "MAYBE", "DELETE", "UNKNOWN"].map((decision) => (
                    <option key={decision} value={decision}>
                      {translateDecision(decision, t)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </SectionCard>

          <div className="split-layout">
            <SectionCard
              title={t("library.catalog.title", { count: filteredRows.length })}
              eyebrow={t("library.catalog.eyebrow")}
              className="split-layout__main"
            >
              <VirtualTable<ReportRow>
                columns={[
                  {
                    key: "title",
                    label: t("column.title"),
                    width: "36%",
                    render: (current) => String(current.title || t("app.empty_dash"))
                  },
                  {
                    key: "year",
                    label: t("column.year"),
                    width: "10%",
                    render: (current) => String(current.year || t("app.empty_dash"))
                  },
                  {
                    key: "library",
                    label: t("column.library"),
                    width: "22%",
                    render: (current) => String(current.library || t("app.empty_dash"))
                  },
                  {
                    key: "decision",
                    label: t("column.decision"),
                    width: "14%",
                    render: (current) => translateDecision(String(current.decision || "UNKNOWN"), t)
                  },
                  {
                    key: "imdb",
                    label: t("column.imdb"),
                    width: "10%",
                    render: (current) => String(current.imdb_rating || t("app.empty_dash"))
                  },
                  {
                    key: "size",
                    label: t("column.size"),
                    width: "8%",
                    render: (current) =>
                      current.file_size_gb
                        ? Number(current.file_size_gb).toLocaleString(locale, {
                            minimumFractionDigits: 1,
                            maximumFractionDigits: 1
                          })
                        : t("app.empty_dash")
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
        </>
      ) : null}
    </div>
  );
}
