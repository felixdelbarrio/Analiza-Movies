import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useState
} from "react";
import { Search } from "lucide-react";

import { useAppContext } from "../app/use-app-context";
import { MovieDetailModal } from "../components/movie-detail-modal";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import {
  VirtualTable,
  type VirtualTableSortState
} from "../components/virtual-table";
import { useReportAll } from "../hooks/use-dashboard-data";
import { translateDecision } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import {
  REPORT_DECISION_OPTIONS,
  buildReportRowKey,
  filterReportRows,
  getDecisionTone,
  parseMaybeNumber,
  sumReportSizeGb,
  uniqueValues,
  type ReportSearchScope
} from "../lib/data";
import type { ReportRow } from "../lib/types";

const DECISION_ORDER: Record<string, number> = {
  KEEP: 0,
  MAYBE: 1,
  DELETE: 2,
  UNKNOWN: 3
};

export function LibraryPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [search, setSearch] = useState("");
  const [searchScope, setSearchScope] = useState<ReportSearchScope>("all");
  const [libraryFilter, setLibraryFilter] = useState("");
  const [decisionFilter, setDecisionFilter] = useState("");
  const [selectedRowKey, setSelectedRowKey] = useState<string | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [sortState, setSortState] = useState<VirtualTableSortState>({
    key: "title",
    direction: "asc"
  });
  const deferredSearch = useDeferredValue(search);

  const libraryOptions = useMemo(() => uniqueValues(reportAll, "library", locale), [locale, reportAll]);
  const filteredRows = useMemo(
    () =>
      filterReportRows(reportAll, {
        decision: decisionFilter,
        library: libraryFilter,
        search: deferredSearch,
        searchScope
      }),
    [decisionFilter, deferredSearch, libraryFilter, reportAll, searchScope]
  );
  const sortedRows = useMemo(() => {
    const multiplier = sortState.direction === "asc" ? 1 : -1;
    return [...filteredRows].sort((left, right) => {
      switch (sortState.key) {
        case "year":
          return (
            ((parseMaybeNumber(left.year) ?? Number.NEGATIVE_INFINITY) -
              (parseMaybeNumber(right.year) ?? Number.NEGATIVE_INFINITY)) *
            multiplier
          );
        case "library":
          return (
            String(left.library || "").localeCompare(String(right.library || ""), locale) *
            multiplier
          );
        case "decision":
          return (
            ((DECISION_ORDER[String(left.decision || "UNKNOWN").toUpperCase()] ?? 99) -
              (DECISION_ORDER[String(right.decision || "UNKNOWN").toUpperCase()] ?? 99)) *
            multiplier
          );
        case "imdb":
          return (
            ((parseMaybeNumber(left.imdb_rating) ?? Number.NEGATIVE_INFINITY) -
              (parseMaybeNumber(right.imdb_rating) ?? Number.NEGATIVE_INFINITY)) *
            multiplier
          );
        case "size":
          return (
            ((parseMaybeNumber(left.file_size_gb) ?? Number.NEGATIVE_INFINITY) -
              (parseMaybeNumber(right.file_size_gb) ?? Number.NEGATIVE_INFINITY)) *
            multiplier
          );
        case "title":
        default:
          return (
            String(left.title || "").localeCompare(String(right.title || ""), locale) *
            multiplier
          );
      }
    });
  }, [filteredRows, locale, sortState]);
  const visibleSizeGb = useMemo(
    () => sumReportSizeGb(sortedRows),
    [sortedRows]
  );
  const activeSortLabel = useMemo(() => {
    const keyToLabel: Record<string, string> = {
      title: t("column.title"),
      year: t("column.year"),
      library: t("column.library"),
      decision: t("column.decision"),
      imdb: t("column.imdb"),
      size: t("column.size")
    };
    return keyToLabel[sortState.key] ?? t("column.title");
  }, [sortState.key, t]);

  useEffect(() => {
    if (!sortedRows.length) {
      setSelectedRowKey(null);
      setDetailModalOpen(false);
      return;
    }

    if (!selectedRowKey) {
      setSelectedRowKey(buildReportRowKey(sortedRows[0]));
      return;
    }

    const stillVisible = sortedRows.some((row) => buildReportRowKey(row) === selectedRowKey);
    if (!stillVisible) {
      setSelectedRowKey(buildReportRowKey(sortedRows[0]));
    }
  }, [selectedRowKey, sortedRows]);

  const selectedIndex = useMemo(
    () =>
      Math.max(
        sortedRows.findIndex((row) => buildReportRowKey(row) === selectedRowKey),
        0
      ),
    [selectedRowKey, sortedRows]
  );
  const row = sortedRows[selectedIndex] ?? sortedRows[0] ?? null;
  const hasActiveFilters = Boolean(search.trim() || libraryFilter || decisionFilter);

  function selectRow(current: ReportRow) {
    setSelectedRowKey(buildReportRowKey(current));
  }

  function openDetail(current: ReportRow) {
    selectRow(current);
    setDetailModalOpen(true);
  }

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
          <SectionCard
            title={t("library.filters.title")}
            eyebrow={t("library.filters.eyebrow")}
          >
            <div className="library-toolbar">
              <div className="library-toolbar__search">
                <label className="search-field search-field--toolbar">
                  <Search size={16} />
                  <input
                    onChange={(event) =>
                      startTransition(() => {
                        setSearch(event.target.value);
                      })
                    }
                    placeholder={t("library.search.placeholder")}
                    value={search}
                  />
                </label>
                <div className="scope-switch" role="tablist" aria-label={t("library.search.scope")}>
                  {(["all", "title"] as const).map((scope) => (
                    <button
                      key={scope}
                      className={searchScope === scope ? "is-active" : ""}
                      onClick={() => setSearchScope(scope)}
                      type="button"
                    >
                      {scope === "all"
                        ? t("library.search.scope.all")
                        : t("library.search.scope.title")}
                    </button>
                  ))}
                </div>
              </div>

              <label className="form-field form-field--compact">
                <span>{t("library.filter.library")}</span>
                <select
                  onChange={(event) => setLibraryFilter(event.target.value)}
                  value={libraryFilter}
                >
                  <option value="">{t("library.filter.all_libraries")}</option>
                  {libraryOptions.map((library) => (
                    <option key={library} value={library}>
                      {library}
                    </option>
                  ))}
                </select>
              </label>

              <label className="form-field form-field--compact">
                <span>{t("library.filter.decision")}</span>
                <select
                  onChange={(event) => setDecisionFilter(event.target.value)}
                  value={decisionFilter}
                >
                  <option value="">{t("library.filter.all_decisions")}</option>
                  {REPORT_DECISION_OPTIONS.map((decision) => (
                    <option key={decision} value={decision}>
                      {translateDecision(decision, t)}
                    </option>
                  ))}
                </select>
              </label>

              <div className="library-toolbar__metrics">
                <article className="library-glance">
                  <span>{t("library.filter.results")}</span>
                  <strong>{sortedRows.length.toLocaleString(locale)}</strong>
                </article>
                <article className="library-glance">
                  <span>{t("library.filter.visible_space")}</span>
                  <strong>
                    {visibleSizeGb.toLocaleString(locale, {
                      minimumFractionDigits: 1,
                      maximumFractionDigits: 1
                    })}{" "}
                    {t("unit.gb")}
                  </strong>
                </article>
                <article className="library-glance">
                  <span>{t("library.filter.sorting")}</span>
                  <strong>{activeSortLabel}</strong>
                </article>
                <button
                  className="secondary-button"
                  disabled={!hasActiveFilters}
                  onClick={() => {
                    setSearch("");
                    setLibraryFilter("");
                    setDecisionFilter("");
                    setSearchScope("all");
                  }}
                  type="button"
                >
                  {t("library.filter.clear")}
                </button>
              </div>
            </div>
          </SectionCard>

          <div className="split-layout split-layout--library">
            <SectionCard
              title={t("library.catalog.title", { count: sortedRows.length })}
              eyebrow={t("library.catalog.eyebrow")}
              className="split-layout__main library-table-card"
            >
              <VirtualTable<ReportRow>
                columns={[
                  {
                    key: "title",
                    label: t("column.title"),
                    width: "36%",
                    sortable: true,
                    render: (current) => (
                      <button
                        className="table-title-button"
                        onClick={(event) => {
                          event.stopPropagation();
                          openDetail(current);
                        }}
                        type="button"
                      >
                        <span
                          className={`table-title-button__dot table-title-button__dot--${getDecisionTone(
                            current.decision
                          )}`}
                        />
                        <span className="table-title-button__label">
                          {String(current.title || t("app.empty_dash"))}
                        </span>
                      </button>
                    )
                  },
                  {
                    key: "year",
                    label: t("column.year"),
                    width: "10%",
                    sortable: true,
                    render: (current) => String(current.year || t("app.empty_dash"))
                  },
                  {
                    key: "library",
                    label: t("column.library"),
                    width: "22%",
                    sortable: true,
                    render: (current) => String(current.library || t("app.empty_dash"))
                  },
                  {
                    key: "decision",
                    label: t("column.decision"),
                    width: "14%",
                    sortable: true,
                    render: (current) => (
                      <span
                        className={`decision-chip decision-chip--${getDecisionTone(
                          current.decision
                        )}`}
                      >
                        {translateDecision(String(current.decision || "UNKNOWN"), t)}
                      </span>
                    )
                  },
                  {
                    key: "imdb",
                    label: t("column.imdb"),
                    width: "10%",
                    align: "right",
                    sortable: true,
                    render: (current) => String(current.imdb_rating || t("app.empty_dash"))
                  },
                  {
                    key: "size",
                    label: t("column.size"),
                    width: "8%",
                    align: "right",
                    sortable: true,
                    render: (current) =>
                      current.file_size_gb
                        ? Number(current.file_size_gb).toLocaleString(locale, {
                            minimumFractionDigits: 1,
                            maximumFractionDigits: 1
                          })
                        : t("app.empty_dash")
                  }
                ]}
                fillHeight
                onSelect={(index) => selectRow(sortedRows[index])}
                onSortChange={setSortState}
                rowTone={(current) => getDecisionTone(current.decision)}
                rows={sortedRows}
                selectedIndex={selectedIndex}
                sortState={sortState}
              />
            </SectionCard>

            <div className="split-layout__aside library-detail-panel">
              <MovieDetailPanel row={row} />
            </div>
          </div>

          <MovieDetailModal
            open={detailModalOpen}
            onClose={() => setDetailModalOpen(false)}
            row={row}
          />
        </>
      ) : null}
    </div>
  );
}
