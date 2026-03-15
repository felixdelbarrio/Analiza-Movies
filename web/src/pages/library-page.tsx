import {
  Download,
  Image,
  LayoutGrid,
  Rows3,
  Search
} from "lucide-react";
import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent
} from "react";

import { useAppContext } from "../app/use-app-context";
import { MovieDetailModal } from "../components/movie-detail-modal";
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
  getPoster,
  parseMaybeNumber,
  sumReportSizeGb,
  uniqueValues,
  type ReportSearchScope
} from "../lib/data";
import { downloadCsv } from "../lib/export";
import { useStoredState } from "../lib/preferences";
import type { ReportRow } from "../lib/types";

const DECISION_ORDER: Record<string, number> = {
  KEEP: 0,
  MAYBE: 1,
  DELETE: 2,
  UNKNOWN: 3
};

type LibraryViewMode = "table" | "cards" | "poster";

function formatSizeValue(
  value: unknown,
  locale: string,
  unitLabel: string,
  fallback: string
) {
  const size = parseMaybeNumber(value);
  if (size === null) {
    return fallback;
  }
  return `${size.toLocaleString(locale, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1
  })} ${unitLabel}`;
}

function formatMetricValue(
  value: unknown,
  fallback: string,
  options?: { suffix?: string; round?: boolean }
) {
  const numeric = parseMaybeNumber(value);
  if (numeric === null) {
    return fallback;
  }
  if (options?.round) {
    return `${Math.round(numeric)}${options.suffix ?? ""}`;
  }
  if (options?.suffix) {
    return `${numeric}${options.suffix}`;
  }
  return String(numeric);
}

function handleActivate(event: KeyboardEvent<HTMLElement>, onActivate: () => void) {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  event.preventDefault();
  onActivate();
}

interface MetricChipProps {
  label: string;
  value: string;
}

function MetricChip({ label, value }: MetricChipProps) {
  return (
    <div className="library-metric-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

interface PosterGalleryProps {
  locale: string;
  rows: ReportRow[];
  onOpenDetail: (row: ReportRow) => void;
  t: ReturnType<typeof useI18n>["t"];
}

function PosterGallery({ locale, rows, onOpenDetail, t }: PosterGalleryProps) {
  if (!rows.length) {
    return <div className="virtual-table__empty">{t("table.empty")}</div>;
  }

  return (
    <div className="library-poster-grid">
      {rows.map((row) => {
        const tone = getDecisionTone(row.decision);
        const poster = getPoster(row);
        const subtitle = row.actors || row.director || row.genre || t("app.empty_dash");
        return (
          <article
            key={buildReportRowKey(row)}
            className="library-poster-card"
            data-tone={tone}
            onClick={() => onOpenDetail(row)}
            onKeyDown={(event) => handleActivate(event, () => onOpenDetail(row))}
            role="button"
            tabIndex={0}
          >
            <div className="library-poster-card__poster">
              {poster ? (
                <img alt={String(row.title || t("detail.poster_alt"))} src={poster} />
              ) : (
                <span>{t("detail.poster_missing")}</span>
              )}
            </div>

            <div className="library-poster-card__body">
              <div className="library-poster-card__header">
                <div className="library-poster-card__heading">
                  <span className="library-poster-card__eyebrow">
                    {row.library || t("detail.fallback.library")}
                  </span>
                  <h3>{row.title || t("app.empty_dash")}</h3>
                  <p>{[row.year, row.imdb_id].filter(Boolean).join(" · ") || t("app.empty_dash")}</p>
                </div>
                <span className={`decision-chip decision-chip--${tone}`}>
                  {translateDecision(String(row.decision || "UNKNOWN"), t)}
                </span>
              </div>

              <div className="library-poster-card__metrics">
                <MetricChip
                  label={t("column.imdb")}
                  value={formatMetricValue(row.imdb_rating, t("app.empty_dash"))}
                />
                <MetricChip
                  label="RT"
                  value={formatMetricValue(row.rt_score, t("app.empty_dash"), { suffix: "%" })}
                />
                <MetricChip
                  label="MC"
                  value={formatMetricValue(row.metacritic_score, t("app.empty_dash"), { round: true })}
                />
                <MetricChip
                  label={t("detail.metric.size")}
                  value={formatSizeValue(row.file_size_gb, locale, t("unit.gb"), t("app.empty_dash"))}
                />
              </div>

              <p className="library-poster-card__summary">{subtitle}</p>
            </div>
          </article>
        );
      })}
    </div>
  );
}

interface SignalCardGridProps {
  locale: string;
  rows: ReportRow[];
  onOpenDetail: (row: ReportRow) => void;
  t: ReturnType<typeof useI18n>["t"];
}

function SignalCardGrid({ locale, rows, onOpenDetail, t }: SignalCardGridProps) {
  if (!rows.length) {
    return <div className="virtual-table__empty">{t("table.empty")}</div>;
  }

  return (
    <div className="library-signal-grid">
      {rows.map((row) => {
        const tone = getDecisionTone(row.decision);
        return (
          <article
            key={buildReportRowKey(row)}
            className="library-signal-card"
            data-tone={tone}
            onClick={() => onOpenDetail(row)}
            onKeyDown={(event) => handleActivate(event, () => onOpenDetail(row))}
            role="button"
            tabIndex={0}
          >
            <div className="library-signal-card__header">
              <div>
                <span className="library-signal-card__eyebrow">
                  {row.library || t("detail.fallback.library")}
                </span>
                <h3>{row.title || t("app.empty_dash")}</h3>
                <p>{[row.year, row.imdb_id].filter(Boolean).join(" · ") || t("app.empty_dash")}</p>
              </div>
              <span className={`decision-chip decision-chip--${tone}`}>
                {translateDecision(String(row.decision || "UNKNOWN"), t)}
              </span>
            </div>

            <div className="library-signal-card__metrics">
              <MetricChip
                label={t("column.imdb")}
                value={formatMetricValue(row.imdb_rating, t("app.empty_dash"))}
              />
              <MetricChip
                label={t("detail.metric.size")}
                value={formatSizeValue(row.file_size_gb, locale, t("unit.gb"), t("app.empty_dash"))}
              />
              <MetricChip
                label={t("detail.director")}
                value={String(row.director || t("app.empty_dash"))}
              />
            </div>

            <div className="library-signal-card__footer">
              <span>{t("detail.actors")}</span>
              <strong>{String(row.actors || row.genre || t("app.empty_dash"))}</strong>
            </div>
          </article>
        );
      })}
    </div>
  );
}

export function LibraryPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [search, setSearch] = useState("");
  const [searchScope, setSearchScope] = useState<ReportSearchScope>("title");
  const [libraryFilter, setLibraryFilter] = useState("");
  const [decisionFilter, setDecisionFilter] = useState("");
  const [selectedRowKey, setSelectedRowKey] = useState<string | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [viewMode, setViewMode] = useStoredState<LibraryViewMode>("am.library.view", "table");
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
  const fullSearchRows = useMemo(
    () =>
      deferredSearch.trim()
        ? filterReportRows(reportAll, {
            decision: decisionFilter,
            library: libraryFilter,
            search: deferredSearch,
            searchScope: "all"
          })
        : filteredRows,
    [decisionFilter, deferredSearch, filteredRows, libraryFilter, reportAll]
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
  const visibleSizeGb = useMemo(() => sumReportSizeGb(sortedRows), [sortedRows]);
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
  const searchAssistVisible =
    searchScope === "title" &&
    deferredSearch.trim().length > 0 &&
    fullSearchRows.length > filteredRows.length;

  const exportAllColumns = useMemo(
    () => [
      { header: t("column.title"), value: (row: ReportRow) => row.title },
      { header: t("column.year"), value: (row: ReportRow) => row.year },
      { header: t("column.library"), value: (row: ReportRow) => row.library },
      {
        header: t("column.decision"),
        value: (row: ReportRow) => translateDecision(String(row.decision || "UNKNOWN"), t)
      },
      { header: t("column.imdb_id"), value: (row: ReportRow) => row.imdb_id },
      { header: t("column.imdb"), value: (row: ReportRow) => row.imdb_rating },
      { header: "RT", value: (row: ReportRow) => row.rt_score },
      { header: "Metacritic", value: (row: ReportRow) => row.metacritic_score },
      { header: t("column.size"), value: (row: ReportRow) => row.file_size_gb },
      { header: t("detail.genre"), value: (row: ReportRow) => row.genre },
      { header: t("detail.director"), value: (row: ReportRow) => row.director },
      { header: t("detail.actors"), value: (row: ReportRow) => row.actors },
      { header: t("detail.plot"), value: (row: ReportRow) => row.plot },
      { header: t("column.route"), value: (row: ReportRow) => row.file }
    ],
    [t]
  );

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
  const hasActiveFilters = Boolean(
    search.trim() || libraryFilter || decisionFilter || searchScope !== "title"
  );

  function selectRow(current: ReportRow) {
    setSelectedRowKey(buildReportRowKey(current));
  }

  function openDetail(current: ReportRow) {
    selectRow(current);
    setDetailModalOpen(true);
  }

  function exportRows(rows: ReportRow[], scope: "visible" | "all") {
    downloadCsv(
      `catalog-${scope}-${new Date().toISOString().slice(0, 10)}.csv`,
      exportAllColumns,
      rows
    );
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
          <SectionCard title={t("library.filters.title")} eyebrow={t("library.filters.eyebrow")}>
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
                  {(["title", "all"] as const).map((scope) => (
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

                {searchAssistVisible ? (
                  <div className="library-search-assist">
                    <p>
                      {t("library.search.assist", {
                        visible: filteredRows.length,
                        total: fullSearchRows.length
                      })}
                    </p>
                    <button
                      className="inline-link"
                      onClick={() => setSearchScope("all")}
                      type="button"
                    >
                      {t("library.search.try_all")}
                    </button>
                  </div>
                ) : null}
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
                    setSearchScope("title");
                  }}
                  type="button"
                >
                  {t("library.filter.clear")}
                </button>
              </div>
            </div>
          </SectionCard>

          <SectionCard
            title={t("library.catalog.title", { count: sortedRows.length })}
            eyebrow={t("library.catalog.eyebrow")}
            className="library-catalog-card"
            actions={
              <div className="inline-actions">
                <div
                  className="scope-switch scope-switch--compact"
                  role="tablist"
                  aria-label={t("library.view.label")}
                >
                  <button
                    className={viewMode === "table" ? "is-active" : ""}
                    onClick={() => setViewMode("table")}
                    type="button"
                  >
                    <Rows3 size={16} />
                    {t("library.view.table")}
                  </button>
                  <button
                    className={viewMode === "cards" ? "is-active" : ""}
                    onClick={() => setViewMode("cards")}
                    type="button"
                  >
                    <LayoutGrid size={16} />
                    {t("library.view.cards")}
                  </button>
                  <button
                    className={viewMode === "poster" ? "is-active" : ""}
                    onClick={() => setViewMode("poster")}
                    type="button"
                  >
                    <Image size={16} />
                    {t("library.view.poster")}
                  </button>
                </div>

                {hasActiveFilters ? (
                  <button
                    className="secondary-button"
                    onClick={() => exportRows(reportAll, "all")}
                    type="button"
                  >
                    <Download size={16} />
                    {t("app.action.export_all_csv")}
                  </button>
                ) : null}

                <button
                  className="secondary-button"
                  onClick={() => exportRows(sortedRows, "visible")}
                  type="button"
                >
                  <Download size={16} />
                  {hasActiveFilters
                    ? t("app.action.export_visible_csv")
                    : t("app.action.export_csv")}
                </button>
              </div>
            }
          >
            {viewMode === "table" ? (
              <VirtualTable<ReportRow>
                columns={[
                  {
                    key: "title",
                    label: t("column.title"),
                    width: "28%",
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
                    width: "8%",
                    align: "center",
                    sortable: true,
                    render: (current) => String(current.year || t("app.empty_dash"))
                  },
                  {
                    key: "library",
                    label: t("column.library"),
                    width: "16%",
                    sortable: true,
                    render: (current) => String(current.library || t("app.empty_dash"))
                  },
                  {
                    key: "decision",
                    label: t("column.decision"),
                    width: "14%",
                    align: "center",
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
                    width: "8%",
                    align: "center",
                    sortable: true,
                    render: (current) => String(current.imdb_rating || t("app.empty_dash"))
                  },
                  {
                    key: "size",
                    label: t("column.size"),
                    width: "10%",
                    align: "center",
                    sortable: true,
                    render: (current) =>
                      current.file_size_gb
                        ? Number(current.file_size_gb).toLocaleString(locale, {
                            minimumFractionDigits: 1,
                            maximumFractionDigits: 1
                          })
                        : t("app.empty_dash")
                  },
                  {
                    key: "director",
                    label: t("detail.director"),
                    width: "16%",
                    render: (current) => String(current.director || t("app.empty_dash"))
                  }
                ]}
                maxHeight={780}
                onSelect={(index) => selectRow(sortedRows[index])}
                onSortChange={setSortState}
                rowTone={(current) => getDecisionTone(current.decision)}
                rows={sortedRows}
                selectedIndex={selectedIndex}
                sortState={sortState}
                variant="sheet"
              />
            ) : null}

            {viewMode === "cards" ? (
              <SignalCardGrid locale={locale} onOpenDetail={openDetail} rows={sortedRows} t={t} />
            ) : null}

            {viewMode === "poster" ? (
              <PosterGallery locale={locale} onOpenDetail={openDetail} rows={sortedRows} t={t} />
            ) : null}
          </SectionCard>

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
