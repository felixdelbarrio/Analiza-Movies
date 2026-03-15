import {
  Suspense,
  lazy,
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useState
} from "react";
import { Search } from "lucide-react";

import { useAppContext } from "../app/use-app-context";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { useReportAll } from "../hooks/use-dashboard-data";
import { translateDecision } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { buildChartOption } from "../lib/charts";
import {
  DASHBOARD_VIEWS,
  REPORT_DECISION_OPTIONS,
  filterReportRows,
  getDashboardViews,
  sumReportSizeGb,
  uniqueValues,
  type ReportSearchScope
} from "../lib/data";
import { useStoredState } from "../lib/preferences";
import type { DashboardViewKey } from "../lib/types";

const ChartCard = lazy(async () => {
  const module = await import("../components/chart-card");
  return { default: module.ChartCard };
});

export function AnalyticsPage() {
  const { locale, t } = useI18n();
  const { activeProfileId, preferences } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [selectedView, setSelectedView] = useStoredState<DashboardViewKey>(
    "am.analytics.view",
    DASHBOARD_VIEWS[0]
  );
  const [search, setSearch] = useState("");
  const [searchScope, setSearchScope] = useState<ReportSearchScope>("all");
  const [libraryFilter, setLibraryFilter] = useState("");
  const [decisionFilter, setDecisionFilter] = useState("");
  const deferredSearch = useDeferredValue(search);

  const viewOptions = useMemo(() => getDashboardViews(t), [t]);
  const libraryOptions = useMemo(
    () => uniqueValues(reportAll, "library", locale),
    [locale, reportAll]
  );
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
  const visibleSizeGb = useMemo(() => sumReportSizeGb(filteredRows), [filteredRows]);
  const selectedViewLabel = useMemo(
    () =>
      viewOptions.find((option) => option.key === selectedView)?.label ??
      viewOptions[0]?.label ??
      "",
    [selectedView, viewOptions]
  );
  const selectedChartOption = useMemo(
    () => buildChartOption(selectedView, filteredRows, { locale, t }),
    [filteredRows, locale, preferences.theme, selectedView, t]
  );
  const hasActiveFilters = Boolean(search.trim() || libraryFilter || decisionFilter);

  useEffect(() => {
    if (!DASHBOARD_VIEWS.includes(selectedView)) {
      setSelectedView(DASHBOARD_VIEWS[0]);
    }
  }, [selectedView, setSelectedView]);

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("nav.analytics")}
        title={t("analytics.hero.title")}
        description={t("analytics.hero.description")}
      />

      {reportAllQuery.isLoading ? (
        <PageState
          eyebrow={t("stage.connecting")}
          title={t("analytics.loading.title")}
          message={t("analytics.loading.message")}
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
          title={t("analytics.error.title")}
          message={t("analytics.error.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && !reportAll.length ? (
        <PageState
          eyebrow={t("analytics.empty.eyebrow")}
          title={t("analytics.empty.title")}
          message={t("analytics.empty.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && reportAll.length ? (
        <>
          <SectionCard
            title={t("analytics.controls.title")}
            eyebrow={t("analytics.controls.eyebrow")}
          >
            <div className="analytics-control-stack">
              <div className="analytics-picker">
                <label className="form-field">
                  <span>{t("analytics.view.label")}</span>
                  <select
                    onChange={(event) =>
                      setSelectedView(event.target.value as DashboardViewKey)
                    }
                    value={selectedView}
                  >
                    {viewOptions.map((option) => (
                      <option key={option.key} value={option.key}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="analytics-picker__hint">{t("analytics.view.hint")}</p>
              </div>

              <div className="analytics-toolbar">
                <div className="analytics-toolbar__search">
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
                  <div
                    className="scope-switch"
                    role="tablist"
                    aria-label={t("library.search.scope")}
                  >
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
              </div>

              <div className="library-toolbar__metrics">
                <article className="library-glance">
                  <span>{t("analytics.view.label")}</span>
                  <strong>{selectedViewLabel}</strong>
                </article>
                <article className="library-glance">
                  <span>{t("library.filter.results")}</span>
                  <strong>{filteredRows.length.toLocaleString(locale)}</strong>
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
                <button
                  className="secondary-button"
                  disabled={!hasActiveFilters}
                  onClick={() => {
                    setSearch("");
                    setSearchScope("all");
                    setLibraryFilter("");
                    setDecisionFilter("");
                  }}
                  type="button"
                >
                  {t("library.filter.clear")}
                </button>
              </div>
            </div>
          </SectionCard>

          {!filteredRows.length ? (
            <PageState
              eyebrow={t("analytics.controls.eyebrow")}
              title={t("analytics.empty.title")}
              message={t("table.empty")}
            />
          ) : (
            <Suspense fallback={<div className="chart-card-skeleton" />}>
              <div className="analytics-chart-stage">
                <ChartCard
                  eyebrow={t("analytics.chart.eyebrow")}
                  height={580}
                  option={selectedChartOption}
                  subtitle={t("analytics.chart.subtitle")}
                  title={selectedViewLabel}
                />
              </div>
            </Suspense>
          )}
        </>
      ) : null}
    </div>
  );
}
