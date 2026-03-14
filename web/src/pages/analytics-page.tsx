import { Suspense, lazy, useMemo } from "react";

import { useAppContext } from "../app/use-app-context";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { useReportAll } from "../hooks/use-dashboard-data";
import { translateDashboardView } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { buildChartOption } from "../lib/charts";
import { DASHBOARD_VIEWS } from "../lib/data";

const ChartCard = lazy(async () => {
  const module = await import("../components/chart-card");
  return { default: module.ChartCard };
});

export function AnalyticsPage() {
  const { locale, t } = useI18n();
  const { activeProfileId, preferences } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const chartViews = useMemo(
    () =>
      DASHBOARD_VIEWS.map((view) => ({
        key: view,
        label: translateDashboardView(view, t),
        option: buildChartOption(view, reportAll, { locale, t })
      })),
    [locale, preferences.theme, reportAll, t]
  );

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
        <div className="chart-grid">
          {chartViews.map((view) => (
            <Suspense key={view.key} fallback={<div className="chart-card-skeleton" />}>
              <ChartCard
                eyebrow={t("analytics.chart.eyebrow")}
                title={view.label}
                option={view.option}
                subtitle={t("analytics.chart.subtitle")}
              />
            </Suspense>
          ))}
        </div>
      ) : null}
    </div>
  );
}
