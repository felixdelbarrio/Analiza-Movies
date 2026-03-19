import { Suspense, lazy, useMemo } from "react";
import { useNavigate } from "react-router-dom";

import { useAppContext } from "../app/use-app-context";
import { KpiCard } from "../components/kpi-card";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { useReportAll } from "../hooks/use-dashboard-data";
import { translateDashboardView } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { buildChartOption } from "../lib/charts";
import {
  computeSummary,
  formatCountSize,
  normalizeDashboardViews
} from "../lib/data";

const ChartCard = lazy(async () => {
  const module = await import("../components/chart-card");
  return { default: module.ChartCard };
});

export function DashboardPage() {
  const { locale, t } = useI18n();
  const { activeProfileId, preferences } = useAppContext();
  const navigate = useNavigate();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const summary = useMemo(() => computeSummary(reportAll), [reportAll]);
  const selectedViews = useMemo(
    () => normalizeDashboardViews(preferences.dashboardViews),
    [preferences.dashboardViews]
  );
  const chartViews = useMemo(
    () =>
      selectedViews.map((viewKey) => ({
        key: viewKey,
        label: translateDashboardView(viewKey, t),
        option: buildChartOption(viewKey, reportAll, { locale, t })
      })),
    [locale, preferences.theme, reportAll, selectedViews, t]
  );
  const titleUnitLabel = useMemo(
    () => t("chart.metric.titles").toLocaleLowerCase(locale),
    [locale, t]
  );

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("dashboard.hero.eyebrow")}
        title={t("dashboard.hero.title")}
        description={t("dashboard.hero.description")}
      />

      {!activeProfileId ? (
        <PageState
          action={
            <button
              className="primary-button"
              onClick={() => navigate("/settings")}
              type="button"
            >
              {t("dashboard.setup.action")}
            </button>
          }
          eyebrow={t("dashboard.setup.eyebrow")}
          title={t("dashboard.setup.title")}
          message={t("dashboard.setup.message")}
        />
      ) : null}

      {activeProfileId && reportAllQuery.isLoading ? (
        <PageState
          eyebrow={t("stage.connecting")}
          title={t("dashboard.loading.title")}
          message={t("dashboard.loading.message")}
        />
      ) : null}

      {activeProfileId && reportAllQuery.error ? (
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
          title={t("dashboard.error.title")}
          message={t("dashboard.error.message")}
        />
      ) : null}

      {activeProfileId && !reportAllQuery.isLoading && !reportAllQuery.error && !reportAll.length ? (
        <PageState
          eyebrow={t("dashboard.empty.eyebrow")}
          title={t("dashboard.empty.title")}
          message={t("dashboard.empty.message")}
        />
      ) : null}

      {activeProfileId && !reportAllQuery.isLoading && !reportAllQuery.error && reportAll.length ? (
        <>
          <section className="kpi-grid">
            <KpiCard
              label={t("dashboard.kpi.catalog")}
              value={`${summary.totalSizeGb.toLocaleString(locale, {
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
              })} GB`}
              detail={`${summary.totalCount.toLocaleString(locale)} ${titleUnitLabel}`}
            />
            <KpiCard
              label={t("decision.keep")}
              tone="keep"
              value={`${summary.keepSizeGb.toLocaleString(locale, {
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
              })} GB`}
              detail={`${summary.keepCount.toLocaleString(locale)} ${titleUnitLabel}`}
            />
            <KpiCard
              label={t("decision.delete")}
              tone="delete"
              value={`${summary.deleteSizeGb.toLocaleString(locale, {
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
              })} GB`}
              detail={`${summary.deleteCount.toLocaleString(locale)} ${titleUnitLabel}`}
            />
            <KpiCard
              label={t("decision.maybe")}
              tone="maybe"
              value={`${summary.maybeSizeGb.toLocaleString(locale, {
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
              })} GB`}
              detail={`${summary.maybeCount.toLocaleString(locale)} ${titleUnitLabel}`}
            />
            <KpiCard
              label={t("dashboard.kpi.imdb_mean")}
              value={
                summary.imdbMean !== null
                  ? summary.imdbMean.toLocaleString(locale, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2
                    })
                  : t("app.na")
              }
            />
          </section>

          <SectionCard title={t("dashboard.exec.title")} eyebrow={t("dashboard.exec.eyebrow")}>
            <div className="insight-grid">
              <article>
                <span>{t("dashboard.exec.review_pending")}</span>
                <strong>{formatCountSize(summary.reviewCount, summary.reviewSizeGb, locale, t)}</strong>
                <p>{t("dashboard.exec.review_pending_copy")}</p>
              </article>
              <article>
                <span>{t("dashboard.exec.dominant_profile")}</span>
                <strong>
                  {summary.keepCount > summary.reviewCount
                    ? t("dashboard.exec.catalog_healthy")
                    : t("dashboard.exec.catalog_pressured")}
                </strong>
                <p>{t("dashboard.exec.dominant_profile_copy")}</p>
              </article>
              <article>
                <span>{t("dashboard.exec.critical_signal")}</span>
                <strong>
                  {summary.deleteSizeGb > summary.keepSizeGb * 0.2
                    ? t("dashboard.exec.high_opportunity")
                    : t("dashboard.exec.contained_risk")}
                </strong>
                <p>{t("dashboard.exec.critical_signal_copy")}</p>
              </article>
            </div>
          </SectionCard>

          <div className="chart-grid chart-grid--hero">
            {chartViews.map((view) => (
              <Suspense key={view.key} fallback={<div className="chart-card-skeleton" />}>
                <ChartCard eyebrow={t("dashboard.chart.eyebrow")} title={view.label} option={view.option} />
              </Suspense>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
