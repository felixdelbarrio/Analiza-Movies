import { useEffect, useMemo, useState } from "react";

import { useAppContext } from "../app/use-app-context";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { useReportAll } from "../hooks/use-dashboard-data";
import { useI18n } from "../i18n/provider";
import { findDuplicates } from "../lib/data";
import type { ReportRow } from "../lib/types";

export function DuplicatesPage() {
  const { t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const duplicateRows = useMemo(() => findDuplicates(reportAll), [reportAll]);

  useEffect(() => {
    if (selectedIndex >= duplicateRows.length) {
      setSelectedIndex(0);
    }
  }, [duplicateRows.length, selectedIndex]);

  const row = duplicateRows[selectedIndex] ?? duplicateRows[0] ?? null;

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("nav.duplicates")}
        title={t("duplicates.hero.title")}
        description={t("duplicates.hero.description")}
      />

      {reportAllQuery.isLoading ? (
        <PageState
          eyebrow={t("stage.connecting")}
          title={t("duplicates.loading.title")}
          message={t("duplicates.loading.message")}
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
          title={t("duplicates.error.title")}
          message={t("duplicates.error.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && !duplicateRows.length ? (
        <PageState
          eyebrow={t("duplicates.clean.eyebrow")}
          title={t("duplicates.clean.title")}
          message={t("duplicates.clean.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && duplicateRows.length ? (
        <div className="split-layout">
          <SectionCard
            title={t("duplicates.table.title", { count: duplicateRows.length })}
            eyebrow={t("metadata.filters.eyebrow")}
            className="split-layout__main"
          >
            <VirtualTable<ReportRow>
              columns={[
                {
                  key: "title",
                  label: t("column.title"),
                  width: "40%",
                  render: (current) => String(current.title || t("app.empty_dash"))
                },
                {
                  key: "library",
                  label: t("column.library"),
                  width: "26%",
                  render: (current) => String(current.library || t("app.empty_dash"))
                },
                {
                  key: "imdb",
                  label: t("column.imdb_id"),
                  width: "18%",
                  render: (current) => String(current.imdb_id || t("app.empty_dash"))
                },
                {
                  key: "dup",
                  label: t("column.count"),
                  width: "16%",
                  render: (current) => String(current.dup_count || t("app.empty_dash"))
                }
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
      ) : null}
    </div>
  );
}
