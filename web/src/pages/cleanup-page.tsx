import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { useAppContext } from "../app/use-app-context";
import { MovieDetailPanel } from "../components/movie-detail-panel";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { useReportFiltered } from "../hooks/use-dashboard-data";
import { translateDecision } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { runDeleteAction } from "../lib/api";
import type { ReportRow } from "../lib/types";

export function CleanupPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportFilteredQuery = useReportFiltered(activeProfileId);
  const reportFiltered = reportFilteredQuery.data ?? [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [dryRun, setDryRun] = useState(true);
  const selectedRow = reportFiltered[selectedIndex] ?? reportFiltered[0] ?? null;

  useEffect(() => {
    if (selectedIndex >= reportFiltered.length) {
      setSelectedIndex(0);
    }
  }, [reportFiltered.length, selectedIndex]);

  const deleteMutation = useMutation({
    mutationFn: async () => {
      const rows = selectedRow ? [selectedRow] : [];
      return runDeleteAction(rows, dryRun);
    }
  });

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("nav.cleanup")}
        title={t("cleanup.hero.title")}
        description={t("cleanup.hero.description")}
        actions={
          <label className="toggle-pill">
            <input
              checked={dryRun}
              onChange={(event) => setDryRun(event.target.checked)}
              type="checkbox"
            />
            <span>{dryRun ? t("cleanup.mode.dry") : t("cleanup.mode.real")}</span>
          </label>
        }
      />

      {reportFilteredQuery.isLoading ? (
        <PageState
          eyebrow={t("stage.connecting")}
          title={t("cleanup.loading.title")}
          message={t("cleanup.loading.message")}
        />
      ) : null}

      {reportFilteredQuery.error ? (
        <PageState
          action={
            <button
              className="primary-button"
              onClick={() => reportFilteredQuery.refetch()}
              type="button"
            >
              {t("app.action.retry")}
            </button>
          }
          eyebrow={t("stage.failed")}
          title={t("cleanup.error.title")}
          message={t("cleanup.error.message")}
        />
      ) : null}

      {!reportFilteredQuery.isLoading &&
      !reportFilteredQuery.error &&
      !reportFiltered.length ? (
        <PageState
          eyebrow={t("cleanup.empty.eyebrow")}
          title={t("cleanup.empty.title")}
          message={t("cleanup.empty.message")}
        />
      ) : null}

      {!reportFilteredQuery.isLoading &&
      !reportFilteredQuery.error &&
      reportFiltered.length ? (
        <div className="split-layout">
          <SectionCard
            title={t("cleanup.queue.title", { count: reportFiltered.length })}
            eyebrow={t("cleanup.queue.eyebrow")}
            className="split-layout__main"
            actions={
              <button
                className="primary-button"
                disabled={!selectedRow || deleteMutation.isPending}
                onClick={() => deleteMutation.mutate()}
                type="button"
              >
                {dryRun ? t("cleanup.action.simulate") : t("cleanup.action.delete")}
              </button>
            }
          >
            <VirtualTable<ReportRow>
              columns={[
                {
                  key: "title",
                  label: t("column.title"),
                  width: "36%",
                  render: (row) => String(row.title || t("app.empty_dash"))
                },
                {
                  key: "library",
                  label: t("column.library"),
                  width: "24%",
                  render: (row) => String(row.library || t("app.empty_dash"))
                },
                {
                  key: "decision",
                  label: t("column.decision"),
                  width: "14%",
                  render: (row) => translateDecision(row.decision, t)
                },
                {
                  key: "imdb",
                  label: t("column.imdb"),
                  width: "10%",
                  render: (row) => String(row.imdb_rating || t("app.empty_dash"))
                },
                {
                  key: "size",
                  label: t("column.size"),
                  width: "8%",
                  render: (row) =>
                    row.file_size_gb
                      ? Number(row.file_size_gb).toLocaleString(locale, {
                          minimumFractionDigits: 1,
                          maximumFractionDigits: 1
                        })
                      : t("app.empty_dash")
                },
                {
                  key: "file",
                  label: t("column.route"),
                  width: "28%",
                  render: (row) => String(row.file || t("app.empty_dash"))
                }
              ]}
              onSelect={setSelectedIndex}
              rows={reportFiltered}
              selectedIndex={selectedIndex}
            />
          </SectionCard>

          <div className="split-layout__aside">
            <MovieDetailPanel row={selectedRow} />
            <SectionCard title={t("cleanup.result.title")} eyebrow={t("cleanup.result.eyebrow")}>
              {deleteMutation.data ? (
                <div className="log-stack">
                  <p>{t("cleanup.result.ok_err", { ok: deleteMutation.data.ok, err: deleteMutation.data.err })}</p>
                  <pre>{deleteMutation.data.logs.join("\n")}</pre>
                </div>
              ) : (
                <p>{t("cleanup.result.placeholder")}</p>
              )}
            </SectionCard>
          </div>
        </div>
      ) : null}
    </div>
  );
}
