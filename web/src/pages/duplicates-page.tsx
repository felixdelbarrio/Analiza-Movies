import { Download } from "lucide-react";
import { useEffect, useMemo, useState, type KeyboardEvent } from "react";

import { useAppContext } from "../app/use-app-context";
import { DuplicateDetailPanel } from "../components/duplicate-detail-panel";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { useReportAll } from "../hooks/use-dashboard-data";
import { useI18n } from "../i18n/provider";
import { buildDuplicateGroups } from "../lib/data";
import { downloadCsv } from "../lib/export";
import type { DuplicateGroup } from "../lib/types";

function decisionBreakdown(group: DuplicateGroup, t: ReturnType<typeof useI18n>["t"]) {
  const parts = [];
  if (group.decisionCounts.DELETE > 0) {
    parts.push(`${group.decisionCounts.DELETE} ${t("decision.delete").toLowerCase()}`);
  }
  if (group.decisionCounts.MAYBE > 0) {
    parts.push(`${group.decisionCounts.MAYBE} ${t("decision.maybe").toLowerCase()}`);
  }
  if (group.decisionCounts.KEEP > 0) {
    parts.push(`${group.decisionCounts.KEEP} ${t("decision.keep").toLowerCase()}`);
  }
  return parts.join(" · ");
}

function handleRowKeyDown(
  event: KeyboardEvent<HTMLElement>,
  onActivate: () => void
) {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  event.preventDefault();
  onActivate();
}

function groupLibraries(group: DuplicateGroup, locale: string) {
  return Array.from(
    new Set(
      group.copies
        .map((copy) => String(copy.library || "").trim())
        .filter(Boolean)
    )
  ).sort((left, right) => left.localeCompare(right, locale));
}

export function DuplicatesPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [selectedImdbId, setSelectedImdbId] = useState<string | null>(null);
  const duplicateGroups = useMemo(() => buildDuplicateGroups(reportAll), [reportAll]);
  const selectedIndex = useMemo(() => {
    if (!duplicateGroups.length) {
      return 0;
    }
    if (!selectedImdbId) {
      return 0;
    }
    const index = duplicateGroups.findIndex((group) => group.imdbId === selectedImdbId);
    return index >= 0 ? index : 0;
  }, [duplicateGroups, selectedImdbId]);

  useEffect(() => {
    if (!duplicateGroups.length) {
      if (selectedImdbId !== null) {
        setSelectedImdbId(null);
      }
      return;
    }
    if (!selectedImdbId || !duplicateGroups.some((group) => group.imdbId === selectedImdbId)) {
      setSelectedImdbId(duplicateGroups[0].imdbId);
    }
  }, [duplicateGroups, selectedImdbId]);

  const selectedGroup = duplicateGroups[selectedIndex] ?? duplicateGroups[0] ?? null;
  const duplicateCopies = duplicateGroups.reduce((total, group) => total + group.duplicateCount, 0);
  const criticalGroups = duplicateGroups.filter((group) => group.tone === "delete").length;

  function exportDuplicates() {
    const rows = duplicateGroups.flatMap((group) =>
      group.copies.map((copy) => ({
        group,
        copy
      }))
    );
    downloadCsv(
      `duplicates-${new Date().toISOString().slice(0, 10)}.csv`,
      [
        {
          header: t("column.imdb_id"),
          value: ({ group }: { group: DuplicateGroup }) => group.imdbId
        },
        {
          header: t("column.title"),
          value: ({ group }: { group: DuplicateGroup }) => group.imdbTitle
        },
        {
          header: t("column.count"),
          value: ({ group }: { group: DuplicateGroup }) => group.duplicateCount
        },
        {
          header: t("library.filter.library"),
          value: ({ copy }: { copy: DuplicateGroup["copies"][number] }) => copy.library
        },
        {
          header: t("column.decision"),
          value: ({ copy }: { copy: DuplicateGroup["copies"][number] }) => copy.decision
        },
        {
          header: t("column.size"),
          value: ({ copy }: { copy: DuplicateGroup["copies"][number] }) => copy.file_size_gb
        },
        {
          header: t("column.route"),
          value: ({ copy }: { copy: DuplicateGroup["copies"][number] }) => copy.file
        }
      ],
      rows
    );
  }

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

      {!reportAllQuery.isLoading && !reportAllQuery.error && !duplicateGroups.length ? (
        <PageState
          eyebrow={t("duplicates.clean.eyebrow")}
          title={t("duplicates.clean.title")}
          message={t("duplicates.clean.message")}
        />
      ) : null}

      {!reportAllQuery.isLoading && !reportAllQuery.error && duplicateGroups.length ? (
        <>
          <div className="duplicate-glance-grid">
            <article className="library-glance">
              <span>{t("duplicates.summary.groups")}</span>
              <strong>{duplicateGroups.length.toLocaleString(locale)}</strong>
            </article>
            <article className="library-glance">
              <span>{t("duplicates.summary.routes")}</span>
              <strong>{duplicateCopies.toLocaleString(locale)}</strong>
            </article>
            <article className="library-glance">
              <span>{t("duplicates.summary.critical")}</span>
              <strong>{criticalGroups.toLocaleString(locale)}</strong>
            </article>
          </div>

          <div className="split-layout split-layout--library">
            <SectionCard
              title={t("duplicates.table.title", { count: duplicateGroups.length })}
              eyebrow={t("duplicates.table.eyebrow")}
              className="split-layout__main duplicate-list-card"
              actions={
                <button className="secondary-button" onClick={exportDuplicates} type="button">
                  <Download size={16} />
                  {t("app.action.export_csv")}
                </button>
              }
            >
              <div className="duplicate-list-stage">
                {duplicateGroups.map((group) => {
                  const selected = group.imdbId === selectedGroup?.imdbId;
                  const libraries = groupLibraries(group, locale);
                  return (
                    <article
                      key={group.imdbId}
                      className={`duplicate-group-card${selected ? " is-selected" : ""}`}
                      data-tone={group.tone}
                      onClick={() => setSelectedImdbId(group.imdbId)}
                      onKeyDown={(event) =>
                        handleRowKeyDown(event, () => setSelectedImdbId(group.imdbId))
                      }
                      role="button"
                      tabIndex={0}
                    >
                      <div className="duplicate-group-card__id">{group.imdbId}</div>

                      <div className="duplicate-group-card__title">
                        <strong>{group.imdbTitle}</strong>
                        <small>
                          {[group.year, decisionBreakdown(group, t)].filter(Boolean).join(" · ")}
                        </small>
                      </div>

                      <div className="duplicate-group-card__libraries">
                        <span>{libraries.join(" · ") || t("app.empty_dash")}</span>
                        <strong>
                          {group.libraryCount} {t("unit.libraries")}
                        </strong>
                      </div>

                      <div className="duplicate-group-card__metrics">
                        <div>
                          <span>{t("column.count")}</span>
                          <strong>{group.duplicateCount}</strong>
                        </div>
                        <div>
                          <span>{t("column.decision")}</span>
                          <strong>{decisionBreakdown(group, t)}</strong>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            </SectionCard>

            <div className="split-layout__aside library-detail-panel">
              <DuplicateDetailPanel group={selectedGroup} />
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
