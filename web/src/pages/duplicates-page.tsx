import { Search } from "lucide-react";
import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";

import { useAppContext } from "../app/use-app-context";
import { DuplicateDetailPanel } from "../components/duplicate-detail-panel";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { useReportAll } from "../hooks/use-dashboard-data";
import { useI18n } from "../i18n/provider";
import { buildDuplicateGroups } from "../lib/data";
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

export function DuplicatesPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const reportAllQuery = useReportAll(activeProfileId);
  const reportAll = reportAllQuery.data ?? [];
  const [search, setSearch] = useState("");
  const [selectedImdbId, setSelectedImdbId] = useState<string | null>(null);
  const deferredSearch = useDeferredValue(search);
  const duplicateGroups = useMemo(() => buildDuplicateGroups(reportAll), [reportAll]);
  const filteredGroups = useMemo(() => {
    const term = deferredSearch.trim().toLowerCase();
    if (!term) {
      return duplicateGroups;
    }
    return duplicateGroups.filter((group) =>
      [
        group.imdbId,
        group.imdbTitle,
        ...group.copies.flatMap((copy) => [copy.file, copy.library, copy.title])
      ]
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .join(" ")
        .toLowerCase()
        .includes(term)
    );
  }, [deferredSearch, duplicateGroups]);
  const selectedIndex = useMemo(() => {
    if (!filteredGroups.length) {
      return 0;
    }
    if (!selectedImdbId) {
      return 0;
    }
    const index = filteredGroups.findIndex((group) => group.imdbId === selectedImdbId);
    return index >= 0 ? index : 0;
  }, [filteredGroups, selectedImdbId]);

  useEffect(() => {
    if (!filteredGroups.length) {
      if (selectedImdbId !== null) {
        setSelectedImdbId(null);
      }
      return;
    }
    if (!selectedImdbId || !filteredGroups.some((group) => group.imdbId === selectedImdbId)) {
      setSelectedImdbId(filteredGroups[0].imdbId);
    }
  }, [filteredGroups, selectedImdbId]);

  const selectedGroup = filteredGroups[selectedIndex] ?? filteredGroups[0] ?? null;
  const duplicateCopies = filteredGroups.reduce((total, group) => total + group.duplicateCount, 0);
  const criticalGroups = filteredGroups.filter((group) => group.tone === "delete").length;

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
              <strong>{filteredGroups.length.toLocaleString(locale)}</strong>
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
              title={t("duplicates.table.title", { count: filteredGroups.length })}
              eyebrow={t("duplicates.table.eyebrow")}
              className="split-layout__main library-table-card"
              actions={
                <label className="search-field search-field--toolbar duplicate-search">
                  <Search size={16} />
                  <input
                    onChange={(event) =>
                      startTransition(() => {
                        setSearch(event.target.value);
                      })
                    }
                    placeholder={t("duplicates.search.placeholder")}
                    value={search}
                  />
                </label>
              }
            >
              <VirtualTable<DuplicateGroup>
                columns={[
                  {
                    key: "imdbId",
                    label: t("column.imdb_id"),
                    width: "22%",
                    render: (group) => (
                      <span className="duplicate-imdb-id">{group.imdbId}</span>
                    )
                  },
                  {
                    key: "imdbTitle",
                    label: t("column.title"),
                    width: "56%",
                    render: (group) => (
                      <div className="duplicate-title-cell">
                        <strong>{group.imdbTitle}</strong>
                        <small>{decisionBreakdown(group, t)}</small>
                      </div>
                    )
                  },
                  {
                    key: "duplicateCount",
                    label: t("column.count"),
                    width: "22%",
                    align: "right",
                    render: (group) => (
                      <div className="duplicate-count-cell">
                        <strong>{group.duplicateCount}</strong>
                        <small>
                          {group.libraryCount} {t("unit.libraries")}
                        </small>
                      </div>
                    )
                  }
                ]}
                fillHeight
                onSelect={(index) => setSelectedImdbId(filteredGroups[index]?.imdbId ?? null)}
                rowTone={(group) => group.tone}
                rows={filteredGroups}
                selectedIndex={selectedIndex}
              />
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
