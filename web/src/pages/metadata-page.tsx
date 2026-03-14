import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";

import { useAppContext } from "../app/use-app-context";
import { PageHero } from "../components/page-hero";
import { PageState } from "../components/page-state";
import { SectionCard } from "../components/section-card";
import { VirtualTable } from "../components/virtual-table";
import { useMetadata } from "../hooks/use-dashboard-data";
import { translateMetadataAction, translateMetadataField } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { filterMetadata } from "../lib/data";
import type { MetadataRow } from "../lib/types";

export function MetadataPage() {
  const { locale, t } = useI18n();
  const { activeProfileId } = useAppContext();
  const metadataQuery = useMetadata(activeProfileId);
  const metadataRows = metadataQuery.data ?? [];
  const [search, setSearch] = useState("");
  const [libraries, setLibraries] = useState<string[]>([]);
  const [actions, setActions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const deferredSearch = useDeferredValue(search);

  const libraryValues = useMemo(
    () =>
      Array.from(
        new Set(
          metadataRows
            .map((row) => row.library)
            .filter(
              (value): value is string =>
                typeof value === "string" && value.trim().length > 0
            )
        )
      ).sort((left, right) => left.localeCompare(right, locale)),
    [locale, metadataRows]
  );
  const actionValues = useMemo(
    () =>
      Array.from(
        new Set(
          metadataRows
            .map((row) => row.action)
            .filter(
              (value): value is string =>
                typeof value === "string" && value.trim().length > 0
            )
        )
      ).sort((left, right) => left.localeCompare(right, locale)),
    [locale, metadataRows]
  );
  const filtered = useMemo(
    () =>
      filterMetadata(metadataRows, {
        library: libraries,
        action: actions,
        search: deferredSearch
      }),
    [actions, deferredSearch, libraries, metadataRows]
  );

  useEffect(() => {
    if (selectedIndex >= filtered.length) {
      setSelectedIndex(0);
    }
  }, [filtered.length, selectedIndex]);

  const selected = filtered[selectedIndex] ?? filtered[0] ?? null;

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("nav.metadata")}
        title={t("metadata.hero.title")}
        description={t("metadata.hero.description")}
      />

      {metadataQuery.isLoading ? (
        <PageState
          eyebrow={t("stage.connecting")}
          title={t("metadata.loading.title")}
          message={t("metadata.loading.message")}
        />
      ) : null}

      {metadataQuery.error ? (
        <PageState
          action={
            <button className="primary-button" onClick={() => metadataQuery.refetch()} type="button">
              {t("app.action.retry")}
            </button>
          }
          eyebrow={t("stage.failed")}
          title={t("metadata.error.title")}
          message={t("metadata.error.message")}
        />
      ) : null}

      {!metadataQuery.isLoading && !metadataQuery.error && !metadataRows.length ? (
        <PageState
          eyebrow={t("metadata.empty.eyebrow")}
          title={t("metadata.empty.title")}
          message={t("metadata.empty.message")}
        />
      ) : null}

      {!metadataQuery.isLoading && !metadataQuery.error && metadataRows.length ? (
        <>
          <SectionCard title={t("metadata.filters.title")} eyebrow={t("metadata.filters.eyebrow")}>
            <div className="filter-grid">
              <label className="search-field">
                <Search size={16} />
                <input
                  onChange={(event) => {
                    setSearch(event.target.value);
                    setSelectedIndex(0);
                  }}
                  placeholder={t("metadata.search.placeholder")}
                  value={search}
                />
              </label>
              <label>
                <span>{t("library.filter.library")}</span>
                <select
                  multiple
                  onChange={(event) => {
                    setLibraries(
                      Array.from(event.target.selectedOptions).map((option) => option.value)
                    );
                    setSelectedIndex(0);
                  }}
                  value={libraries}
                >
                  {libraryValues.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>{t("metadata.filter.action")}</span>
                <select
                  multiple
                  onChange={(event) => {
                    setActions(
                      Array.from(event.target.selectedOptions).map((option) => option.value)
                    );
                    setSelectedIndex(0);
                  }}
                  value={actions}
                >
                  {actionValues.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </SectionCard>

          <div className="split-layout">
            <SectionCard
              title={t("metadata.list.title", { count: filtered.length })}
              eyebrow={t("library.catalog.eyebrow")}
              className="split-layout__main"
            >
              <VirtualTable<MetadataRow>
                columns={[
                  {
                    key: "plex_title",
                    label: t("column.plex_title"),
                    width: "28%",
                    render: (row) => String(row.plex_title || t("app.empty_dash"))
                  },
                  {
                    key: "omdb_title",
                    label: t("column.omdb_title"),
                    width: "28%",
                    render: (row) => String(row.omdb_title || t("app.empty_dash"))
                  },
                  {
                    key: "library",
                    label: t("column.library"),
                    width: "20%",
                    render: (row) => String(row.library || t("app.empty_dash"))
                  },
                  {
                    key: "action",
                    label: t("column.action"),
                    width: "14%",
                    render: (row) => translateMetadataAction(row.action, t)
                  },
                  {
                    key: "rating",
                    label: t("column.imdb"),
                    width: "10%",
                    render: (row) => String(row.imdb_rating || t("app.empty_dash"))
                  }
                ]}
                onSelect={setSelectedIndex}
                rows={filtered}
                selectedIndex={selectedIndex}
              />
            </SectionCard>

            <SectionCard title={t("metadata.detail.title")} eyebrow={t("metadata.detail.eyebrow")} className="split-layout__aside">
              {selected ? (
                <dl className="detail-definition-list">
                  {Object.entries(selected).map(([key, value]) => (
                    <div key={key}>
                      <dt>{translateMetadataField(key, t)}</dt>
                      <dd>
                        {key === "action"
                          ? translateMetadataAction(String(value || ""), t)
                          : String(value ?? t("app.empty_dash"))}
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p>{t("metadata.detail.empty")}</p>
              )}
            </SectionCard>
          </div>
        </>
      ) : null}
    </div>
  );
}
