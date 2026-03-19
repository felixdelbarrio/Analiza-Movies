import { Expand, ExternalLink, Film, LibraryBig } from "lucide-react";
import type { ReactNode } from "react";

import { translateDecision, translateReason } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import {
  getDecisionTone,
  getImdbUrl,
  getPlexUrl,
  getPoster,
  parseOmdbJson
} from "../lib/data";
import { openInAppContainer } from "../lib/desktop";
import type { ConsolidatedPayload, ReportRow } from "../lib/types";

interface MovieDetailPanelProps {
  details?: ConsolidatedPayload | null;
  detailsLoading?: boolean;
  onOpenDetail?: (() => void) | null;
  row?: ReportRow | null;
  variant?: "panel" | "modal";
}

interface DetailField {
  label: string;
  value: string;
}

function metric(label: string, value: string | number | null | undefined, fallback: string) {
  return (
    <div className="detail-metric">
      <span>{label}</span>
      <strong>{value ?? fallback}</strong>
    </div>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function cleanDetailValue(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed || trimmed === "N/A") {
      return null;
    }
    return trimmed;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => cleanDetailValue(item))
      .filter((item): item is string => Boolean(item));
    return parts.length ? parts.join(", ") : null;
  }
  if (isRecord(value)) {
    try {
      return JSON.stringify(value);
    } catch {
      return null;
    }
  }
  return null;
}

function firstDetailValue(...values: unknown[]) {
  for (const value of values) {
    const clean = cleanDetailValue(value);
    if (clean) {
      return clean;
    }
  }
  return null;
}

function pushField(fields: DetailField[], label: string, value: unknown) {
  const clean = cleanDetailValue(value);
  if (!clean) {
    return;
  }
  fields.push({ label, value: clean });
}

function renderFieldSection(title: string, fields: DetailField[]) {
  if (!fields.length) {
    return null;
  }
  return (
    <section className="detail-panel__section">
      <span className="detail-panel__section-title">{title}</span>
      <dl className="detail-definition-list">
        {fields.map((field) => (
          <div key={`${title}-${field.label}`}>
            <dt>{field.label}</dt>
            <dd>{field.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function sourceTimestamp(
  value: string | number | null | undefined,
  formatTime: ReturnType<typeof useI18n>["formatTime"]
) {
  if (typeof value === "number") {
    return formatTime(value * 1000);
  }
  if (typeof value === "string" && value.trim()) {
    return formatTime(value);
  }
  return null;
}

export function MovieDetailPanel({
  details,
  detailsLoading = false,
  onOpenDetail = null,
  row,
  variant = "panel"
}: MovieDetailPanelProps) {
  const { locale, formatTime, t } = useI18n();
  if (!row) {
    return (
      <div className="detail-panel detail-panel--empty">
        <Film size={28} />
        <p>{t("detail.empty")}</p>
      </div>
    );
  }

  const omdbFromRow = parseOmdbJson(row);
  const merged = isRecord(details?.merged) ? details.merged : null;
  const omdb = isRecord(details?.omdb) ? details.omdb : omdbFromRow;
  const wiki = isRecord(details?.wiki) ? details.wiki : null;
  const wikidata = isRecord(details?.wikidata) ? details.wikidata : null;
  const wikiFromOmdbCache = isRecord(details?.wiki_from_omdb_cache)
    ? details.wiki_from_omdb_cache
    : null;
  const poster = getPoster(row) ?? firstDetailValue(merged?.poster);
  const imdbUrl = getImdbUrl(row);
  const plexUrl = getPlexUrl(row);
  const tone = getDecisionTone(row.decision);
  const title = firstDetailValue(row.title, merged?.title) ?? t("app.empty_dash");
  const year = firstDetailValue(row.year, merged?.year);
  const director = firstDetailValue(row.director, merged?.director, omdb?.Director);
  const genre = firstDetailValue(row.genre, merged?.genre, omdb?.Genre);
  const actors = firstDetailValue(row.actors, merged?.actors, omdb?.Actors);
  const plot = firstDetailValue(row.plot, merged?.plot, omdb?.Plot);
  const writer = firstDetailValue(omdb?.Writer);
  const runtime = firstDetailValue(omdb?.Runtime);
  const country = firstDetailValue(omdb?.Country);
  const language = firstDetailValue(omdb?.Language);
  const awards = firstDetailValue(omdb?.Awards);
  const rated = firstDetailValue(omdb?.Rated);
  const boxOffice = firstDetailValue(omdb?.BoxOffice);
  const imdbVotes = firstDetailValue(row.imdb_votes, merged?.imdbVotes, omdb?.imdbVotes);
  const wikipediaTitle = firstDetailValue(
    row.wikipedia_title,
    merged?.wikipedia_title,
    wiki?.wikipedia_title,
    wikiFromOmdbCache?.wikipedia_title
  );
  const wikipediaSummary = firstDetailValue(
    row.wikipedia_summary,
    wiki?.summary,
    wiki?.description
  );
  const synopsis = wikipediaSummary ?? plot;
  const wikidataId = firstDetailValue(
    row.wikidata_id,
    merged?.wikidata_id,
    wikidata?.wikidata_id,
    wikidata?.qid,
    wikiFromOmdbCache?.wikidata_id
  );
  const sourceLanguage = firstDetailValue(
    row.source_language,
    merged?.source_language,
    wiki?.source_language,
    wiki?.language,
    wikiFromOmdbCache?.source_language
  );
  const ratingValues = firstDetailValue(omdb?.Ratings);
  const omdbFetchedAt = sourceTimestamp(details?.sources?.omdb?.fetched_at, formatTime);
  const wikiFetchedAt = sourceTimestamp(details?.sources?.wiki?.fetched_at, formatTime);

  const editorialFields: DetailField[] = [];
  pushField(editorialFields, t("detail.director"), director);
  pushField(editorialFields, t("detail.genre"), genre);
  pushField(editorialFields, t("detail.actors"), actors);

  const modalEditorialFields: DetailField[] = [];
  pushField(modalEditorialFields, "Writer", writer);
  pushField(modalEditorialFields, "Runtime", runtime);
  pushField(modalEditorialFields, "Country", country);
  pushField(modalEditorialFields, "Language", language);
  pushField(modalEditorialFields, "Awards", awards);
  pushField(modalEditorialFields, "Rated", rated);
  pushField(modalEditorialFields, "Box office", boxOffice);
  pushField(modalEditorialFields, "IMDb votes", imdbVotes);
  pushField(modalEditorialFields, t("detail.wikipedia"), wikipediaTitle);
  pushField(modalEditorialFields, "Wikidata", wikidataId);
  pushField(modalEditorialFields, "Source language", sourceLanguage);
  pushField(modalEditorialFields, "Ratings", ratingValues);

  const technicalFields: DetailField[] = [];
  pushField(technicalFields, "IMDb ID", row.imdb_id);
  pushField(technicalFields, "GUID", row.guid);
  pushField(technicalFields, "Rating key", row.rating_key);
  pushField(technicalFields, "Trailer", row.trailer_url);
  pushField(technicalFields, "Thumb", row.thumb);
  pushField(technicalFields, "Path", row.file);

  const sourceFields: DetailField[] = [];
  pushField(sourceFields, "OMDb status", details?.sources?.omdb?.status);
  pushField(sourceFields, "OMDb fetched", omdbFetchedAt);
  pushField(sourceFields, "Wiki status", details?.sources?.wiki?.status);
  pushField(sourceFields, "Wiki fetched", wikiFetchedAt);

  const omdbFields: DetailField[] = [];
  pushField(omdbFields, "Released", omdb?.Released);
  pushField(omdbFields, "DVD", omdb?.DVD);
  pushField(omdbFields, "Type", omdb?.Type);
  pushField(omdbFields, "Production", omdb?.Production);
  pushField(omdbFields, "Website", omdb?.Website);

  const wikiFields: DetailField[] = [];
  pushField(wikiFields, t("detail.wikipedia"), wiki?.wikipedia_title);
  pushField(wikiFields, "Wikibase item", wiki?.wikibase_item);
  pushField(wikiFields, "Page ID", wiki?.wikipedia_pageid);
  pushField(wikiFields, "Language", wiki?.language);
  pushField(wikiFields, "Source language", wiki?.source_language);

  const wikidataFields: DetailField[] = [];
  pushField(wikidataFields, "QID", wikidata?.qid);
  pushField(wikidataFields, "Directors", wikidata?.directors);
  pushField(wikidataFields, "Countries", wikidata?.countries);
  pushField(wikidataFields, "Genres", wikidata?.genres);

  let modalContent: ReactNode = null;
  if (variant === "modal") {
    modalContent = (
      <>
        {detailsLoading ? <p className="detail-panel__helper">{t("stage.connecting")}</p> : null}
        {renderFieldSection("Editorial", modalEditorialFields)}
        {renderFieldSection("Sources", sourceFields)}
        {renderFieldSection("OMDb", omdbFields)}
        {renderFieldSection("Wikipedia", wikiFields)}
        {renderFieldSection("Wikidata", wikidataFields)}
        {renderFieldSection("Technical", technicalFields)}
      </>
    );
  }

  return (
    <article className={`detail-panel detail-panel--${tone} detail-panel--${variant}`}>
      <div className="detail-panel__hero">
        <div className="detail-panel__poster">
          {poster ? (
            <img alt={String(title || t("detail.poster_alt"))} src={poster} />
          ) : (
            <span>{t("detail.poster_missing")}</span>
          )}
        </div>

        <div className="detail-panel__hero-copy">
          <div className="detail-panel__header">
            <div>
              <span className={`decision-chip decision-chip--${tone}`}>
                {translateDecision(String(row.decision || "UNKNOWN"), t)}
              </span>
              {variant === "panel" && onOpenDetail ? (
                <button className="detail-panel__title-button" onClick={onOpenDetail} type="button">
                  <span className="detail-panel__title-text">
                    {title}
                    {year ? <small>{` · ${year}`}</small> : null}
                  </span>
                  <Expand size={16} />
                </button>
              ) : (
                <h3>
                  {title}
                  {year ? <small>{` · ${year}`}</small> : null}
                </h3>
              )}
            </div>
          </div>

          <div className="detail-metrics-grid">
            {metric(
              t("detail.metric.imdb"),
              row.imdb_rating as number | string | null,
              t("app.na")
            )}
            {metric(t("detail.metric.rt"), row.rt_score ? `${row.rt_score}%` : null, t("app.na"))}
            {metric(
              t("detail.metric.metacritic"),
              row.metacritic_score as number | string | null,
              t("app.na")
            )}
            {metric(
              t("detail.metric.size"),
              row.file_size_gb
                ? `${Number(row.file_size_gb).toLocaleString(locale, {
                    minimumFractionDigits: 1,
                    maximumFractionDigits: 1
                  })} ${t("unit.gb")}`
                : null,
              t("app.na")
            )}
          </div>
        </div>
      </div>

      <div className="detail-panel__content">
        {variant === "modal" && (row.reason || row.reason_code) ? (
          <section className="detail-copy">
            <span>{t("detail.reasoning")}</span>
            <p>{translateReason(row, t)}</p>
          </section>
        ) : null}

        {synopsis ? (
          <section className="detail-copy">
            <span>{t("detail.plot")}</span>
            <p>{synopsis}</p>
          </section>
        ) : null}

        {imdbUrl || plexUrl ? (
          <div className="detail-panel__links">
            {imdbUrl ? (
              <button
                className="secondary-button"
                onClick={() =>
                  void openInAppContainer(
                    imdbUrl,
                    `IMDb · ${String(title || t("detail.fallback.movie"))}`
                  )
                }
                type="button"
              >
                IMDb <ExternalLink size={14} />
              </button>
            ) : null}
            {plexUrl ? (
              <button
                className="secondary-button"
                onClick={() =>
                  void openInAppContainer(
                    plexUrl,
                    `Plex · ${String(title || t("detail.fallback.movie"))}`
                  )
                }
                type="button"
              >
                Plex <ExternalLink size={14} />
              </button>
            ) : null}
          </div>
        ) : null}

        {row.file ? (
          <section className="detail-copy detail-copy--path">
            <span>{t("column.route")}</span>
            <p className="detail-panel__path">
              <LibraryBig size={14} />
              {row.file}
            </p>
          </section>
        ) : null}

        {editorialFields.length ? (
          <dl className="detail-definition-list">
            {editorialFields.map((field) => (
              <div key={field.label}>
                <dt>{field.label}</dt>
                <dd>{field.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        {modalContent}
      </div>
    </article>
  );
}
