import { ExternalLink, Film, FolderTree } from "lucide-react";

import { translateDecision } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { getImdbUrl, getPlexUrl, getPoster, parseMaybeNumber } from "../lib/data";
import { openInAppContainer } from "../lib/desktop";
import type { DuplicateGroup } from "../lib/types";

interface DuplicateDetailPanelProps {
  group?: DuplicateGroup | null;
}

function metric(label: string, value: string | number | null | undefined, fallback: string) {
  return (
    <div className="detail-metric">
      <span>{label}</span>
      <strong>{value ?? fallback}</strong>
    </div>
  );
}

export function DuplicateDetailPanel({ group }: DuplicateDetailPanelProps) {
  const { locale, t } = useI18n();

  if (!group) {
    return (
      <div className="detail-panel detail-panel--empty">
        <Film size={28} />
        <p>{t("duplicates.detail.empty")}</p>
      </div>
    );
  }

  const poster = getPoster(group.primaryRow);
  const imdbUrl = getImdbUrl(group.primaryRow);
  const plexUrl = getPlexUrl(group.primaryRow);

  return (
    <article className={`detail-panel duplicate-detail-panel detail-panel--${group.tone}`}>
      <div className="detail-panel__hero">
        <div className="detail-panel__poster">
          {poster ? (
            <img alt={String(group.imdbTitle || t("detail.poster_alt"))} src={poster} />
          ) : (
            <span>{t("detail.poster_missing")}</span>
          )}
        </div>

        <div className="detail-panel__hero-copy">
          <div className="detail-panel__header">
            <div>
              <span className="detail-panel__eyebrow">{group.imdbId}</span>
              <h3>
                {group.imdbTitle}
                {group.year ? <small>{` · ${group.year}`}</small> : null}
              </h3>
            </div>
            <span className={`decision-chip decision-chip--${group.tone}`}>
              {translateDecision(
                group.tone === "delete" ? "DELETE" : "MAYBE",
                t
              )}
            </span>
          </div>

          <p className="duplicate-detail-panel__summary">
            {t("duplicates.detail.summary", {
              count: group.duplicateCount,
              libraries: group.libraryCount
            })}
          </p>

          <div className="detail-panel__links">
            {imdbUrl ? (
              <button
                className="secondary-button"
                onClick={() =>
                  void openInAppContainer(
                    imdbUrl,
                    `IMDb · ${String(group.imdbTitle || t("detail.fallback.movie"))}`
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
                    `Plex · ${String(group.imdbTitle || t("detail.fallback.movie"))}`
                  )
                }
                type="button"
              >
                Plex <ExternalLink size={14} />
              </button>
            ) : null}
          </div>

          <div className="detail-metrics-grid">
            {metric(
              t("detail.metric.imdb"),
              group.imdbRating !== null ? group.imdbRating.toFixed(1) : null,
              t("app.na")
            )}
            {metric(
              t("detail.metric.rt"),
              group.rtScore !== null ? `${group.rtScore}%` : null,
              t("app.na")
            )}
            {metric(
              t("detail.metric.metacritic"),
              group.metacriticScore !== null ? Math.round(group.metacriticScore) : null,
              t("app.na")
            )}
          </div>
        </div>
      </div>

      <div className="duplicate-detail-panel__section">
        <div className="duplicate-detail-panel__section-header">
          <div>
            <span>{t("duplicates.detail.routes_eyebrow")}</span>
            <h4>{t("duplicates.detail.routes_title")}</h4>
          </div>
          <p>{t("duplicates.detail.routes_hint")}</p>
        </div>

        <div className="duplicate-route-list">
          {group.copies.map((copy, index) => {
            const sizeValue = parseMaybeNumber(copy.file_size_gb);
            const size = sizeValue !== null
              ? `${sizeValue.toLocaleString(locale, {
                  minimumFractionDigits: 1,
                  maximumFractionDigits: 1
                })} ${t("unit.gb")}`
              : t("app.na");
            const tone =
              String(copy.decision || "").toUpperCase() === "DELETE"
                ? "delete"
                : String(copy.decision || "").toUpperCase() === "KEEP"
                  ? "keep"
                : String(copy.decision || "").toUpperCase() === "MAYBE"
                  ? "maybe"
                  : "maybe";

            return (
              <article
                key={`${group.imdbId}-${index}-${String(copy.file || copy.guid || copy.title)}`}
                className={`duplicate-route duplicate-route--${tone}`}
              >
                <div className="duplicate-route__meta">
                  <span className="duplicate-route__library">
                    {copy.library || t("detail.fallback.library")}
                  </span>
                  <span className={`decision-chip decision-chip--${tone}`}>
                    {translateDecision(copy.decision, t)}
                  </span>
                </div>

                <div className="duplicate-route__facts">
                  <div className="duplicate-route__path">
                    <FolderTree size={15} />
                    <span>{copy.file || t("app.empty_dash")}</span>
                  </div>
                  <strong>{size}</strong>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </article>
  );
}
