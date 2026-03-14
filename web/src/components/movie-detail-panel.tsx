import { ExternalLink, Film, LibraryBig } from "lucide-react";

import { translateDecision, translateReason } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { getDecisionTone, getImdbUrl, getPlexUrl, getPoster, parseOmdbJson } from "../lib/data";
import { openInAppContainer } from "../lib/desktop";
import type { ReportRow } from "../lib/types";

interface MovieDetailPanelProps {
  row?: ReportRow | null;
}

function metric(label: string, value: string | number | null | undefined, fallback: string) {
  return (
    <div className="detail-metric">
      <span>{label}</span>
      <strong>{value ?? fallback}</strong>
    </div>
  );
}

export function MovieDetailPanel({ row }: MovieDetailPanelProps) {
  const { locale, t } = useI18n();
  if (!row) {
    return (
      <div className="detail-panel detail-panel--empty">
        <Film size={28} />
        <p>{t("detail.empty")}</p>
      </div>
    );
  }

  const omdb = parseOmdbJson(row);
  const poster = getPoster(row);
  const imdbUrl = getImdbUrl(row);
  const plexUrl = getPlexUrl(row);
  const tone = getDecisionTone(row.decision);

  return (
    <article className={`detail-panel detail-panel--${tone}`}>
      <div className="detail-panel__poster">
        {poster ? <img alt={String(row.title || t("detail.poster_alt"))} src={poster} /> : <span>{t("detail.poster_missing")}</span>}
      </div>

      <div className="detail-panel__body">
        <div className="detail-panel__header">
          <div>
            <span className="detail-panel__eyebrow">{row.library || t("detail.fallback.library")}</span>
            <h3>
              {row.title}
              {row.year ? <small>{` · ${row.year}`}</small> : null}
            </h3>
          </div>
          <span className={`decision-chip decision-chip--${tone}`}>
            {translateDecision(row.decision, t)}
          </span>
        </div>

        <div className="detail-panel__links">
          {imdbUrl ? (
            <button
              className="secondary-button"
              onClick={() => void openInAppContainer(imdbUrl, `IMDb · ${String(row.title || t("detail.fallback.movie"))}`)}
              type="button"
            >
              IMDb <ExternalLink size={14} />
            </button>
          ) : null}
          {plexUrl ? (
            <button
              className="secondary-button"
              onClick={() => void openInAppContainer(plexUrl, `Plex · ${String(row.title || t("detail.fallback.movie"))}`)}
              type="button"
            >
              Plex <ExternalLink size={14} />
            </button>
          ) : null}
          {row.file ? (
            <span className="detail-panel__path">
              <LibraryBig size={14} />
              {row.file}
            </span>
          ) : null}
        </div>

        <div className="detail-metrics-grid">
          {metric(t("detail.metric.imdb"), row.imdb_rating as number | string | null, t("app.na"))}
          {metric(t("detail.metric.rt"), row.rt_score ? `${row.rt_score}%` : null, t("app.na"))}
          {metric(t("detail.metric.metacritic"), row.metacritic_score as number | string | null, t("app.na"))}
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

        {row.reason || row.reason_code ? (
          <section className="detail-copy">
            <span>{t("detail.reasoning")}</span>
            <p>{translateReason(row, t)}</p>
          </section>
        ) : null}

        {typeof omdb?.Plot === "string" && omdb.Plot.trim() && omdb.Plot !== "N/A" ? (
          <section className="detail-copy">
            <span>{t("detail.plot")}</span>
            <p>{omdb.Plot}</p>
          </section>
        ) : null}

        <dl className="detail-definition-list">
          <div>
            <dt>{t("detail.director")}</dt>
            <dd>{typeof omdb?.Director === "string" ? omdb.Director : t("app.empty_dash")}</dd>
          </div>
          <div>
            <dt>{t("detail.genre")}</dt>
            <dd>{typeof omdb?.Genre === "string" ? omdb.Genre : t("app.empty_dash")}</dd>
          </div>
          <div>
            <dt>{t("detail.actors")}</dt>
            <dd>{typeof omdb?.Actors === "string" ? omdb.Actors : t("app.empty_dash")}</dd>
          </div>
          <div>
            <dt>{t("detail.wikipedia")}</dt>
            <dd>{typeof row.wikipedia_title === "string" ? row.wikipedia_title : t("app.empty_dash")}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}
