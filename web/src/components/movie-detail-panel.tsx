import { ExternalLink, Film, LibraryBig } from "lucide-react";

import { getDecisionTone, getImdbUrl, getPlexUrl, getPoster, parseOmdbJson } from "../lib/data";
import type { ReportRow } from "../lib/types";

interface MovieDetailPanelProps {
  row?: ReportRow | null;
}

function metric(label: string, value: string | number | null | undefined) {
  return (
    <div className="detail-metric">
      <span>{label}</span>
      <strong>{value ?? "N/A"}</strong>
    </div>
  );
}

export function MovieDetailPanel({ row }: MovieDetailPanelProps) {
  if (!row) {
    return (
      <div className="detail-panel detail-panel--empty">
        <Film size={28} />
        <p>Selecciona una fila para abrir la ficha editorial de la película.</p>
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
        {poster ? <img alt={String(row.title || "Poster")} src={poster} /> : <span>Sin póster</span>}
      </div>

      <div className="detail-panel__body">
        <div className="detail-panel__header">
          <div>
            <span className="detail-panel__eyebrow">{row.library || "Biblioteca"}</span>
            <h3>
              {row.title}
              {row.year ? <small>{` · ${row.year}`}</small> : null}
            </h3>
          </div>
          <span className={`decision-chip decision-chip--${tone}`}>
            {String(row.decision || "UNKNOWN").toUpperCase()}
          </span>
        </div>

        <div className="detail-panel__links">
          {imdbUrl ? (
            <a href={imdbUrl} rel="noreferrer" target="_blank">
              IMDb <ExternalLink size={14} />
            </a>
          ) : null}
          {plexUrl ? (
            <a href={plexUrl} rel="noreferrer" target="_blank">
              Plex <ExternalLink size={14} />
            </a>
          ) : null}
          {row.file ? (
            <span className="detail-panel__path">
              <LibraryBig size={14} />
              {row.file}
            </span>
          ) : null}
        </div>

        <div className="detail-metrics-grid">
          {metric("IMDb", row.imdb_rating as number | string | null)}
          {metric("RT", row.rt_score ? `${row.rt_score}%` : null)}
          {metric("Metacritic", row.metacritic_score as number | string | null)}
          {metric("Tamaño", row.file_size_gb ? `${Number(row.file_size_gb).toFixed(1)} GB` : null)}
        </div>

        {row.reason ? (
          <section className="detail-copy">
            <span>Razonamiento</span>
            <p>{String(row.reason)}</p>
          </section>
        ) : null}

        {typeof omdb?.Plot === "string" && omdb.Plot.trim() && omdb.Plot !== "N/A" ? (
          <section className="detail-copy">
            <span>Sinopsis</span>
            <p>{omdb.Plot}</p>
          </section>
        ) : null}

        <dl className="detail-definition-list">
          <div>
            <dt>Director</dt>
            <dd>{typeof omdb?.Director === "string" ? omdb.Director : "—"}</dd>
          </div>
          <div>
            <dt>Género</dt>
            <dd>{typeof omdb?.Genre === "string" ? omdb.Genre : "—"}</dd>
          </div>
          <div>
            <dt>Actores</dt>
            <dd>{typeof omdb?.Actors === "string" ? omdb.Actors : "—"}</dd>
          </div>
          <div>
            <dt>Wikipedia</dt>
            <dd>{typeof row.wikipedia_title === "string" ? row.wikipedia_title : "—"}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}
