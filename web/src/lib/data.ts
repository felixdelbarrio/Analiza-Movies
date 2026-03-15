import type {
  DashboardViewKey,
  MetadataRow,
  ReportRow,
  SummaryMetrics
} from "./types";

import type { Translator } from "../i18n/catalog";
import { translateDashboardView } from "../i18n/helpers";

export const DASHBOARD_VIEWS: DashboardViewKey[] = [
  "imdb-metacritic",
  "decision-distribution",
  "boxplot-library",
  "imdb-rt",
  "waste-map",
  "value-per-gb",
  "space-library",
  "decade-distribution",
  "genre-distribution",
  "director-ranking",
  "word-ranking",
  "imdb-by-decision"
];

export const REPORT_DECISION_OPTIONS = ["KEEP", "MAYBE", "DELETE", "UNKNOWN"] as const;
export type ReportSearchScope = "all" | "title";

interface ReportFilterOptions {
  search: string;
  searchScope?: ReportSearchScope;
  library?: string | null;
  decision?: string | null;
}

const DEFAULT_DASHBOARD_VIEW_KEYS: DashboardViewKey[] = [
  "imdb-metacritic",
  "decision-distribution",
  "boxplot-library"
];

export function normalizeDashboardViews(values?: string[] | null): DashboardViewKey[] {
  const valid = new Set(DASHBOARD_VIEWS);
  const out: DashboardViewKey[] = [];
  for (const value of values ?? []) {
    if (valid.has(value as DashboardViewKey) && !out.includes(value as DashboardViewKey)) {
      out.push(value as DashboardViewKey);
    }
  }
  return out.length ? out.slice(0, 3) : DEFAULT_DASHBOARD_VIEW_KEYS;
}

export function parseMaybeNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const parsed = Number(String(value).replace(/[,%]/g, "").trim());
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseOmdbJson(row: ReportRow) {
  if (!row.omdb_json || typeof row.omdb_json !== "string") {
    return null;
  }
  try {
    const parsed = JSON.parse(row.omdb_json);
    return typeof parsed === "object" && parsed ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim() && value !== "N/A") {
      return value.trim();
    }
  }
  return null;
}

function buildSearchText(row: ReportRow, metadata: Array<string | null>) {
  const searchParts = [
    row.title,
    row.library,
    row.file,
    row.imdb_id,
    row.year,
    row.decision,
    row.reason,
    row.wikipedia_title,
    ...metadata
  ]
    .filter((value): value is string | number => typeof value === "string" || typeof value === "number")
    .map((value) => String(value).trim())
    .filter(Boolean);

  return searchParts.join(" ").toLowerCase();
}

export function enrichRows(rows: ReportRow[]): ReportRow[] {
  return rows.map((row) => {
    const year = parseMaybeNumber(row.year);
    const fileSize = parseMaybeNumber(row.file_size);
    const fileSizeGb = parseMaybeNumber(row.file_size_gb);
    const omdb = parseOmdbJson(row);
    const metacritic = parseMaybeNumber(
      row.metacritic_score ?? (omdb?.Metascore as string | number | undefined)
    );
    const imdbRating = parseMaybeNumber(row.imdb_rating);
    const rtScore = parseMaybeNumber(row.rt_score);
    const plexRating = parseMaybeNumber(row.plex_rating);
    const genre = firstString(row.genre, omdb?.Genre);
    const director = firstString(row.director, omdb?.Director);
    const actors = firstString(row.actors, omdb?.Actors);
    const plot = firstString(row.plot, omdb?.Plot);
    const decade = year ? Math.floor(year / 10) * 10 : null;
    return {
      ...row,
      year,
      file_size: fileSize,
      file_size_gb: fileSizeGb ?? (fileSize ? fileSize / 1024 ** 3 : 0),
      metacritic_score: metacritic,
      imdb_rating: imdbRating,
      rt_score: rtScore,
      plex_rating: plexRating,
      genre,
      director,
      actors,
      plot,
      decade,
      decade_label: decade ? String(decade) : null,
      search_title: String(row.title || "").trim().toLowerCase(),
      search_all: buildSearchText(row, [genre, director, actors, plot])
    };
  });
}

export function computeSummary(rows: ReportRow[]): SummaryMetrics {
  const totalCount = rows.length;
  let totalSizeGb = 0;
  let keepCount = 0;
  let keepSizeGb = 0;
  let deleteCount = 0;
  let deleteSizeGb = 0;
  let maybeCount = 0;
  let maybeSizeGb = 0;
  let imdbSum = 0;
  let imdbCount = 0;

  rows.forEach((row) => {
    const size = parseMaybeNumber(row.file_size_gb) ?? 0;
    const decision = String(row.decision || "").toUpperCase();
    totalSizeGb += size;
    if (decision === "KEEP") {
      keepCount += 1;
      keepSizeGb += size;
    }
    if (decision === "DELETE") {
      deleteCount += 1;
      deleteSizeGb += size;
    }
    if (decision === "MAYBE") {
      maybeCount += 1;
      maybeSizeGb += size;
    }
    const imdb = parseMaybeNumber(row.imdb_rating);
    if (imdb !== null) {
      imdbSum += imdb;
      imdbCount += 1;
    }
  });

  return {
    totalCount,
    totalSizeGb,
    keepCount,
    keepSizeGb,
    deleteCount,
    deleteSizeGb,
    maybeCount,
    maybeSizeGb,
    reviewCount: deleteCount + maybeCount,
    reviewSizeGb: deleteSizeGb + maybeSizeGb,
    imdbMean: imdbCount ? imdbSum / imdbCount : null
  };
}

export function uniqueValues(rows: ReportRow[], key: keyof ReportRow, locale = "en") {
  return Array.from(
    new Set(
      rows
        .map((row) => row[key])
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .map((value) => value.trim())
    )
  ).sort((left, right) => left.localeCompare(right, locale));
}

export function getPoster(row: ReportRow) {
  if (typeof row.poster_url === "string" && row.poster_url.trim() && row.poster_url !== "N/A") {
    return row.poster_url;
  }
  const omdb = parseOmdbJson(row);
  const poster = omdb?.Poster;
  return typeof poster === "string" && poster !== "N/A" ? poster : null;
}

export function getImdbUrl(row: ReportRow) {
  return row.imdb_id ? `https://www.imdb.com/title/${row.imdb_id}` : null;
}

export function getPlexUrl(row: ReportRow) {
  return typeof row.guid === "string" && row.guid.trim() ? row.guid : null;
}

export function getDecisionTone(decision?: string | null) {
  const value = String(decision || "").toUpperCase();
  if (value === "KEEP") return "keep";
  if (value === "DELETE") return "delete";
  if (value === "MAYBE") return "maybe";
  return "neutral";
}

export function buildReportRowKey(row: ReportRow) {
  return String(
    row.guid ??
      row.file ??
      row.imdb_id ??
      [row.title, row.year, row.library].filter(Boolean).join("::")
  );
}

export function filterReportRows(rows: ReportRow[], filters: ReportFilterOptions) {
  const searchValue = filters.search.trim().toLowerCase();
  const searchScope = filters.searchScope ?? "all";
  const libraryFilter = String(filters.library || "").trim();
  const decisionFilter = String(filters.decision || "").trim().toUpperCase();

  return rows.filter((row) => {
    const matchesLibrary =
      !libraryFilter || (typeof row.library === "string" && row.library === libraryFilter);
    const decision = String(row.decision || "").toUpperCase();
    const matchesDecision = !decisionFilter || decision === decisionFilter;
    const haystack =
      searchScope === "title"
        ? String(row.search_title ?? row.title ?? "").toLowerCase()
        : String(row.search_all ?? "").toLowerCase();
    const matchesSearch = !searchValue || haystack.includes(searchValue);
    return matchesLibrary && matchesDecision && matchesSearch;
  });
}

export function sumReportSizeGb(rows: ReportRow[]) {
  return rows.reduce((total, row) => total + (parseMaybeNumber(row.file_size_gb) ?? 0), 0);
}

export function findDuplicates(rows: ReportRow[]) {
  const counts = new Map<string, number>();
  rows.forEach((row) => {
    const imdbId = typeof row.imdb_id === "string" ? row.imdb_id.trim().toLowerCase() : "";
    if (!imdbId) {
      return;
    }
    counts.set(imdbId, (counts.get(imdbId) ?? 0) + 1);
  });
  return rows
    .filter((row) => {
      const imdbId = typeof row.imdb_id === "string" ? row.imdb_id.trim().toLowerCase() : "";
      return imdbId && (counts.get(imdbId) ?? 0) > 1;
    })
    .map((row) => ({
      ...row,
      dup_count: counts.get(String(row.imdb_id).trim().toLowerCase()) ?? 0
    }));
}

export function filterMetadata(
  rows: MetadataRow[],
  filters: { library: string[]; action: string[]; search: string }
) {
  const term = filters.search.trim().toLowerCase();
  return rows.filter((row) => {
    const matchesLibrary =
      !filters.library.length ||
      (typeof row.library === "string" && filters.library.includes(row.library));
    const matchesAction =
      !filters.action.length ||
      (typeof row.action === "string" && filters.action.includes(row.action));
    const matchesSearch =
      !term ||
      Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(term));
    return matchesLibrary && matchesAction && matchesSearch;
  });
}

export function collectGenres(rows: ReportRow[]) {
  const counts = new Map<string, { keep: number; maybe: number; delete: number }>();
  rows.forEach((row) => {
    const rawGenre = firstString(row.genre, parseOmdbJson(row)?.Genre) ?? "";
    const decision = String(row.decision || "").toUpperCase();
    rawGenre
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .forEach((genre) => {
        const current = counts.get(genre) ?? { keep: 0, maybe: 0, delete: 0 };
        if (decision === "KEEP") current.keep += 1;
        if (decision === "MAYBE") current.maybe += 1;
        if (decision === "DELETE") current.delete += 1;
        counts.set(genre, current);
      });
  });
  return Array.from(counts.entries())
    .map(([genre, value]) => ({
      genre,
      total: value.keep + value.maybe + value.delete,
      ...value
    }))
    .sort((left, right) => right.total - left.total)
    .slice(0, 12);
}

export function collectDirectors(rows: ReportRow[]) {
  const counts = new Map<string, { total: number; ratingTotal: number }>();
  rows.forEach((row) => {
    const rawDirector = firstString(row.director, parseOmdbJson(row)?.Director) ?? "";
    const imdb = parseMaybeNumber(row.imdb_rating) ?? 0;
    rawDirector
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .forEach((director) => {
        const current = counts.get(director) ?? { total: 0, ratingTotal: 0 };
        current.total += 1;
        current.ratingTotal += imdb;
        counts.set(director, current);
      });
  });
  return Array.from(counts.entries())
    .map(([director, value]) => ({
      director,
      total: value.total,
      mean: value.total ? value.ratingTotal / value.total : 0
    }))
    .sort((left, right) => right.mean - left.mean)
    .slice(0, 12);
}

export function collectTitleWords(rows: ReportRow[]) {
  const stopWords = new Set([
    "the",
    "and",
    "las",
    "los",
    "una",
    "uno",
    "con",
    "para",
    "del",
    "que",
    "por"
  ]);
  const counts = new Map<string, number>();
  rows
    .filter((row) => ["DELETE", "MAYBE"].includes(String(row.decision || "").toUpperCase()))
    .forEach((row) => {
      String(row.title || "")
        .toLowerCase()
        .replace(/[^a-z0-9áéíóúñü\s]/gi, " ")
        .split(/\s+/)
        .map((item) => item.trim())
        .filter((item) => item.length > 2 && !stopWords.has(item))
        .forEach((word) => counts.set(word, (counts.get(word) ?? 0) + 1));
    });

  return Array.from(counts.entries())
    .map(([word, count]) => ({ word, count }))
    .sort((left, right) => right.count - left.count)
    .slice(0, 14);
}

export function getDashboardViews(t: Translator) {
  return DASHBOARD_VIEWS.map((key) => ({ key, label: translateDashboardView(key, t) }));
}

export function formatCountSize(count: number, sizeGb: number, locale: string, t: Translator) {
  const countLabel = count.toLocaleString(locale);
  const sizeLabel = sizeGb.toLocaleString(locale, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1
  });
  return t("data.count_size", {
    count: countLabel,
    size: sizeLabel
  });
}
