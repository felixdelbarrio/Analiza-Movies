import type { TranslationKey, TranslationParams } from "../i18n/catalog";

export interface ReportRow {
  title?: string | null;
  year?: number | string | null;
  library?: string | null;
  decision?: string | null;
  reason?: string | null;
  reason_code?: string | null;
  file?: string | null;
  file_size?: number | string | null;
  file_size_gb?: number | null;
  imdb_rating?: number | string | null;
  imdb_votes?: number | string | null;
  rt_score?: number | string | null;
  metacritic_score?: number | string | null;
  plex_rating?: number | string | null;
  imdb_id?: string | null;
  guid?: string | null;
  poster_url?: string | null;
  trailer_url?: string | null;
  omdb_json?: string | null;
  genre?: string | null;
  director?: string | null;
  actors?: string | null;
  plot?: string | null;
  wikipedia_title?: string | null;
  wikidata_id?: string | null;
  decade?: number | null;
  decade_label?: string | null;
  search_all?: string | null;
  search_title?: string | null;
  [key: string]: unknown;
}

export interface DuplicateGroup {
  imdbId: string;
  imdbTitle: string;
  year?: number | string | null;
  duplicateCount: number;
  libraryCount: number;
  tone: "neutral" | "keep" | "maybe" | "delete";
  decisionCounts: {
    KEEP: number;
    MAYBE: number;
    DELETE: number;
    UNKNOWN: number;
  };
  imdbRating: number | null;
  rtScore: number | null;
  metacriticScore: number | null;
  primaryRow: ReportRow;
  copies: ReportRow[];
}

export interface MetadataRow {
  library?: string | null;
  action?: string | null;
  plex_title?: string | null;
  plex_year?: string | null;
  omdb_title?: string | null;
  imdb_rating?: string | number | null;
  imdb_votes?: string | number | null;
  [key: string]: unknown;
}

export interface Profile {
  id: string;
  name: string;
  source_type: "plex" | "dlna";
  host?: string | null;
  port?: number | null;
  base_url?: string | null;
  location?: string | null;
  device_id?: string | null;
  machine_identifier?: string | null;
  plex_token?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface RunEvent {
  at: string;
  level: string;
  stage?: string | null;
  scope?: string | null;
  message?: string | null;
  message_key?: TranslationKey | null;
  message_params?: TranslationParams | null;
}

export interface RunProgress {
  run_id: string;
  profile_id: string;
  profile_name: string;
  source_type: string;
  status: string;
  stage?: string | null;
  message?: string | null;
  message_key?: TranslationKey | null;
  message_params?: TranslationParams | null;
  scope?: string | null;
  current?: number | null;
  total?: number | null;
  unit?: string | null;
  percent?: number | null;
  errors?: number | null;
  started_at: string;
  updated_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  counters?: Record<string, number>;
  recent?: RunEvent[];
}

export interface RunState {
  run_id: string;
  profile_id: string;
  profile_name: string;
  source_type: string;
  status: string;
  started_at: string;
  finished_at?: string | null;
  exit_code?: number | null;
  log_path: string;
  pid?: number | null;
  progress?: RunProgress | null;
}

export interface RunStatusResponse {
  run: RunState | null;
}

export interface RunLogsResponse {
  run: RunState | null;
  lines: string[];
}

export interface ConfigState {
  version: number;
  omdb_api_keys: string;
  active_profile_id?: string | null;
  profiles: Profile[];
  updated_at?: string;
  has_omdb_api_keys?: boolean;
}

export interface ServerDiscovery {
  name: string;
  source_type: "plex" | "dlna";
  host?: string | null;
  port?: number | null;
  base_url?: string | null;
  location?: string | null;
  device_id?: string | null;
  machine_identifier?: string | null;
  plex_token?: string | null;
  local?: boolean;
  relay?: boolean;
  uri?: string | null;
  discovery?: string;
}

export interface PagedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface DeleteActionResponse {
  ok: number;
  err: number;
  dry_run: boolean;
  logs: string[];
}

export interface SummaryMetrics {
  totalCount: number;
  totalSizeGb: number;
  keepCount: number;
  keepSizeGb: number;
  deleteCount: number;
  deleteSizeGb: number;
  maybeCount: number;
  maybeSizeGb: number;
  reviewCount: number;
  reviewSizeGb: number;
  imdbMean: number | null;
}

export type DashboardViewKey =
  | "imdb-metacritic"
  | "decision-distribution"
  | "boxplot-library"
  | "imdb-rt"
  | "waste-map"
  | "value-per-gb"
  | "space-library"
  | "decade-distribution"
  | "genre-distribution"
  | "director-ranking"
  | "word-ranking"
  | "imdb-by-decision";
