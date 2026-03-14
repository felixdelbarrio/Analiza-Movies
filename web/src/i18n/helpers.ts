import type { DashboardViewKey, MetadataRow, ReportRow, RunEvent, RunProgress } from "../lib/types";
import type { TranslationKey, Translator } from "./catalog";

const DASHBOARD_VIEW_LABELS: Record<DashboardViewKey, TranslationKey> = {
  "imdb-metacritic": "dashboard.view.imdb-metacritic",
  "decision-distribution": "dashboard.view.decision-distribution",
  "boxplot-library": "dashboard.view.boxplot-library",
  "imdb-rt": "dashboard.view.imdb-rt",
  "waste-map": "dashboard.view.waste-map",
  "value-per-gb": "dashboard.view.value-per-gb",
  "space-library": "dashboard.view.space-library",
  "decade-distribution": "dashboard.view.decade-distribution",
  "genre-distribution": "dashboard.view.genre-distribution",
  "director-ranking": "dashboard.view.director-ranking",
  "word-ranking": "dashboard.view.word-ranking",
  "imdb-by-decision": "dashboard.view.imdb-by-decision"
};

const DECISION_LABELS: Record<string, TranslationKey> = {
  KEEP: "decision.keep",
  MAYBE: "decision.maybe",
  DELETE: "decision.delete",
  UNKNOWN: "decision.unknown"
};

const STATUS_LABELS: Record<string, TranslationKey> = {
  running: "status.running",
  succeeded: "status.succeeded",
  failed: "status.failed",
  cancelled: "status.cancelled",
  stopping: "status.stopping"
};

const STAGE_LABELS: Record<string, TranslationKey> = {
  queued: "stage.queued",
  booting: "stage.booting",
  dashboard: "stage.dashboard",
  starting: "stage.starting",
  connecting: "stage.connecting",
  planning: "stage.planning",
  scanning: "stage.scanning",
  analyzing: "stage.analyzing",
  writing: "stage.writing",
  finished: "stage.finished",
  failed: "stage.failed",
  stopping: "stage.stopping"
};

const UNIT_LABELS: Record<string, TranslationKey> = {
  movies: "unit.movies",
  items: "unit.items",
  containers: "unit.containers",
  libraries: "unit.libraries"
};

const METADATA_ACTION_LABELS: Record<string, TranslationKey> = {
  "Skip (IMDb mismatch)": "metadata.action.skip_imdb_mismatch",
  "Fix title & year": "metadata.action.fix_title_year",
  "Fix title": "metadata.action.fix_title",
  "Fix year (alt title info)": "metadata.action.fix_year_alt",
  "Alt title info": "metadata.action.alt_title_info",
  "Fix year": "metadata.action.fix_year"
};

const METADATA_FIELD_LABELS: Record<string, TranslationKey> = {
  plex_guid: "metadata.field.plex_guid",
  library: "metadata.field.library",
  context_lang: "metadata.field.context_lang",
  plex_original_title: "metadata.field.plex_original_title",
  plex_imdb_id: "metadata.field.plex_imdb_id",
  omdb_imdb_id: "metadata.field.omdb_imdb_id",
  imdb_rating: "metadata.field.imdb_rating",
  imdb_votes: "metadata.field.imdb_votes",
  plex_title: "metadata.field.plex_title",
  plex_year: "metadata.field.plex_year",
  omdb_title: "metadata.field.omdb_title",
  omdb_year: "metadata.field.omdb_year",
  action: "metadata.field.action",
  suggestions_json: "metadata.field.suggestions_json"
};

const REASON_LABELS: Record<string, TranslationKey> = {
  strong_without_external: "detail.reason.strong_without_external",
  insufficient_signals_no_external: "detail.reason.insufficient_signals_no_external",
  external_failed_or_unusable: "detail.reason.external_failed_or_unusable",
  external_unusable_no_signals: "detail.reason.external_unusable_no_signals",
  external_signals_applied: "detail.reason.external_signals_applied",
  unknown: "detail.reason.unknown"
};

export function translateDashboardView(viewKey: DashboardViewKey, t: Translator) {
  return t(DASHBOARD_VIEW_LABELS[viewKey]);
}

export function translateDecision(decision: string | null | undefined, t: Translator) {
  const key = DECISION_LABELS[String(decision || "UNKNOWN").trim().toUpperCase()] ?? "decision.unknown";
  return t(key);
}

export function translateStatus(status: string | null | undefined, t: Translator) {
  const key = STATUS_LABELS[String(status || "").trim().toLowerCase()] ?? "status.running";
  return t(key);
}

export function translateStage(stage: string | null | undefined, t: Translator) {
  const key = STAGE_LABELS[String(stage || "").trim().toLowerCase()] ?? "stage.state";
  return t(key);
}

export function translateUnit(unit: string | null | undefined, t: Translator) {
  const key = UNIT_LABELS[String(unit || "").trim().toLowerCase()];
  return key ? t(key) : String(unit || "").trim();
}

export function translateMetadataAction(action: MetadataRow["action"], t: Translator) {
  if (!action) {
    return t("app.empty_dash");
  }
  const key = METADATA_ACTION_LABELS[String(action).trim()];
  return key ? t(key) : String(action);
}

export function translateMetadataField(field: string, t: Translator) {
  const key = METADATA_FIELD_LABELS[field];
  return key ? t(key) : field.replaceAll("_", " ");
}

export function translateReason(row: ReportRow, t: Translator) {
  const key = REASON_LABELS[String(row.reason_code || "").trim()];
  if (key) {
    return t(key);
  }
  if (typeof row.reason === "string" && row.reason.trim()) {
    return row.reason;
  }
  return t("detail.reason.unknown");
}

export function translateRunMessage(
  progressOrEvent: Pick<RunProgress, "message" | "message_key" | "message_params"> | Pick<RunEvent, "message" | "message_key" | "message_params"> | null | undefined,
  t: Translator
) {
  const key = progressOrEvent?.message_key;
  if (key) {
    return t(key, progressOrEvent.message_params ?? undefined);
  }
  if (progressOrEvent?.message) {
    return progressOrEvent.message;
  }
  return "";
}
