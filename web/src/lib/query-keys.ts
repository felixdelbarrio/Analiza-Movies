export const queryKeys = {
  configState: () => ["config-state"] as const,
  runState: () => ["run-state"] as const,
  runLogs: (limit = 80) => ["run-logs", limit] as const,
  reportAll: (profileId?: string | null) => ["report-all", profileId ?? "__default__"] as const,
  reportFiltered: (profileId?: string | null) =>
    ["report-filtered", profileId ?? "__default__"] as const,
  metadata: (profileId?: string | null) => ["metadata-fix", profileId ?? "__default__"] as const
};
