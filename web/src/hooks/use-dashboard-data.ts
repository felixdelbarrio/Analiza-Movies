import { useQuery } from "@tanstack/react-query";

import {
  fetchConfigState,
  fetchMetadata,
  fetchReportAll,
  fetchReportFiltered,
  fetchRunLogs,
  fetchRunStatus
} from "../lib/api";
import { enrichRows } from "../lib/data";
import { queryKeys } from "../lib/query-keys";

const CONFIG_STALE_TIME_MS = 30_000;
const REPORT_STALE_TIME_MS = 60_000;
const REPORT_GC_TIME_MS = 120_000;

export function useConfigState() {
  return useQuery({
    queryKey: queryKeys.configState(),
    queryFn: fetchConfigState,
    staleTime: CONFIG_STALE_TIME_MS,
    gcTime: REPORT_GC_TIME_MS,
    retry: 1,
    refetchOnWindowFocus: false
  });
}

export function useRunState() {
  return useQuery({
    queryKey: queryKeys.runState(),
    queryFn: fetchRunStatus,
    staleTime: 0,
    gcTime: REPORT_GC_TIME_MS,
    retry: 1,
    refetchOnWindowFocus: false,
    refetchInterval: (query) => {
      const status = query.state.data?.run?.status;
      return status === "running" || status === "stopping" ? 1500 : false;
    },
    refetchIntervalInBackground: true
  });
}

export function useRunLogs(enabled: boolean, limit = 80) {
  return useQuery({
    queryKey: queryKeys.runLogs(limit),
    queryFn: () => fetchRunLogs(limit),
    enabled,
    staleTime: 0,
    gcTime: REPORT_GC_TIME_MS,
    retry: 1,
    refetchOnWindowFocus: false,
    refetchInterval: (query) => {
      const status = query.state.data?.run?.status;
      return status === "running" || status === "stopping" ? 4000 : false;
    },
    refetchIntervalInBackground: true
  });
}

export function useReportAll(profileId?: string | null, enabled = true) {
  return useQuery({
    queryKey: queryKeys.reportAll(profileId),
    queryFn: async () => enrichRows(await fetchReportAll(profileId)),
    enabled,
    staleTime: REPORT_STALE_TIME_MS,
    gcTime: REPORT_GC_TIME_MS,
    retry: 1,
    refetchOnWindowFocus: false
  });
}

export function useReportFiltered(profileId?: string | null, enabled = true) {
  return useQuery({
    queryKey: queryKeys.reportFiltered(profileId),
    queryFn: async () => enrichRows(await fetchReportFiltered(profileId)),
    enabled,
    staleTime: REPORT_STALE_TIME_MS,
    gcTime: REPORT_GC_TIME_MS,
    retry: 1,
    refetchOnWindowFocus: false
  });
}

export function useMetadata(profileId?: string | null, enabled = true) {
  return useQuery({
    queryKey: queryKeys.metadata(profileId),
    queryFn: () => fetchMetadata(profileId),
    enabled,
    staleTime: REPORT_STALE_TIME_MS,
    gcTime: REPORT_GC_TIME_MS,
    retry: 1,
    refetchOnWindowFocus: false
  });
}
