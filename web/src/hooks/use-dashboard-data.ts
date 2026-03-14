import { useQuery } from "@tanstack/react-query";

import { fetchConfigState, fetchMetadata, fetchReportAll, fetchReportFiltered } from "../lib/api";
import { enrichRows } from "../lib/data";

export function useConfigState() {
  return useQuery({
    queryKey: ["config-state"],
    queryFn: fetchConfigState,
    staleTime: 10_000
  });
}

export function useReportAll(profileId?: string | null) {
  return useQuery({
    queryKey: ["report-all", profileId],
    queryFn: async () => enrichRows(await fetchReportAll(profileId)),
    staleTime: 15_000
  });
}

export function useReportFiltered(profileId?: string | null) {
  return useQuery({
    queryKey: ["report-filtered", profileId],
    queryFn: async () => enrichRows(await fetchReportFiltered(profileId)),
    staleTime: 15_000
  });
}

export function useMetadata(profileId?: string | null) {
  return useQuery({
    queryKey: ["metadata-fix", profileId],
    queryFn: () => fetchMetadata(profileId),
    staleTime: 15_000
  });
}
