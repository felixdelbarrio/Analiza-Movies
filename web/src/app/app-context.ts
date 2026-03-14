import type { ConfigState, MetadataRow, ReportRow } from "../lib/types";

export interface AppPreferences {
  theme: "aurora" | "cinema";
  dashboardViews: string[];
  numericFilters: boolean;
  chartThresholds: boolean;
  setTheme: (value: "aurora" | "cinema") => void;
  setDashboardViews: (value: string[]) => void;
  setNumericFilters: (value: boolean) => void;
  setChartThresholds: (value: boolean) => void;
}

export interface AppOutletContext {
  config: ConfigState | null;
  activeProfileId: string | null;
  reportAll: ReportRow[];
  reportFiltered: ReportRow[];
  metadataRows: MetadataRow[];
  preferences: AppPreferences;
  isLoading: boolean;
  refreshAll: () => void;
}
