import type { ConfigState, RunState } from "../lib/types";

export interface AppPreferences {
  theme: "aurora" | "cinema";
  locale: "en" | "es" | "fr" | "de" | "it" | "pt";
  dashboardViews: string[];
  numericFilters: boolean;
  chartThresholds: boolean;
  setTheme: (value: "aurora" | "cinema") => void;
  setLocale: (value: "en" | "es" | "fr" | "de" | "it" | "pt") => void;
  setDashboardViews: (value: string[]) => void;
  setNumericFilters: (value: boolean) => void;
  setChartThresholds: (value: boolean) => void;
}

export interface AppOutletContext {
  config: ConfigState | null;
  run: RunState | null;
  activeProfileId: string | null;
  preferences: AppPreferences;
  refreshConfig: () => Promise<void>;
  refreshRun: () => Promise<void>;
}
