import {
  QueryClient,
  QueryClientProvider,
  useQueryClient
} from "@tanstack/react-query";
import { Suspense, lazy, useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { Navigate, RouterProvider, createBrowserRouter } from "react-router-dom";

import { useConfigState, useRunState } from "../hooks/use-dashboard-data";
import { I18nProvider, useI18n } from "../i18n/provider";
import { useStoredState } from "../lib/preferences";
import { queryKeys } from "../lib/query-keys";
import { AppShell } from "../components/app-shell";
import type { AppOutletContext } from "./app-context";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1
    }
  }
});

const DashboardPage = lazy(async () => {
  const module = await import("../pages/dashboard-page");
  return { default: module.DashboardPage };
});
const LibraryPage = lazy(async () => {
  const module = await import("../pages/library-page");
  return { default: module.LibraryPage };
});
const AnalyticsPage = lazy(async () => {
  const module = await import("../pages/analytics-page");
  return { default: module.AnalyticsPage };
});
const DuplicatesPage = lazy(async () => {
  const module = await import("../pages/duplicates-page");
  return { default: module.DuplicatesPage };
});
const MetadataPage = lazy(async () => {
  const module = await import("../pages/metadata-page");
  return { default: module.MetadataPage };
});
const CleanupPage = lazy(async () => {
  const module = await import("../pages/cleanup-page");
  return { default: module.CleanupPage };
});
const SettingsPage = lazy(async () => {
  const module = await import("../pages/settings-page");
  return { default: module.SettingsPage };
});

function RouteFallback() {
  const { t } = useI18n();
  return (
    <div className="route-loading">
      <div className="loading-screen__orb" />
      <p>{t("app.loading.module")}</p>
    </div>
  );
}

function withSuspense(element: ReactNode) {
  return <Suspense fallback={<RouteFallback />}>{element}</Suspense>;
}

async function refreshCompletedRunData(client: QueryClient, profileId: string | null) {
  await Promise.all([
    client.refetchQueries({
      queryKey: queryKeys.reportAll(profileId),
      exact: true,
      type: "all"
    }),
    client.refetchQueries({
      queryKey: queryKeys.reportFiltered(profileId),
      exact: true,
      type: "all"
    }),
    client.refetchQueries({
      queryKey: queryKeys.metadata(profileId),
      exact: true,
      type: "all"
    }),
    client.refetchQueries({
      queryKey: queryKeys.configState(),
      exact: true,
      type: "all"
    })
  ]);
}

function AppRoot() {
  const configQuery = useConfigState();
  const runQuery = useRunState(configQuery.isSuccess);
  const { locale, setLocale, t } = useI18n();
  const activeProfileId = configQuery.data?.active_profile_id ?? null;
  const run = runQuery.data?.run ?? null;
  const [theme, setTheme] = useStoredState<"aurora" | "cinema">(
    "am.theme",
    "aurora"
  );
  const [dashboardViews, setDashboardViews] = useStoredState<string[]>(
    "am.dashboard.views",
    ["imdb-metacritic", "decision-distribution", "boxplot-library"]
  );
  const [numericFilters, setNumericFilters] = useStoredState<boolean>(
    "am.filters.numeric",
    false
  );
  const [chartThresholds, setChartThresholds] = useStoredState<boolean>(
    "am.charts.thresholds",
    false
  );
  const client = useQueryClient();
  const error = configQuery.error;
  const lastTerminalizedRunIdRef = useRef<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const status = run?.status ?? null;
    const runId = run?.run_id ?? null;
    const isTerminal =
      status === "succeeded" || status === "failed" || status === "cancelled";

    if (!runId || !isTerminal || lastTerminalizedRunIdRef.current === runId) {
      return;
    }

    lastTerminalizedRunIdRef.current = runId;
    const completedProfileId = run?.profile_id ?? null;

    void refreshCompletedRunData(client, completedProfileId);
  }, [client, run?.profile_id, run?.run_id, run?.status]);

  const context: AppOutletContext = {
    config: configQuery.data ?? null,
    run,
    activeProfileId,
    preferences: {
      theme,
      locale,
      dashboardViews,
      numericFilters,
      chartThresholds,
      setTheme,
      setLocale,
      setDashboardViews,
      setNumericFilters,
      setChartThresholds
    },
    refreshConfig: async () => {
      await client.invalidateQueries({ queryKey: queryKeys.configState() });
    },
    refreshRun: async () => {
      await client.invalidateQueries({ queryKey: queryKeys.runState() });
    }
  };

  if (configQuery.isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-screen__orb" />
        <p>{t("app.loading.shell")}</p>
      </div>
    );
  }

  if (error) {
    const message = error instanceof Error ? error.message : t("app.error.unknown");
    return (
      <div className="loading-screen">
        <div className="loading-screen__orb" />
        <p>{t("app.error.load_react")}</p>
        <button
          className="primary-button"
          onClick={() => {
            void client.invalidateQueries({ queryKey: queryKeys.configState() });
          }}
          type="button"
        >
          {t("app.action.retry")}
        </button>
        <p className="loading-screen__error">{message}</p>
      </div>
    );
  }

  return <AppShell context={context} />;
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppRoot />,
    children: [
      { index: true, element: withSuspense(<DashboardPage />) },
      { path: "library", element: withSuspense(<LibraryPage />) },
      { path: "analytics", element: withSuspense(<AnalyticsPage />) },
      { path: "duplicates", element: withSuspense(<DuplicatesPage />) },
      { path: "metadata", element: withSuspense(<MetadataPage />) },
      { path: "cleanup", element: withSuspense(<CleanupPage />) },
      { path: "settings", element: withSuspense(<SettingsPage />) },
      { path: "*", element: <Navigate to="/" replace /> }
    ]
  }
]);

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <RouterProvider router={router} />
      </I18nProvider>
    </QueryClientProvider>
  );
}
