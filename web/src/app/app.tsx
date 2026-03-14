import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { Navigate, RouterProvider, createBrowserRouter } from "react-router-dom";

import { useConfigState, useMetadata, useReportAll, useReportFiltered } from "../hooks/use-dashboard-data";
import { useStoredState } from "../lib/preferences";
import { AppShell } from "../components/app-shell";
import { DashboardPage } from "../pages/dashboard-page";
import { LibraryPage } from "../pages/library-page";
import { AnalyticsPage } from "../pages/analytics-page";
import { DuplicatesPage } from "../pages/duplicates-page";
import { MetadataPage } from "../pages/metadata-page";
import { CleanupPage } from "../pages/cleanup-page";
import { SettingsPage } from "../pages/settings-page";
import type { AppOutletContext } from "./app-context";

const queryClient = new QueryClient();

function AppRoot() {
  const configQuery = useConfigState();
  const activeProfileId = configQuery.data?.active_profile_id ?? null;
  const allQuery = useReportAll(activeProfileId);
  const filteredQuery = useReportFiltered(activeProfileId);
  const metadataQuery = useMetadata(activeProfileId);
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
  const error =
    configQuery.error ?? allQuery.error ?? filteredQuery.error ?? metadataQuery.error;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const context: AppOutletContext = {
    config: configQuery.data ?? null,
    activeProfileId,
    reportAll: allQuery.data ?? [],
    reportFiltered: filteredQuery.data ?? [],
    metadataRows: metadataQuery.data ?? [],
    preferences: {
      theme,
      dashboardViews,
      numericFilters,
      chartThresholds,
      setTheme,
      setDashboardViews,
      setNumericFilters,
      setChartThresholds
    },
    isLoading:
      configQuery.isLoading ||
      allQuery.isLoading ||
      filteredQuery.isLoading ||
      metadataQuery.isLoading,
    refreshAll: () => {
      void client.invalidateQueries();
    }
  };

  if (context.isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-screen__orb" />
        <p>Cargando la sala de mando cinematográfica…</p>
      </div>
    );
  }

  if (error) {
    const message = error instanceof Error ? error.message : "Error desconocido";
    return (
      <div className="loading-screen">
        <div className="loading-screen__orb" />
        <p>No se pudo cargar la experiencia React.</p>
        <button
          className="primary-button"
          onClick={() => {
            void client.invalidateQueries();
          }}
          type="button"
        >
          Reintentar
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
      { index: true, element: <DashboardPage /> },
      { path: "library", element: <LibraryPage /> },
      { path: "analytics", element: <AnalyticsPage /> },
      { path: "duplicates", element: <DuplicatesPage /> },
      { path: "metadata", element: <MetadataPage /> },
      { path: "cleanup", element: <CleanupPage /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "*", element: <Navigate to="/" replace /> }
    ]
  }
]);

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
