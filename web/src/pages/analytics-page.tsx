import { Suspense, lazy } from "react";
import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "../app/app-context";
import { PageHero } from "../components/page-hero";
import { DASHBOARD_VIEWS } from "../lib/data";
import { buildChartOption } from "../lib/charts";

const ChartCard = lazy(async () => {
  const module = await import("../components/chart-card");
  return { default: module.ChartCard };
});

export function AnalyticsPage() {
  const { reportAll } = useOutletContext<AppOutletContext>();

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Analítica"
        title="Un sistema visual hecho para impresionar y decidir"
        description="Todas las vistas cuantitativas en una retícula editorial pensada para dirección, auditoría y storytelling."
      />

      <div className="chart-grid">
        {DASHBOARD_VIEWS.map((view) => (
          <Suspense key={view.key} fallback={<div className="chart-card-skeleton" />}>
            <ChartCard
              eyebrow="Visualización"
              title={view.label}
              option={buildChartOption(view.key, reportAll)}
              subtitle="Datos calculados en cliente con enfoque de catálogo completo"
            />
          </Suspense>
        ))}
      </div>
    </div>
  );
}
