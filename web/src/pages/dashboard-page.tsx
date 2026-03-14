import { Suspense, lazy } from "react";
import { useOutletContext } from "react-router-dom";

import { KpiCard } from "../components/kpi-card";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import type { AppOutletContext } from "../app/app-context";
import { buildChartOption } from "../lib/charts";
import { computeSummary, DASHBOARD_VIEWS, formatCountSize, normalizeDashboardViews } from "../lib/data";

const ChartCard = lazy(async () => {
  const module = await import("../components/chart-card");
  return { default: module.ChartCard };
});

export function DashboardPage() {
  const { reportAll, config, preferences, activeProfileId } = useOutletContext<AppOutletContext>();
  const summary = computeSummary(reportAll);
  const selectedViews = normalizeDashboardViews(preferences.dashboardViews);
  const activeProfile = config?.profiles.find((profile) => profile.id === activeProfileId);

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Sala de mando"
        title="La vista editorial de tu catálogo"
        description="Un dashboard de alta dirección para decidir qué conservar, qué revisar y dónde está el verdadero desperdicio de espacio."
        actions={
          <div className="hero-badge">
            <span>Origen activo</span>
            <strong>{activeProfile ? activeProfile.name : "Default"}</strong>
          </div>
        }
      />

      <section className="kpi-grid">
        <KpiCard label="Catálogo" value={formatCountSize(summary.totalCount, summary.totalSizeGb)} />
        <KpiCard
          label="KEEP"
          tone="keep"
          value={formatCountSize(summary.keepCount, summary.keepSizeGb)}
        />
        <KpiCard
          label="DELETE"
          tone="delete"
          value={formatCountSize(summary.deleteCount, summary.deleteSizeGb)}
        />
        <KpiCard
          label="MAYBE"
          tone="maybe"
          value={formatCountSize(summary.maybeCount, summary.maybeSizeGb)}
        />
        <KpiCard
          label="IMDb medio"
          value={summary.imdbMean ? summary.imdbMean.toFixed(2) : "N/A"}
        />
      </section>

      <SectionCard title="Lectura ejecutiva" eyebrow="Narrativa">
        <div className="insight-grid">
          <article>
            <span>Revisión pendiente</span>
            <strong>{formatCountSize(summary.reviewCount, summary.reviewSizeGb)}</strong>
            <p>DELETE y MAYBE concentran el área con mayor retorno potencial de limpieza.</p>
          </article>
          <article>
            <span>Perfil dominante</span>
            <strong>{summary.keepCount > summary.reviewCount ? "Catálogo sano" : "Catálogo presionando revisión"}</strong>
            <p>La mezcla entre volumen y calidad deja ver si el catálogo sigue una lógica curatorial o solo acumula espacio.</p>
          </article>
          <article>
            <span>Señal crítica</span>
            <strong>{summary.deleteSizeGb > summary.keepSizeGb * 0.2 ? "Oportunidad alta" : "Riesgo contenido"}</strong>
            <p>Si el espacio comprometido en DELETE supera un umbral relevante, conviene ejecutar limpieza guiada.</p>
          </article>
        </div>
      </SectionCard>

      <div className="chart-grid chart-grid--hero">
        {selectedViews.map((viewKey) => {
          const label = DASHBOARD_VIEWS.find((item) => item.key === viewKey)?.label ?? viewKey;
          return (
            <Suspense key={viewKey} fallback={<div className="chart-card-skeleton" />}>
              <ChartCard
                eyebrow="Dashboard"
                title={label}
                option={buildChartOption(viewKey, reportAll)}
              />
            </Suspense>
          );
        })}
      </div>
    </div>
  );
}
