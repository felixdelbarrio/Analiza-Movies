import { Film, Gauge, LayoutDashboard, Settings2, Sparkles, Trash2, Wrench } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router-dom";
import { motion } from "framer-motion";

import type { AppOutletContext } from "../app/app-context";
import { translateRunMessage, translateStage, translateStatus, translateUnit } from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import { setActiveProfile } from "../lib/api";

interface AppShellProps {
  context: AppOutletContext;
}

export function AppShell({ context }: AppShellProps) {
  const { t } = useI18n();
  const profiles = context.config?.profiles ?? [];
  const activeProfile = profiles.find((profile) => profile.id === context.activeProfileId) ?? null;
  const activeRun = context.run;
  const progress = activeRun?.progress ?? null;
  const runPercent = typeof progress?.percent === "number" ? Math.max(0, Math.min(100, progress.percent)) : null;
  const navItems = [
    { to: "/", label: t("nav.dashboard"), icon: LayoutDashboard },
    { to: "/library", label: t("nav.library"), icon: Film },
    { to: "/analytics", label: t("nav.analytics"), icon: Gauge },
    { to: "/duplicates", label: t("nav.duplicates"), icon: Sparkles },
    { to: "/metadata", label: t("nav.metadata"), icon: Wrench },
    { to: "/cleanup", label: t("nav.cleanup"), icon: Trash2 },
    { to: "/settings", label: t("nav.settings"), icon: Settings2 }
  ];
  const switchProfileMutation = useMutation({
    mutationFn: async (profileId: string) => setActiveProfile(profileId),
    onSuccess: async () => {
      await context.refreshConfig();
    }
  });

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <span className="brand-kicker">Analiza Movies</span>
          <h1>{t("brand.heading")}</h1>
          <p>{t("brand.tagline")}</p>
        </div>

        {profiles.length ? (
          <div className="sidebar-profile">
            <label className="form-field">
              <span>{t("sidebar.visible_source")}</span>
              <select
                disabled={switchProfileMutation.isPending}
                onChange={(event) => {
                  const nextProfileId = event.target.value;
                  if (nextProfileId) {
                    switchProfileMutation.mutate(nextProfileId);
                  }
                }}
                value={context.activeProfileId ?? ""}
              >
                <option value="">{t("sidebar.select_source")}</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="sidebar-meta">
              <span>
                {activeProfile
                  ? `${activeProfile.source_type.toUpperCase()} · ${activeProfile.host || t("sidebar.host_pending")}`
                  : t("sidebar.no_active_source")}
              </span>
              {activeRun ? (
                <strong className={`status-pill status-pill--${activeRun.status.toLowerCase()}`}>
                  {translateStatus(activeRun.status, t)}
                </strong>
              ) : null}
            </div>

            {activeRun ? (
              <div className="run-spotlight">
                <div className="run-spotlight__header">
                  <strong>{activeRun.profile_name}</strong>
                  <span>{translateStage(progress?.stage || activeRun.status, t)}</span>
                </div>
                <p>{translateRunMessage(progress, t) || t("sidebar.run_active")}</p>
                {runPercent !== null ? (
                  <div className="progress-meter" aria-label={t("settings.run.progress_aria")}>
                    <span style={{ width: `${runPercent}%` }} />
                  </div>
                ) : null}
                <div className="run-spotlight__meta">
                  <span>
                    {typeof progress?.current === "number"
                      ? `${progress.current}${progress?.total ? ` / ${progress.total}` : ""} ${translateUnit(progress?.unit, t)}`.trim()
                      : translateStage(progress?.stage || activeRun.status, t)}
                  </span>
                  <NavLink className="inline-link inline-link--subtle" to="/settings">
                    {t("sidebar.run_monitor")}
                  </NavLink>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        <nav className="nav-list" aria-label={t("sidebar.main_sections")}>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
                to={item.to}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <p>{t("sidebar.footer")}</p>
        </div>
      </aside>

      <main className="content-shell">
        <motion.div
          className="page-shell"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
        >
          <Outlet context={context} />
        </motion.div>
      </main>
    </div>
  );
}
