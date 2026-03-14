import { Film, Gauge, LayoutDashboard, Settings2, Sparkles, Trash2, Wrench } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router-dom";
import { motion } from "framer-motion";

import type { AppOutletContext } from "../app/app-context";
import { setActiveProfile } from "../lib/api";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/library", label: "Biblioteca", icon: Film },
  { to: "/analytics", label: "Analítica", icon: Gauge },
  { to: "/duplicates", label: "Duplicados", icon: Sparkles },
  { to: "/metadata", label: "Metadata", icon: Wrench },
  { to: "/cleanup", label: "Limpieza", icon: Trash2 },
  { to: "/settings", label: "Configuración", icon: Settings2 }
];

interface AppShellProps {
  context: AppOutletContext;
}

export function AppShell({ context }: AppShellProps) {
  const profiles = context.config?.profiles ?? [];
  const activeProfile = profiles.find((profile) => profile.id === context.activeProfileId) ?? null;
  const activeRun = context.config?.run ?? null;
  const switchProfileMutation = useMutation({
    mutationFn: async (profileId: string) => setActiveProfile(profileId),
    onSuccess: () => {
      context.refreshAll();
    }
  });

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <span className="brand-kicker">Analiza Movies</span>
          <h1>Cinematic Intelligence Control Room</h1>
          <p>Plex, DLNA y criterio editorial de primer nivel en una sola superficie.</p>
        </div>

        {profiles.length ? (
          <div className="sidebar-profile">
            <label className="form-field">
              <span>Fuente visible</span>
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
                <option value="">Selecciona un origen</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="sidebar-meta">
              <span>{activeProfile ? `${activeProfile.source_type.toUpperCase()} · ${activeProfile.host || "host pendiente"}` : "Sin origen activo"}</span>
              {activeRun ? (
                <strong className={`status-pill status-pill--${activeRun.status.toLowerCase()}`}>
                  {activeRun.status}
                </strong>
              ) : null}
            </div>
          </div>
        ) : null}

        <nav className="nav-list" aria-label="Secciones principales">
          {NAV_ITEMS.map((item) => {
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
          <p>Diseñado para explorar grandes catálogos sin perder precisión, contexto ni ritmo.</p>
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
