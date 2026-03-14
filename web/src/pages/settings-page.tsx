import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Radar, RefreshCcw, Rocket } from "lucide-react";
import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "../app/app-context";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import {
  discoverDlna,
  discoverPlex,
  pollPlexAuth,
  saveProfile,
  setActiveProfile,
  startAnalysis,
  startPlexAuth,
  updateConfigState
} from "../lib/api";
import { DASHBOARD_VIEWS, normalizeDashboardViews } from "../lib/data";
import type { Profile, ServerDiscovery } from "../lib/types";

export function SettingsPage() {
  const { config, preferences, refreshAll, activeProfileId } = useOutletContext<AppOutletContext>();
  const queryClient = useQueryClient();
  const [sourceKind, setSourceKind] = useState<"plex" | "dlna">("plex");
  const [omdbValue, setOmdbValue] = useState(config?.omdb_api_keys ?? "");
  const [plexSessionId, setPlexSessionId] = useState("");
  const [plexAuthUrl, setPlexAuthUrl] = useState("");
  const [discovery, setDiscovery] = useState<ServerDiscovery[]>([]);

  useEffect(() => {
    setOmdbValue(config?.omdb_api_keys ?? "");
  }, [config?.omdb_api_keys]);

  const invalidate = async () => {
    await queryClient.invalidateQueries();
    refreshAll();
  };

  const omdbMutation = useMutation({
    mutationFn: async () => updateConfigState({ omdb_api_keys: omdbValue }),
    onSuccess: invalidate
  });

  const startPlexMutation = useMutation({
    mutationFn: startPlexAuth,
    onSuccess: (payload) => {
      setPlexSessionId(payload.session_id);
      setPlexAuthUrl(payload.auth_url);
    }
  });

  const pollPlexMutation = useMutation({
    mutationFn: async () => {
      if (!plexSessionId) {
        throw new Error("No hay sesión Plex activa.");
      }
      return pollPlexAuth(plexSessionId);
    },
    onSuccess: (payload) => {
      if (payload.servers) {
        setDiscovery(payload.servers);
      }
    }
  });

  const discoverMutation = useMutation({
    mutationFn: async () =>
      sourceKind === "plex" ? discoverPlex(plexSessionId || undefined) : discoverDlna(),
    onSuccess: (payload) => {
      if ("servers" in payload) {
        setDiscovery(payload.servers);
      } else {
        setDiscovery(payload.devices);
      }
    }
  });

  const saveProfileMutation = useMutation({
    mutationFn: async (server: ServerDiscovery) =>
      saveProfile({ profile: server, set_active: true }),
    onSuccess: invalidate
  });

  const activateMutation = useMutation({
    mutationFn: async (profileId: string) => setActiveProfile(profileId),
    onSuccess: invalidate
  });

  const runMutation = useMutation({
    mutationFn: async (profileId: string) => startAnalysis(profileId),
    onSuccess: invalidate
  });

  const profiles = config?.profiles ?? [];

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="Configuración"
        title="Control total sobre origen, estética y operación"
        description="Todo lo operativo vive aquí: perfiles Plex/DLNA, OMDb, preferencias visuales y lanzamiento de análisis."
      />

      <div className="settings-grid">
        <SectionCard title="Infraestructura" eyebrow="Conectividad">
          <div className="settings-stack">
            <label className="form-field">
              <span>OMDb API Key</span>
              <input
                onChange={(event) => setOmdbValue(event.target.value)}
                placeholder="Tu clave OMDb"
                type="password"
                value={omdbValue}
              />
            </label>
            <div className="inline-actions">
              <button className="primary-button" onClick={() => omdbMutation.mutate()} type="button">
                Guardar OMDb
              </button>
              <a
                className="secondary-button"
                href="https://www.omdbapi.com/apikey.aspx"
                rel="noreferrer"
                target="_blank"
              >
                Generar clave <ArrowUpRight size={14} />
              </a>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Apariencia" eyebrow="Experience layer">
          <div className="settings-stack">
            <label className="form-field">
              <span>Tema</span>
              <select
                onChange={(event) => preferences.setTheme(event.target.value as "aurora" | "cinema")}
                value={preferences.theme}
              >
                <option value="aurora">Aurora Deck</option>
                <option value="cinema">Cinema Ivory</option>
              </select>
            </label>

            <label className="form-field">
              <span>Gráficos del dashboard (máximo 3)</span>
              <select
                multiple
                onChange={(event) =>
                  preferences.setDashboardViews(
                    normalizeDashboardViews(
                      Array.from(event.target.selectedOptions).map((option) => option.value)
                    )
                  )
                }
                value={normalizeDashboardViews(preferences.dashboardViews)}
              >
                {DASHBOARD_VIEWS.map((view) => (
                  <option key={view.key} value={view.key}>
                    {view.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="toggle-pill">
              <input
                checked={preferences.numericFilters}
                onChange={(event) => preferences.setNumericFilters(event.target.checked)}
                type="checkbox"
              />
              <span>Mostrar filtros numéricos avanzados</span>
            </label>

            <label className="toggle-pill">
              <input
                checked={preferences.chartThresholds}
                onChange={(event) => preferences.setChartThresholds(event.target.checked)}
                type="checkbox"
              />
              <span>Mostrar umbrales de charts</span>
            </label>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Perfiles de origen" eyebrow="Plex / DLNA">
        <div className="settings-stack">
          <div className="inline-actions">
            <label className="form-field form-field--compact">
              <span>Tipo de origen</span>
              <select
                onChange={(event) => setSourceKind(event.target.value as "plex" | "dlna")}
                value={sourceKind}
              >
                <option value="plex">Plex</option>
                <option value="dlna">DLNA</option>
              </select>
            </label>
            <button className="secondary-button" onClick={() => discoverMutation.mutate()} type="button">
              <Radar size={14} /> Buscar en la red
            </button>
            {sourceKind === "plex" ? (
              <>
                <button
                  className="secondary-button"
                  onClick={() => startPlexMutation.mutate()}
                  type="button"
                >
                  <Rocket size={14} /> Vincular Plex
                </button>
                <button
                  className="secondary-button"
                  onClick={() => pollPlexMutation.mutate()}
                  type="button"
                >
                  <RefreshCcw size={14} /> Comprobar vinculación
                </button>
              </>
            ) : null}
          </div>

          {plexAuthUrl ? (
            <a className="inline-link" href={plexAuthUrl} rel="noreferrer" target="_blank">
              Abrir Plex Web y conceder acceso <ArrowUpRight size={14} />
            </a>
          ) : null}

          <div className="profile-grid">
            {discovery.map((server) => (
              <article key={`${server.source_type}-${server.host}-${server.port}-${server.name}`} className="profile-card">
                <span>{server.source_type === "plex" ? "Plex" : "DLNA"}</span>
                <strong>{server.name}</strong>
                <p>
                  {server.host}:{server.port}
                </p>
                <div className="inline-actions">
                  <button
                    className="primary-button"
                    onClick={() => saveProfileMutation.mutate(server)}
                    type="button"
                  >
                    Guardar perfil
                  </button>
                </div>
              </article>
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Origen activo" eyebrow="Operación">
        <div className="profile-grid">
          {profiles.map((profile) => (
            <article key={profile.id} className={`profile-card${profile.id === activeProfileId ? " active" : ""}`}>
              <span>{profile.source_type === "plex" ? "Plex" : "DLNA"}</span>
              <strong>{profile.name}</strong>
              <p>
                {profile.host}:{profile.port}
              </p>
              <div className="inline-actions">
                <button
                  className="secondary-button"
                  disabled={profile.id === activeProfileId}
                  onClick={() => activateMutation.mutate(profile.id)}
                  type="button"
                >
                  Ver este origen
                </button>
                <button
                  className="primary-button"
                  onClick={() => runMutation.mutate(profile.id)}
                  type="button"
                >
                  Analizar ahora
                </button>
              </div>
            </article>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
