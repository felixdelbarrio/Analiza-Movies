import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowUpRight, Radar, RefreshCcw, Rocket, Square } from "lucide-react";

import { useAppContext } from "../app/use-app-context";
import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";
import { useRunLogs } from "../hooks/use-dashboard-data";
import { SUPPORTED_LOCALES } from "../i18n/catalog";
import {
  translateRunMessage,
  translateStage,
  translateStatus,
  translateUnit
} from "../i18n/helpers";
import { useI18n } from "../i18n/provider";
import {
  discoverDlna,
  discoverPlex,
  pollPlexAuth,
  saveProfile,
  setActiveProfile,
  startAnalysis,
  startPlexAuth,
  stopAnalysis,
  updateConfigState
} from "../lib/api";
import { getDashboardViews, normalizeDashboardViews } from "../lib/data";
import { isDesktopShell, openInAppContainer } from "../lib/desktop";
import type { ServerDiscovery } from "../lib/types";

export function SettingsPage() {
  const { locale, formatTime, t } = useI18n();
  const { config, preferences, refreshConfig, refreshRun, activeProfileId, run } = useAppContext();
  const [sourceKind, setSourceKind] = useState<"plex" | "dlna">("plex");
  const [omdbValue, setOmdbValue] = useState(config?.omdb_api_keys ?? "");
  const [plexSessionId, setPlexSessionId] = useState("");
  const [plexAuthUrl, setPlexAuthUrl] = useState("");
  const [discovery, setDiscovery] = useState<ServerDiscovery[]>([]);
  const desktopShell = isDesktopShell();
  const isRunActive = run?.status === "running" || run?.status === "stopping";
  const runLogsQuery = useRunLogs(Boolean(run), 120);
  const dashboardViews = useMemo(() => getDashboardViews(t), [t]);
  const recentEvents = useMemo(
    () => [...(run?.progress?.recent ?? [])].reverse().slice(0, 6),
    [run?.progress?.recent]
  );
  const runPercent =
    typeof run?.progress?.percent === "number"
      ? Math.max(0, Math.min(100, run.progress?.percent ?? 0))
      : null;

  useEffect(() => {
    setOmdbValue(config?.omdb_api_keys ?? "");
  }, [config?.omdb_api_keys]);

  const omdbMutation = useMutation({
    mutationFn: async () => updateConfigState({ omdb_api_keys: omdbValue }),
    onSuccess: refreshConfig
  });

  const startPlexMutation = useMutation({
    mutationFn: async () => startPlexAuth(!desktopShell),
    onSuccess: async (payload) => {
      setPlexSessionId(payload.session_id);
      setPlexAuthUrl(payload.auth_url);
      if (desktopShell && payload.auth_url) {
        await openInAppContainer(payload.auth_url, t("settings.plex.login_title"));
      }
    }
  });

  const pollPlexMutation = useMutation({
    mutationFn: async () => {
      if (!plexSessionId) {
        throw new Error(t("settings.plex.no_session"));
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
    onSuccess: refreshConfig
  });

  const activateMutation = useMutation({
    mutationFn: async (profileId: string) => setActiveProfile(profileId),
    onSuccess: refreshConfig
  });

  const runMutation = useMutation({
    mutationFn: async (profileId: string) => startAnalysis(profileId),
    onSuccess: async () => {
      await refreshRun();
    }
  });

  const stopRunMutation = useMutation({
    mutationFn: stopAnalysis,
    onSuccess: async () => {
      await refreshRun();
    }
  });

  const profiles = config?.profiles ?? [];

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("nav.settings")}
        title={t("settings.hero.title")}
        description={t("settings.hero.description")}
      />

      <div className="settings-grid">
        <SectionCard title={t("settings.infrastructure.title")} eyebrow={t("settings.infrastructure.eyebrow")}>
          <div className="settings-stack">
            <label className="form-field">
              <span>{t("settings.omdb.label")}</span>
              <input
                onChange={(event) => setOmdbValue(event.target.value)}
                placeholder={t("settings.omdb.placeholder")}
                type="password"
                value={omdbValue}
              />
            </label>
            <p className="section-card__eyebrow">{t("settings.omdb.help")}</p>
            <div className="inline-actions">
              <button className="primary-button" onClick={() => omdbMutation.mutate()} type="button">
                {t("settings.omdb.save")}
              </button>
              <button
                className="secondary-button"
                onClick={() => void openInAppContainer("https://www.omdbapi.com/apikey.aspx", t("settings.omdb.label"))}
                type="button"
              >
                {t("settings.omdb.generate")} <ArrowUpRight size={14} />
              </button>
            </div>
          </div>
        </SectionCard>

        <SectionCard title={t("settings.appearance.title")} eyebrow={t("settings.appearance.eyebrow")}>
          <div className="settings-stack">
            <label className="form-field">
              <span>{t("locale.label")}</span>
              <select
                onChange={(event) => preferences.setLocale(event.target.value as typeof preferences.locale)}
                value={preferences.locale}
              >
                {SUPPORTED_LOCALES.map((value) => (
                  <option key={value} value={value}>
                    {t(`locale.${value}`)}
                  </option>
                ))}
              </select>
            </label>

            <label className="form-field">
              <span>{t("settings.theme.label")}</span>
              <select
                onChange={(event) => preferences.setTheme(event.target.value as "aurora" | "cinema")}
                value={preferences.theme}
              >
                <option value="aurora">{t("theme.aurora")}</option>
                <option value="cinema">{t("theme.cinema")}</option>
              </select>
            </label>

            <label className="form-field">
              <span>{t("settings.dashboard_views.label")}</span>
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
                {dashboardViews.map((view) => (
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
              <span>{t("settings.numeric_filters")}</span>
            </label>

            <label className="toggle-pill">
              <input
                checked={preferences.chartThresholds}
                onChange={(event) => preferences.setChartThresholds(event.target.checked)}
                type="checkbox"
              />
              <span>{t("settings.chart_thresholds")}</span>
            </label>
          </div>
        </SectionCard>
      </div>

      <SectionCard title={t("settings.profiles.title")} eyebrow={t("settings.profiles.eyebrow")}>
        <div className="settings-stack">
          <div className="inline-actions">
            <label className="form-field form-field--compact">
              <span>{t("settings.source_kind.label")}</span>
              <select
                onChange={(event) => setSourceKind(event.target.value as "plex" | "dlna")}
                value={sourceKind}
              >
                <option value="plex">{t("source.plex")}</option>
                <option value="dlna">{t("source.dlna")}</option>
              </select>
            </label>
            <button className="secondary-button" onClick={() => discoverMutation.mutate()} type="button">
              <Radar size={14} /> {t("settings.action.search_network")}
            </button>
            {sourceKind === "plex" ? (
              <>
                <button
                  className="secondary-button"
                  onClick={() => startPlexMutation.mutate()}
                  type="button"
                >
                  <Rocket size={14} /> {t("settings.action.link_plex")}
                </button>
                <button
                  className="secondary-button"
                  onClick={() => pollPlexMutation.mutate()}
                  type="button"
                >
                  <RefreshCcw size={14} /> {t("settings.action.check_link")}
                </button>
              </>
            ) : null}
          </div>

          {plexAuthUrl ? (
            <button
              className="inline-link"
              onClick={() => void openInAppContainer(plexAuthUrl, t("settings.plex.login_title"))}
              type="button"
            >
              {t("settings.action.open_plex_web")} <ArrowUpRight size={14} />
            </button>
          ) : null}

          <div className="profile-grid">
            {discovery.map((server) => (
              <article key={`${server.source_type}-${server.host}-${server.port}-${server.name}`} className="profile-card">
                <span>{server.source_type === "plex" ? t("source.plex") : t("source.dlna")}</span>
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
                    {t("settings.action.save_profile")}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard title={t("settings.operation.title")} eyebrow={t("settings.operation.eyebrow")}>
        <div className="settings-stack">
          <div className="run-console">
            <div className="run-console__header">
              <div>
                <span className="section-card__eyebrow">{t("settings.run.heading")}</span>
                <h3>{run ? `${run.profile_name} · ${run.source_type.toUpperCase()}` : t("settings.run.none")}</h3>
              </div>
              {run ? (
                <strong className={`status-pill status-pill--${run.status.toLowerCase()}`}>{translateStatus(run.status, t)}</strong>
              ) : null}
            </div>

            <p className="run-console__message">
              {translateRunMessage(run?.progress, t) || t("settings.run.default_message")}
            </p>

            {runPercent !== null ? (
              <div className="progress-meter progress-meter--lg" aria-label={t("settings.run.progress_aria")}>
                <span style={{ width: `${runPercent}%` }} />
              </div>
            ) : null}

            {run ? (
              <div className="run-metrics">
                <article>
                  <span>{t("settings.run.phase")}</span>
                  <strong>{translateStage(run.progress?.stage || run.status, t)}</strong>
                </article>
                <article>
                  <span>{t("settings.run.scope")}</span>
                  <strong>{run.progress?.scope || run.profile_name}</strong>
                </article>
                <article>
                  <span>{t("settings.run.progress")}</span>
                  <strong>
                    {typeof run.progress?.current === "number"
                      ? `${run.progress.current}${run.progress?.total ? ` / ${run.progress.total}` : ""} ${translateUnit(run.progress?.unit, t)}`.trim()
                      : t("app.not_available")}
                  </strong>
                </article>
                <article>
                  <span>{t("settings.run.errors")}</span>
                  <strong>{run.progress?.errors ?? 0}</strong>
                </article>
              </div>
            ) : null}

            {recentEvents.length ? (
              <div className="run-events">
                {recentEvents.map((event, index) => (
                  <article key={`${event.at}-${index}`} className="run-event">
                    <span>{translateStage(event.stage, t)}</span>
                    <strong>{translateRunMessage(event, t)}</strong>
                    <small>{formatTime(event.at)}</small>
                  </article>
                ))}
              </div>
            ) : null}

            {runLogsQuery.data?.lines?.length ? (
              <div className="log-stack">
                <pre>{runLogsQuery.data.lines.join("\n")}</pre>
              </div>
            ) : null}

            {run ? (
              <div className="inline-actions">
                {isRunActive ? (
                  <button
                    className="secondary-button"
                    disabled={stopRunMutation.isPending}
                  onClick={() => stopRunMutation.mutate()}
                  type="button"
                >
                  <Square size={14} /> {t("settings.run.stop")}
                </button>
                ) : (
                  <button className="secondary-button" onClick={() => refreshRun()} type="button">
                    <RefreshCcw size={14} /> {t("settings.run.refresh")}
                  </button>
                )}
              </div>
            ) : null}
          </div>

        <div className="profile-grid">
          {profiles.map((profile) => (
            <article key={profile.id} className={`profile-card${profile.id === activeProfileId ? " active" : ""}`}>
              <span>{profile.source_type === "plex" ? t("source.plex") : t("source.dlna")}</span>
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
                  {t("settings.profile.show")}
                </button>
                <button
                  className="primary-button"
                  disabled={isRunActive}
                  onClick={() => runMutation.mutate(profile.id)}
                  type="button"
                >
                  {isRunActive && run?.profile_id === profile.id ? t("settings.profile.analyzing") : t("settings.profile.analyze")}
                </button>
              </div>
            </article>
          ))}
        </div>
        </div>
      </SectionCard>
    </div>
  );
}
