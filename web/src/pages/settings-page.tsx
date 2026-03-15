import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  ArrowUpRight,
  Check,
  Heart,
  Link2,
  Radar,
  RefreshCcw,
  Rocket,
  Square
} from "lucide-react";

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
  startAnalysis,
  startPlexAuth,
  stopAnalysis,
  updateConfigState
} from "../lib/api";
import { getDashboardViews, normalizeDashboardViews } from "../lib/data";
import { isDesktopShell, openInAppContainer } from "../lib/desktop";
import { useStoredState } from "../lib/preferences";
import type { DashboardViewKey, Profile, ServerDiscovery } from "../lib/types";

type SourceKind = "plex" | "dlna";
type PlexLinkState = "idle" | "pending" | "complete" | "error";
const SUPPORT_URL = "https://paypal.me/felixdelbarrio";

function describeEndpoint(host?: string | null, port?: number | null) {
  const cleanHost = String(host || "").trim();
  if (!cleanHost) {
    return "";
  }
  return port ? `${cleanHost}:${port}` : cleanHost;
}

function matchesServer(profile: Profile, server: ServerDiscovery) {
  const profileMachine = String(profile.machine_identifier || "").trim();
  const serverMachine = String(server.machine_identifier || "").trim();
  if (profileMachine && serverMachine && profileMachine === serverMachine) {
    return true;
  }

  const profileDevice = String(profile.device_id || "").trim();
  const serverDevice = String(server.device_id || "").trim();
  if (profileDevice && serverDevice && profileDevice === serverDevice) {
    return true;
  }

  return (
    String(profile.host || "").trim() === String(server.host || "").trim() &&
    Number(profile.port || 0) === Number(server.port || 0)
  );
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error && error.message.trim() ? error.message : fallback;
}

export function SettingsPage() {
  const { locale, formatTime, t } = useI18n();
  const { config, preferences, refreshConfig, refreshRun, activeProfileId, run } = useAppContext();
  const [sourceKind, setSourceKind] = useState<SourceKind>("plex");
  const [omdbValue, setOmdbValue] = useState("");
  const [plexSessionId, setPlexSessionId] = useState("");
  const [plexAuthUrl, setPlexAuthUrl] = useState("");
  const [plexLinkState, setPlexLinkState] = useState<PlexLinkState>("idle");
  const [discovery, setDiscovery] = useState<ServerDiscovery[]>([]);
  const [hasSupportedProject, setHasSupportedProject] = useStoredState<boolean>(
    "am.settings.support.has_contributed",
    false
  );
  const desktopShell = isDesktopShell();
  const isRunActive = run?.status === "running" || run?.status === "stopping";
  const runLogsQuery = useRunLogs(Boolean(run), 120);
  const dashboardViews = useMemo(() => getDashboardViews(t), [t]);
  const selectedDashboardViews = useMemo(
    () => normalizeDashboardViews(preferences.dashboardViews),
    [preferences.dashboardViews]
  );
  const recentEvents = useMemo(
    () => [...(run?.progress?.recent ?? [])].reverse().slice(0, 6),
    [run?.progress?.recent]
  );
  const runPercent =
    typeof run?.progress?.percent === "number"
      ? Math.max(0, Math.min(100, run.progress?.percent ?? 0))
      : null;
  const profiles = config?.profiles ?? [];
  const activeProfile = profiles.find((profile) => profile.id === activeProfileId) ?? null;
  const profileCount = profiles.length;
  const omdbReady = Boolean(config?.has_omdb_api_keys);
  const savedProfiles = useMemo(
    () =>
      [...profiles].sort((left, right) => {
        if (left.id === activeProfileId) {
          return -1;
        }
        if (right.id === activeProfileId) {
          return 1;
        }
        return left.name.localeCompare(right.name, locale);
      }),
    [activeProfileId, locale, profiles]
  );
  const discoveredServers = useMemo(
    () =>
      [...discovery].sort((left, right) => {
        if (Boolean(left.local) !== Boolean(right.local)) {
          return left.local ? -1 : 1;
        }
        return String(left.name || "").localeCompare(String(right.name || ""), locale);
      }),
    [discovery, locale]
  );
  const supportBundles = useMemo(
    () => [
      {
        key: "fair",
        amount: t("settings.support.bundle.entry.value"),
        label: t("settings.support.bundle.entry.label"),
        copy: t("settings.support.bundle.entry.copy")
      },
      {
        key: "useful",
        amount: t("settings.support.bundle.core.value"),
        label: t("settings.support.bundle.core.label"),
        copy: t("settings.support.bundle.core.copy")
      },
      {
        key: "patron",
        amount: t("settings.support.bundle.patron.value"),
        label: t("settings.support.bundle.patron.label"),
        copy: t("settings.support.bundle.patron.copy")
      }
    ],
    [t]
  );

  useEffect(() => {
    if (!omdbReady) {
      setOmdbValue("");
    }
  }, [omdbReady]);

  const omdbMutation = useMutation({
    mutationFn: async () => updateConfigState({ omdb_api_keys: omdbValue }),
    onSuccess: refreshConfig
  });

  const startPlexMutation = useMutation({
    mutationFn: async () => startPlexAuth(!desktopShell),
    onMutate: () => {
      setPlexLinkState("pending");
      setPlexSessionId("");
      setPlexAuthUrl("");
      setDiscovery([]);
    },
    onSuccess: async (payload) => {
      setPlexSessionId(payload.session_id);
      setPlexAuthUrl(payload.auth_url);
      setPlexLinkState("pending");
      if (desktopShell && payload.auth_url) {
        await openInAppContainer(payload.auth_url, t("settings.plex.login_title"));
      }
    },
    onError: () => {
      setPlexLinkState("error");
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
      if (payload.status === "complete") {
        setPlexLinkState("complete");
        setDiscovery(payload.servers ?? []);
        return;
      }
      setPlexLinkState("pending");
    },
    onError: () => {
      setPlexLinkState("error");
    }
  });

  useEffect(() => {
    if (sourceKind !== "plex" || !plexSessionId || plexLinkState !== "pending") {
      return;
    }

    const interval = window.setInterval(() => {
      if (!pollPlexMutation.isPending) {
        pollPlexMutation.mutate();
      }
    }, 2500);

    return () => window.clearInterval(interval);
  }, [plexLinkState, plexSessionId, pollPlexMutation, sourceKind]);

  const discoverMutation = useMutation({
    mutationFn: async () =>
      sourceKind === "plex" ? discoverPlex(plexSessionId || undefined) : discoverDlna(),
    onSuccess: (payload) => {
      if ("servers" in payload) {
        setDiscovery(payload.servers);
        if (payload.auth_complete) {
          setPlexLinkState("complete");
        }
        return;
      }
      setDiscovery(payload.devices);
    }
  });

  const saveProfileMutation = useMutation({
    mutationFn: async (server: ServerDiscovery) =>
      saveProfile({ profile: server, set_active: true }),
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

  function toggleDashboardView(viewKey: DashboardViewKey) {
    if (selectedDashboardViews.includes(viewKey)) {
      if (selectedDashboardViews.length === 1) {
        return;
      }
      preferences.setDashboardViews(
        selectedDashboardViews.filter((item) => item !== viewKey)
      );
      return;
    }
    if (selectedDashboardViews.length >= 3) {
      return;
    }
    preferences.setDashboardViews([...selectedDashboardViews, viewKey]);
  }

  function renderDiscoveryAction(server: ServerDiscovery) {
    const registeredProfile =
      profiles.find((profile) => matchesServer(profile, server)) ?? null;
    const canSave = server.source_type === "dlna" || plexLinkState === "complete";
    const isActive = registeredProfile?.id === activeProfileId;

    if (registeredProfile && isActive) {
      return (
        <button className="secondary-button" disabled type="button">
          <Check size={14} /> {t("settings.profile.active")}
        </button>
      );
    }

    return (
      <button
        className={registeredProfile ? "secondary-button" : "primary-button"}
        disabled={!canSave || saveProfileMutation.isPending}
        onClick={() => saveProfileMutation.mutate(server)}
        type="button"
      >
        {registeredProfile ? t("settings.profile.show") : t("settings.action.save_profile")}
      </button>
    );
  }

  const plexStatusLabel =
    plexLinkState === "complete"
      ? t("settings.plex.status.connected")
      : plexLinkState === "pending"
        ? t("settings.plex.status.pending")
        : plexLinkState === "error"
          ? t("settings.plex.status.error")
          : t("settings.plex.status.idle");
  const plexStatusCopy =
    plexLinkState === "complete"
      ? t("settings.plex.status.connected_copy")
      : plexLinkState === "pending"
        ? t("settings.plex.status.pending_copy")
        : plexLinkState === "error"
          ? errorMessage(startPlexMutation.error ?? pollPlexMutation.error, t("settings.plex.status.error_copy"))
          : t("settings.plex.status.idle_copy");

  return (
    <div className="page-stack">
      <PageHero
        eyebrow={t("nav.settings")}
        title={t("settings.hero.title")}
        description={t("settings.hero.description")}
      />

      <div className="settings-overview-grid">
        <article className="library-glance">
          <span>{t("settings.overview.active_source")}</span>
          <strong>{activeProfile?.name ?? t("sidebar.no_active_source")}</strong>
        </article>
        <article className="library-glance">
          <span>{t("settings.overview.saved_sources")}</span>
          <strong>{profileCount.toLocaleString(locale)}</strong>
        </article>
        <article className="library-glance">
          <span>{t("settings.overview.dashboard")}</span>
          <strong>{selectedDashboardViews.length.toLocaleString(locale)} / 3</strong>
        </article>
        <article className="library-glance">
          <span>{t("settings.overview.omdb")}</span>
          <strong>{omdbReady ? t("settings.state.ready") : t("settings.state.pending")}</strong>
        </article>
      </div>

      <SectionCard
        title={t("settings.appearance.title")}
        eyebrow={t("settings.appearance.eyebrow")}
        actions={
          <span className="settings-state-pill">
            {selectedDashboardViews.length.toLocaleString(locale)} / 3
          </span>
        }
      >
        <div className="settings-preferences-grid">
          <div className="settings-preference-stack">
            <article className="settings-preference-card settings-preference-card--accent">
              <div className="settings-preference-card__header">
                <span>{t("settings.theme.label")}</span>
                <strong>
                  {preferences.theme === "aurora"
                    ? t("theme.aurora")
                    : t("theme.cinema")}
                </strong>
              </div>
              <label className="form-field">
                <select
                  aria-label={t("settings.theme.label")}
                  onChange={(event) => preferences.setTheme(event.target.value as "aurora" | "cinema")}
                  value={preferences.theme}
                >
                  <option value="aurora">{t("theme.aurora")}</option>
                  <option value="cinema">{t("theme.cinema")}</option>
                </select>
              </label>
            </article>

            <article className="settings-preference-card">
              <div className="settings-preference-card__header">
                <span>{t("locale.label")}</span>
                <strong>{t(`locale.${preferences.locale}`)}</strong>
              </div>
              <label className="form-field">
                <select
                  aria-label={t("locale.label")}
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
            </article>

            <div className="settings-option-grid">
              <label
                className={`settings-option-card${
                  preferences.numericFilters ? " is-active" : ""
                }`}
              >
                <div className="settings-option-card__copy">
                  <strong>{t("settings.numeric_filters")}</strong>
                </div>
                <input
                  checked={preferences.numericFilters}
                  onChange={(event) => preferences.setNumericFilters(event.target.checked)}
                  type="checkbox"
                />
              </label>

              <label
                className={`settings-option-card${
                  preferences.chartThresholds ? " is-active" : ""
                }`}
              >
                <div className="settings-option-card__copy">
                  <strong>{t("settings.chart_thresholds")}</strong>
                </div>
                <input
                  checked={preferences.chartThresholds}
                  onChange={(event) => preferences.setChartThresholds(event.target.checked)}
                  type="checkbox"
                />
              </label>
            </div>
          </div>

          <div className="settings-dashboard-curation">
            <div className="settings-section-copy">
              <span>{t("settings.dashboard_views.label")}</span>
              <p>{t("settings.dashboard_views.help")}</p>
            </div>
            <div className="dashboard-chip-grid">
              {dashboardViews.map((view) => {
                const active = selectedDashboardViews.includes(view.key);
                const disabled = !active && selectedDashboardViews.length >= 3;
                return (
                  <button
                    key={view.key}
                    className={`dashboard-chip${active ? " is-active" : ""}`}
                    disabled={disabled}
                    onClick={() => toggleDashboardView(view.key)}
                    type="button"
                  >
                    <span>{view.label}</span>
                    {active ? <Check size={14} /> : null}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title={t("settings.sources.title")}
        eyebrow={t("settings.sources.eyebrow")}
        actions={
          <span className="settings-state-pill">
            {profileCount.toLocaleString(locale)} {t("unit.sources")}
          </span>
        }
      >
        <div className="source-control-layout">
          <div className="source-registry">
            <div className="settings-section-copy">
              <span>{t("settings.sources.saved_title")}</span>
              <p>{t("settings.sources.saved_copy")}</p>
            </div>

            <div className="profile-grid">
              {savedProfiles.length ? (
                savedProfiles.map((profile) => (
                  <article
                    key={profile.id}
                    className={`profile-card settings-profile-card${
                      profile.id === activeProfileId ? " active" : ""
                    }`}
                  >
                    <div className="settings-profile-card__head">
                      <span>{profile.source_type === "plex" ? t("source.plex") : t("source.dlna")}</span>
                      {profile.id === activeProfileId ? (
                        <span className="settings-state-pill settings-state-pill--accent">
                          {t("settings.profile.active")}
                        </span>
                      ) : null}
                    </div>
                    <strong>{profile.name}</strong>
                    <p>{describeEndpoint(profile.host, profile.port) || t("sidebar.host_pending")}</p>
                    <small>{profile.base_url || profile.location || t("app.not_available")}</small>
                    <div className="inline-actions">
                      <button
                        className="secondary-button"
                        disabled={profile.id === activeProfileId}
                        onClick={() => saveProfileMutation.mutate(profile)}
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
                        {isRunActive && run?.profile_id === profile.id
                          ? t("settings.profile.analyzing")
                          : t("settings.profile.analyze")}
                      </button>
                    </div>
                  </article>
                ))
              ) : (
                <article className="profile-card settings-empty-card">
                  <strong>{t("settings.sources.saved_empty_title")}</strong>
                  <p>{t("settings.sources.saved_empty_copy")}</p>
                </article>
              )}
            </div>
          </div>

          <div className="source-workbench">
            <div
              className="scope-switch"
              role="tablist"
              aria-label={t("settings.source_kind.label")}
            >
              {(["plex", "dlna"] as const).map((kind) => (
                <button
                  key={kind}
                  className={sourceKind === kind ? "is-active" : ""}
                  onClick={() => setSourceKind(kind)}
                  type="button"
                >
                  {kind === "plex" ? t("source.plex") : t("source.dlna")}
                </button>
              ))}
            </div>

            {sourceKind === "plex" ? (
              <div className={`source-auth-card source-auth-card--${plexLinkState}`}>
                <div className="source-auth-card__copy">
                  <span>{t("settings.plex.flow_eyebrow")}</span>
                  <h3>{plexStatusLabel}</h3>
                  <p>{plexStatusCopy}</p>
                </div>
                <div className="inline-actions">
                  <button
                    className="primary-button"
                    onClick={() => startPlexMutation.mutate()}
                    type="button"
                  >
                    <Rocket size={14} /> {t("settings.action.link_plex")}
                  </button>
                  {plexAuthUrl ? (
                    <button
                      className="secondary-button"
                      onClick={() =>
                        void openInAppContainer(plexAuthUrl, t("settings.plex.login_title"))
                      }
                      type="button"
                    >
                      <Link2 size={14} /> {t("settings.action.open_plex_web")}
                    </button>
                  ) : null}
                  <button
                    className="secondary-button"
                    onClick={() => discoverMutation.mutate()}
                    type="button"
                  >
                    <Radar size={14} /> {t("settings.action.search_network")}
                  </button>
                </div>
              </div>
            ) : (
              <div className="source-auth-card source-auth-card--idle">
                <div className="source-auth-card__copy">
                  <span>{t("settings.dlna.flow_eyebrow")}</span>
                  <h3>{t("settings.dlna.title")}</h3>
                  <p>{t("settings.dlna.copy")}</p>
                </div>
                <div className="inline-actions">
                  <button
                    className="primary-button"
                    onClick={() => discoverMutation.mutate()}
                    type="button"
                  >
                    <Radar size={14} /> {t("settings.action.search_network")}
                  </button>
                </div>
              </div>
            )}

            {(discoverMutation.error || saveProfileMutation.error) ? (
              <p className="settings-helper settings-helper--error">
                {errorMessage(
                  saveProfileMutation.error ?? discoverMutation.error,
                  t("app.error.unknown")
                )}
              </p>
            ) : null}

            <div className="settings-section-copy">
              <span>{t("settings.sources.discovery_title")}</span>
              <p>
                {sourceKind === "plex"
                  ? t("settings.sources.discovery_copy")
                  : t("settings.sources.discovery_dlna_copy")}
              </p>
            </div>

            <div className="profile-grid">
              {discoveredServers.length ? (
                discoveredServers.map((server) => {
                  const registered = profiles.find((profile) => matchesServer(profile, server));
                  return (
                    <article
                      key={`${server.source_type}-${server.host}-${server.port}-${server.name}`}
                      className="profile-card settings-profile-card"
                    >
                      <div className="settings-profile-card__head">
                        <span>{server.source_type === "plex" ? t("source.plex") : t("source.dlna")}</span>
                        <div className="settings-profile-card__badges">
                          {server.local ? (
                            <span className="settings-mini-badge">{t("settings.sources.badge.local")}</span>
                          ) : null}
                          {server.relay ? (
                            <span className="settings-mini-badge">{t("settings.sources.badge.relay")}</span>
                          ) : null}
                          {registered ? (
                            <span className="settings-mini-badge">{t("settings.sources.badge.saved")}</span>
                          ) : null}
                        </div>
                      </div>
                      <strong>{server.name}</strong>
                      <p>{describeEndpoint(server.host, server.port) || t("sidebar.host_pending")}</p>
                      <small>{server.base_url || server.location || t("app.not_available")}</small>
                      <div className="inline-actions">
                        {renderDiscoveryAction(server)}
                      </div>
                      {server.source_type === "plex" && plexLinkState !== "complete" ? (
                        <p className="settings-helper">
                          {t("settings.plex.link_required")}
                        </p>
                      ) : null}
                    </article>
                  );
                })
              ) : (
                <article className="profile-card settings-empty-card">
                  <strong>{t("settings.sources.discovery_empty_title")}</strong>
                  <p>{t("settings.sources.discovery_empty_copy")}</p>
                </article>
              )}
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title={t("settings.infrastructure.title")}
        eyebrow={t("settings.infrastructure.eyebrow")}
        actions={
          <span className={`settings-state-pill${omdbReady ? " settings-state-pill--accent" : ""}`}>
            {omdbReady ? t("settings.state.ready") : t("settings.state.pending")}
          </span>
        }
      >
        <div className="settings-connector-card">
          <div className="settings-section-copy">
            <span>{t("settings.omdb.label")}</span>
            <p>{omdbReady ? t("settings.omdb.configured_help") : t("settings.omdb.help")}</p>
          </div>
          <div className="settings-connector-card__form">
            <label className="form-field">
              <span>{t("settings.omdb.placeholder")}</span>
              <input
                onChange={(event) => setOmdbValue(event.target.value)}
                placeholder={omdbReady ? "••••••••" : t("settings.omdb.placeholder")}
                type="password"
                value={omdbValue}
              />
            </label>
            <div className="inline-actions">
              <button
                className="primary-button"
                onClick={() => omdbMutation.mutate()}
                type="button"
              >
                {t("settings.omdb.save")}
              </button>
              <button
                className="secondary-button"
                onClick={() =>
                  void openInAppContainer("https://www.omdbapi.com/apikey.aspx", t("settings.omdb.label"))
                }
                type="button"
              >
                {t("settings.omdb.generate")} <ArrowUpRight size={14} />
              </button>
            </div>
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
                <strong className={`status-pill status-pill--${run.status.toLowerCase()}`}>
                  {translateStatus(run.status, t)}
                </strong>
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

            <div className="inline-actions">
              {run && isRunActive ? (
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
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title={t("settings.support.title")}
        eyebrow={t("settings.support.eyebrow")}
        actions={
          hasSupportedProject ? (
            <span className="settings-state-pill settings-state-pill--accent">
              <Check size={14} /> {t("settings.support.supported_badge")}
            </span>
          ) : null
        }
      >
        <div className="settings-support-shell">
          {hasSupportedProject ? (
            <article className="settings-support-thanks">
              <span>{t("settings.support.eyebrow")}</span>
              <h3>{t("settings.support.thanks_title")}</h3>
              <p>{t("settings.support.thanks_copy")}</p>
            </article>
          ) : (
            <article className="settings-support-card">
              <div className="settings-support-card__header">
                <span>{t("settings.support.eyebrow")}</span>
                <h3>{t("settings.support.title")}</h3>
                <p>{t("settings.support.copy")}</p>
              </div>

              <div className="settings-support-bundles">
                {supportBundles.map((bundle) => (
                  <article key={bundle.key} className="settings-support-bundle">
                    <span>{bundle.label}</span>
                    <strong>{bundle.amount}</strong>
                    <p>{bundle.copy}</p>
                  </article>
                ))}
              </div>

              <p className="settings-support-note">{t("settings.support.note")}</p>

              <div className="settings-support-actions">
                <div className="inline-actions">
                  <button
                    className="primary-button"
                    onClick={() =>
                      void openInAppContainer(SUPPORT_URL, t("settings.support.title"))
                    }
                    type="button"
                  >
                    <Heart size={14} /> {t("settings.support.cta")}
                  </button>
                  <button
                    className="secondary-button"
                    onClick={() => setHasSupportedProject(true)}
                    type="button"
                  >
                    <Check size={14} /> {t("settings.support.already_supported")}
                  </button>
                </div>
                <small className="settings-support-meta">{t("settings.support.cta_subtitle")}</small>
              </div>
            </article>
          )}
        </div>
      </SectionCard>
    </div>
  );
}
