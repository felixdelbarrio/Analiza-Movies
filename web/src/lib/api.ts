import type {
  ConfigState,
  DeleteActionResponse,
  MetadataRow,
  PagedResponse,
  Profile,
  ReportRow,
  RunLogsResponse,
  RunStatusResponse,
  ServerDiscovery
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.toString().trim() || "";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function preferredLanguageTag() {
  if (typeof document !== "undefined") {
    return document.documentElement.lang || window.navigator.language || "en";
  }
  return "en";
}

function localizedNetworkMessage(languageTag: string) {
  const language = languageTag.toLowerCase();
  const backendTarget = API_BASE_URL || "http://127.0.0.1:8000";

  if (language.startsWith("es")) {
    return `No se pudo conectar con la API (${backendTarget}). Usa \`make run\` para la app completa o configura \`VITE_API_BASE_URL\` hacia un backend FastAPI disponible.`;
  }
  if (language.startsWith("fr")) {
    return `Connexion impossible a l'API (${backendTarget}). Utilisez \`make run\` pour l'application complete ou configurez \`VITE_API_BASE_URL\` vers un backend FastAPI disponible.`;
  }
  if (language.startsWith("de")) {
    return `Die API (${backendTarget}) ist nicht erreichbar. Nutze \`make run\` fuer die komplette App oder konfiguriere \`VITE_API_BASE_URL\` auf ein verfuegbares FastAPI-Backend.`;
  }
  if (language.startsWith("it")) {
    return `Impossibile raggiungere l'API (${backendTarget}). Usa \`make run\` per l'app completa oppure configura \`VITE_API_BASE_URL\` verso un backend FastAPI disponibile.`;
  }
  if (language.startsWith("pt")) {
    return `Nao foi possivel ligar a API (${backendTarget}). Usa \`make run\` para a aplicacao completa ou configura \`VITE_API_BASE_URL\` para um backend FastAPI disponivel.`;
  }
  return `Could not reach the API (${backendTarget}). Use \`make run\` for the full app or point \`VITE_API_BASE_URL\` to an available FastAPI backend.`;
}

function parseResponsePayload(bodyText: string) {
  const text = bodyText.trim();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function buildUrl(path: string, params?: Record<string, string | number | undefined | null>) {
  const url = new URL(path, API_BASE_URL || window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") {
        return;
      }
      url.searchParams.set(key, String(value));
    });
  }
  return API_BASE_URL ? url.toString() : `${url.pathname}${url.search}`;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  params?: Record<string, string | number | undefined | null>
): Promise<T> {
  const preferredLanguage = preferredLanguageTag();
  let response: Response;
  try {
    response = await fetch(buildUrl(path, params), {
      headers: {
        "Content-Type": "application/json",
        "Accept-Language": preferredLanguage,
        ...(init?.headers ?? {})
      },
      ...init
    });
  } catch {
    throw new ApiError(0, localizedNetworkMessage(preferredLanguage));
  }

  if (response.status === 204) {
    return null as T;
  }

  const bodyText = await response.text();
  const payload = parseResponsePayload(bodyText);

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    if (payload && typeof payload === "object") {
      const detail = (payload as { detail?: unknown }).detail;
      if (typeof detail === "string" && detail.trim()) {
        message = detail;
      }
    } else if (bodyText.trim()) {
      message = bodyText.trim();
    }
    throw new ApiError(response.status, message);
  }

  if (payload === null) {
    return null as T;
  }

  if (typeof payload === "object") {
    return payload as T;
  }

  return bodyText as T;
}

async function fetchAllPages<T>(path: string, profileId?: string | null): Promise<T[]> {
  const pageSize = 2000;
  const pageConcurrency = 4;
  const firstPage = await requestJson<PagedResponse<T> | null>(path, undefined, {
    limit: pageSize,
    offset: 0,
    profile_id: profileId ?? undefined
  });

  if (!firstPage) {
    return [];
  }

  const items = [...firstPage.items];
  const effectiveLimit = firstPage.limit > 0 ? firstPage.limit : pageSize;
  const offsets: number[] = [];
  for (let offset = effectiveLimit; offset < firstPage.total; offset += effectiveLimit) {
    offsets.push(offset);
  }

  for (let index = 0; index < offsets.length; index += pageConcurrency) {
    const chunk = offsets.slice(index, index + pageConcurrency);
    const pages = await Promise.all(
      chunk.map((offset) =>
        requestJson<PagedResponse<T> | null>(path, undefined, {
          limit: effectiveLimit,
          offset,
          profile_id: profileId ?? undefined
        })
      )
    );
    for (const page of pages) {
      if (page) {
        items.push(...page.items);
      }
    }
  }

  return items;
}

export async function fetchConfigState(): Promise<ConfigState> {
  return requestJson<ConfigState>("/config/state");
}

export async function updateConfigState(payload: Partial<ConfigState>): Promise<ConfigState> {
  return requestJson<ConfigState>("/config/state", {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export async function saveProfile(payload: { profile: Partial<Profile>; set_active?: boolean }) {
  return requestJson<ConfigState>("/config/profiles", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function setActiveProfile(profileId: string) {
  return requestJson<ConfigState>("/config/profiles/active", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId })
  });
}

export async function startAnalysis(profileId: string) {
  return requestJson<RunStatusResponse>("/config/run", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId })
  });
}

export async function fetchRunStatus() {
  return requestJson<RunStatusResponse>("/config/run");
}

export async function fetchRunLogs(limit = 80) {
  return requestJson<RunLogsResponse>("/config/run/logs", undefined, { limit });
}

export async function stopAnalysis() {
  return requestJson<RunStatusResponse>("/config/run", {
    method: "DELETE"
  });
}

export async function startPlexAuth(openBrowser = true) {
  return requestJson<{ session_id: string; auth_url: string; status: string }>(
    "/config/plex/auth/start",
    {
      method: "POST",
      body: JSON.stringify({ open_browser: openBrowser })
    }
  );
}

export async function pollPlexAuth(sessionId: string) {
  return requestJson<{ status: string; session_id: string; servers?: ServerDiscovery[] }>(
    `/config/plex/auth/${encodeURIComponent(sessionId)}`
  );
}

export async function discoverPlex(sessionId?: string | null) {
  return requestJson<{
    servers: ServerDiscovery[];
    session_id?: string | null;
    auth_complete?: boolean;
  }>("/config/discover/plex", {
    method: "POST",
    body: JSON.stringify(sessionId ? { session_id: sessionId } : {})
  });
}

export async function discoverDlna() {
  return requestJson<{ devices: ServerDiscovery[] }>("/config/discover/dlna", {
    method: "POST",
    body: JSON.stringify({})
  });
}

export async function fetchReportAll(profileId?: string | null) {
  return fetchAllPages<ReportRow>("/reports/all", profileId);
}

export async function fetchReportFiltered(profileId?: string | null) {
  return fetchAllPages<ReportRow>("/reports/filtered", profileId);
}

export async function fetchMetadata(profileId?: string | null) {
  try {
    return await fetchAllPages<MetadataRow>("/reports/metadata-fix", profileId);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return [] as MetadataRow[];
    }
    throw error;
  }
}

export async function runDeleteAction(
  rows: ReportRow[],
  dryRun: boolean,
  profileId?: string | null
) {
  return requestJson<DeleteActionResponse>("/actions/delete", {
    method: "POST",
    body: JSON.stringify({ rows, dry_run: dryRun, profile_id: profileId })
  });
}
