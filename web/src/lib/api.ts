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

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
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
  const preferredLanguage =
    typeof document !== "undefined"
      ? document.documentElement.lang || window.navigator.language
      : "en";
  const response = await fetch(buildUrl(path, params), {
    headers: {
      "Content-Type": "application/json",
      "Accept-Language": preferredLanguage,
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (response.status === 204) {
    return null as T;
  }

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        message = payload.detail;
      }
    } catch {
      const text = await response.text();
      if (text.trim()) {
        message = text;
      }
    }
    throw new ApiError(response.status, message);
  }

  return (await response.json()) as T;
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
  return requestJson<{ servers: ServerDiscovery[] }>("/config/discover/plex", {
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

export async function runDeleteAction(rows: ReportRow[], dryRun: boolean) {
  return requestJson<DeleteActionResponse>("/actions/delete", {
    method: "POST",
    body: JSON.stringify({ rows, dry_run: dryRun })
  });
}
