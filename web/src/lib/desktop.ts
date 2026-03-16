declare global {
  interface Window {
    pywebview?: {
      api?: {
        is_desktop?: () => Promise<boolean> | boolean;
        open_external_url?: (url: string) => Promise<boolean> | boolean;
        open_url?: (url: string, title?: string) => Promise<boolean> | boolean;
      };
    };
  }
}

function desktopApi() {
  return window.pywebview?.api;
}

export function isDesktopShell() {
  return typeof window !== "undefined" && typeof desktopApi()?.open_url === "function";
}

export async function openInAppContainer(url: string, title?: string) {
  const cleanUrl = String(url || "").trim();
  if (!cleanUrl) {
    return false;
  }

  const api = desktopApi();
  if (typeof api?.open_url === "function") {
    try {
      const opened = await api.open_url(cleanUrl, title);
      if (opened) {
        return true;
      }
    } catch {
      // Browser fallback below.
    }
  }

  window.open(cleanUrl, "_blank", "noopener,noreferrer");
  return false;
}

export async function openExternalUrl(url: string) {
  const cleanUrl = String(url || "").trim();
  if (!cleanUrl) {
    return false;
  }

  const api = desktopApi();
  if (typeof api?.open_external_url === "function") {
    try {
      if (await api.open_external_url(cleanUrl)) {
        return true;
      }
    } catch {
      // Browser fallback below.
    }
  }

  window.open(cleanUrl, "_blank", "noopener,noreferrer");
  return false;
}
