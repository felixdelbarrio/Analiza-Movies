export interface ThemeTokens {
  text: string;
  muted: string;
  line: string;
  panel: string;
  accent: string;
  accent2: string;
  keep: string;
  maybe: string;
  danger: string;
  tooltipBg: string;
  tooltipBorder: string;
  tooltipText: string;
  tooltipMuted: string;
  tooltipShadow: string;
}

const FALLBACKS: ThemeTokens = {
  text: "#edf2f7",
  muted: "#99a7bd",
  line: "rgba(158, 179, 209, 0.14)",
  panel: "#101a2a",
  accent: "#71d5ff",
  accent2: "#7fe6bd",
  keep: "#69d29d",
  maybe: "#f0bf62",
  danger: "#ff7a7a",
  tooltipBg: "#0c1624",
  tooltipBorder: "rgba(113, 213, 255, 0.24)",
  tooltipText: "#f6fbff",
  tooltipMuted: "#c1cfe0",
  tooltipShadow: "0 22px 52px rgba(2, 10, 22, 0.44)"
};

export function readCssVar(name: string, fallback: string) {
  if (typeof window === "undefined") {
    return fallback;
  }
  const value = window.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function readThemeTokens(): ThemeTokens {
  return {
    text: readCssVar("--text", FALLBACKS.text),
    muted: readCssVar("--muted", FALLBACKS.muted),
    line: readCssVar("--line", FALLBACKS.line),
    panel: readCssVar("--panel-solid", FALLBACKS.panel),
    accent: readCssVar("--accent", FALLBACKS.accent),
    accent2: readCssVar("--accent-2", FALLBACKS.accent2),
    keep: readCssVar("--keep", FALLBACKS.keep),
    maybe: readCssVar("--warn", FALLBACKS.maybe),
    danger: readCssVar("--danger", FALLBACKS.danger),
    tooltipBg: readCssVar("--tooltip-bg", FALLBACKS.tooltipBg),
    tooltipBorder: readCssVar("--tooltip-border", FALLBACKS.tooltipBorder),
    tooltipText: readCssVar("--tooltip-text", FALLBACKS.tooltipText),
    tooltipMuted: readCssVar("--tooltip-muted", FALLBACKS.tooltipMuted),
    tooltipShadow: readCssVar("--tooltip-shadow", FALLBACKS.tooltipShadow)
  };
}
