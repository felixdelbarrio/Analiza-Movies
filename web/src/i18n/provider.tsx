import { createContext, useContext, useEffect, useMemo } from "react";
import type { ReactNode } from "react";

import { useStoredState } from "../lib/preferences";
import {
  CATALOGS,
  SUPPORTED_LOCALES,
  type SupportedLocale,
  type TranslationKey,
  type TranslationParams,
  type TranslationPrimitive
} from "./catalog";

interface I18nContextValue {
  locale: SupportedLocale;
  setLocale: (locale: SupportedLocale) => void;
  t: (key: TranslationKey, params?: TranslationParams) => string;
  formatNumber: (value: number, options?: Intl.NumberFormatOptions) => string;
  formatDate: (value: string | number | Date, options?: Intl.DateTimeFormatOptions) => string;
  formatTime: (value: string | number | Date, options?: Intl.DateTimeFormatOptions) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

function replaceParams(template: string, params?: TranslationParams) {
  if (!params) {
    return template;
  }
  return template.replace(/\{(\w+)\}/g, (_, key: string) => {
    const value = params[key];
    if (value === null || value === undefined) {
      return "";
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    return String(value as TranslationPrimitive);
  });
}

export function resolveLocale(input?: string | null): SupportedLocale {
  const value = String(input || "").trim().toLowerCase();
  if (!value) {
    return "en";
  }
  const direct = SUPPORTED_LOCALES.find((locale) => locale === value);
  if (direct) {
    return direct;
  }
  const prefix = value.split(/[-_]/)[0];
  return SUPPORTED_LOCALES.find((locale) => locale === prefix) ?? "en";
}

function detectBrowserLocale(): SupportedLocale {
  if (typeof window === "undefined") {
    return "en";
  }
  return resolveLocale(window.navigator.language);
}

interface I18nProviderProps {
  children: ReactNode;
}

export function I18nProvider({ children }: I18nProviderProps) {
  const [locale, setLocale] = useStoredState<SupportedLocale>("am.locale", detectBrowserLocale());

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const value = useMemo<I18nContextValue>(() => {
    const catalog = CATALOGS[locale] ?? CATALOGS.en;
    return {
      locale,
      setLocale,
      t: (key, params) => replaceParams(catalog[key] ?? CATALOGS.en[key], params),
      formatNumber: (value, options) => new Intl.NumberFormat(locale, options).format(value),
      formatDate: (value, options) =>
        new Intl.DateTimeFormat(locale, options).format(new Date(value)),
      formatTime: (value, options) =>
        new Intl.DateTimeFormat(locale, {
          hour: "2-digit",
          minute: "2-digit",
          ...(options ?? {})
        }).format(new Date(value))
    };
  }, [locale, setLocale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used inside I18nProvider.");
  }
  return context;
}
