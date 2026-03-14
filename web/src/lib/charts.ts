import type { EChartsOption } from "echarts";

import type { Translator } from "../i18n/catalog";
import { translateDecision } from "../i18n/helpers";
import { collectDirectors, collectGenres, collectTitleWords, parseMaybeNumber } from "./data";
import type { DashboardViewKey, ReportRow } from "./types";

interface ChartTheme {
  text: string;
  muted: string;
  line: string;
  panel: string;
  accent: string;
  accent2: string;
  keep: string;
  maybe: string;
  danger: string;
}

type ScatterValue = [number, number, string, string];
type WasteValue = [number, number, number, string];

interface ChartIntl {
  locale: string;
  t: Translator;
}

function readCssVar(name: string, fallback: string) {
  if (typeof window === "undefined") {
    return fallback;
  }
  const value = window.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function chartTheme(): ChartTheme {
  return {
    text: readCssVar("--text", "#edf2f7"),
    muted: readCssVar("--muted", "#99a7bd"),
    line: readCssVar("--line", "rgba(158, 179, 209, 0.14)"),
    panel: readCssVar("--panel-solid", "#101a2a"),
    accent: readCssVar("--accent", "#71d5ff"),
    accent2: readCssVar("--accent-2", "#7fe6bd"),
    keep: readCssVar("--keep", "#69d29d"),
    maybe: readCssVar("--warn", "#f0bf62"),
    danger: readCssVar("--danger", "#ff7a7a")
  };
}

function decisionColors(theme: ChartTheme): Record<string, string> {
  return {
    KEEP: theme.keep,
    MAYBE: theme.maybe,
    DELETE: theme.danger,
    UNKNOWN: theme.muted
  };
}

function formatValue(value: number, locale: string, digits = 1) {
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(value);
}

function quantile(values: number[], ratio: number) {
  if (!values.length) return 0;
  const position = (values.length - 1) * ratio;
  const base = Math.floor(position);
  const rest = position - base;
  if (values[base + 1] !== undefined) {
    return values[base] + rest * (values[base + 1] - values[base]);
  }
  return values[base];
}

function decisionKey(row: ReportRow) {
  return String(row.decision || "UNKNOWN").toUpperCase();
}

function baseChart(): EChartsOption {
  const theme = chartTheme();
  return {
    backgroundColor: "transparent",
    textStyle: { color: theme.text, fontFamily: "Space Grotesk" },
    tooltip: {
      trigger: "axis",
      backgroundColor: theme.panel,
      borderColor: theme.line,
      textStyle: { color: theme.text }
    },
    grid: { top: 48, left: 44, right: 28, bottom: 48 },
    xAxis: {
      axisLine: { lineStyle: { color: theme.line } },
      axisLabel: { color: theme.muted },
      splitLine: { lineStyle: { color: theme.line } }
    },
    yAxis: {
      axisLine: { lineStyle: { color: theme.line } },
      axisLabel: { color: theme.muted },
      splitLine: { lineStyle: { color: theme.line } }
    }
  };
}

function scatterOption(
  rows: ReportRow[],
  xKey: keyof ReportRow,
  yKey: keyof ReportRow,
  xName: string,
  yName: string,
  intl: ChartIntl
): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const groups = ["KEEP", "MAYBE", "DELETE", "UNKNOWN"].map((decision) => ({
    name: translateDecision(decision, intl.t),
    type: "scatter" as const,
    symbolSize: 12,
    itemStyle: { color: colors[decision] },
    data: rows
      .filter((row) => decisionKey(row) === decision)
      .map((row) => {
        const x = parseMaybeNumber(row[xKey]);
        const y = parseMaybeNumber(row[yKey]);
        if (x === null || y === null) return null;
        return [x, y, String(row.title || intl.t("column.title")), String(row.library || intl.t("column.library"))];
      })
      .filter((value): value is ScatterValue => Array.isArray(value))
  }));

  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: { name: xName, type: "value", nameTextStyle: { color: theme.muted } },
    yAxis: { name: yName, type: "value", nameTextStyle: { color: theme.muted } },
    tooltip: {
      formatter: (params: unknown) => {
        const current = Array.isArray(params) ? params[0] : params;
        const value = current && typeof current === "object" && "value" in current
          ? (current.value as ScatterValue)
          : undefined;
        if (!Array.isArray(value)) {
          return intl.t("chart.no_data");
        }
        return `${value[2]}<br/>${value[3]}<br/>${xName}: ${formatValue(value[0], intl.locale, 1)}<br/>${yName}: ${formatValue(value[1], intl.locale, 1)}`;
      }
    },
    series: groups
  };
}

function decisionDistribution(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const counters = new Map<string, number>();
  rows.forEach((row) => {
    const key = decisionKey(row);
    counters.set(key, (counters.get(key) ?? 0) + 1);
  });
  const data = Array.from(counters.entries()).map(([name, value]) => ({
    name: translateDecision(name, intl.t),
    value,
    itemStyle: { color: colors[name] ?? colors.UNKNOWN }
  }));
  return {
    ...baseChart(),
    tooltip: { trigger: "item" },
    legend: { bottom: 0, textStyle: { color: theme.muted } },
    series: [
      {
        type: "pie" as const,
        radius: ["48%", "74%"],
        center: ["50%", "46%"],
        label: { color: theme.text },
        data
      }
    ]
  };
}

function boxplotByLibrary(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const grouped = new Map<string, number[]>();
  rows.forEach((row) => {
    const library = String(row.library || "").trim();
    const rating = parseMaybeNumber(row.imdb_rating);
    if (!library || rating === null) return;
    const bucket = grouped.get(library) ?? [];
    bucket.push(rating);
    grouped.set(library, bucket);
  });

  const libraries = Array.from(grouped.entries())
    .filter(([, values]) => values.length >= 2)
    .map(([library, values]) => {
      values.sort((left, right) => left - right);
      return {
        library,
        stats: [
          values[0],
          quantile(values, 0.25),
          quantile(values, 0.5),
          quantile(values, 0.75),
          values[values.length - 1]
        ]
      };
    })
    .sort((left, right) => right.stats[2] - left.stats[2])
    .slice(0, 12);

  return {
    ...baseChart(),
    tooltip: { trigger: "item" },
    xAxis: {
      type: "category",
      data: libraries.map((item) => item.library),
      axisLabel: { color: theme.muted, rotate: 24 }
    },
    yAxis: { type: "value", name: intl.t("chart.metric.imdb"), nameTextStyle: { color: theme.muted } },
    series: [
      {
        type: "boxplot" as const,
        itemStyle: { color: theme.accent, borderColor: theme.line },
        data: libraries.map((item) => item.stats)
      }
    ]
  };
}

function wasteMap(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: { type: "value", name: intl.t("chart.metric.imdb") },
    yAxis: { type: "value", name: intl.t("chart.metric.gb") },
    series: ["DELETE", "MAYBE", "KEEP"].map((decision) => ({
      name: translateDecision(decision, intl.t),
      type: "scatter" as const,
      itemStyle: { color: colors[decision] },
      symbolSize: (value: Array<number | string>) => Math.max(10, Number(value[2] || 0) * 1.4),
      data: rows
        .filter((row) => decisionKey(row) === decision)
        .map((row) => {
          const imdb = parseMaybeNumber(row.imdb_rating);
          const size = parseMaybeNumber(row.file_size_gb);
          if (imdb === null || size === null) return null;
          return [imdb, size, size, String(row.title || intl.t("column.title"))];
        })
        .filter((value): value is WasteValue => Array.isArray(value))
    }))
  };
}

function valuePerGb(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const grouped = new Map<string, { size: number; score: number; total: number }>();
  rows.forEach((row) => {
    const library = String(row.library || "").trim();
    const size = parseMaybeNumber(row.file_size_gb);
    const imdb = parseMaybeNumber(row.imdb_rating);
    if (!library || size === null || imdb === null) return;
    const current = grouped.get(library) ?? { size: 0, score: 0, total: 0 };
    current.size += size;
    current.score += imdb;
    current.total += 1;
    grouped.set(library, current);
  });
  const data = Array.from(grouped.entries())
    .map(([library, value]) => ({
      library,
      metric: value.size > 0 ? (value.score / value.total) / value.size : 0
    }))
    .sort((left, right) => right.metric - left.metric)
    .slice(0, 12);

  return {
    ...baseChart(),
    xAxis: { type: "value", name: intl.t("chart.metric.average_imdb_per_gb") },
    yAxis: {
      type: "category",
      data: data.map((item) => item.library),
      axisLabel: { color: theme.muted }
    },
    series: [
      {
        type: "bar" as const,
        data: data.map((item) => ({
          value: Number(item.metric.toFixed(4)),
          itemStyle: { color: theme.accent }
        }))
      }
    ]
  };
}

function spaceByLibrary(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const libraries = Array.from(
    new Set(rows.map((row) => String(row.library || "").trim()).filter(Boolean))
  );
  const series = ["KEEP", "MAYBE", "DELETE"].map((decision) => ({
    name: translateDecision(decision, intl.t),
    type: "bar" as const,
    stack: "size",
    itemStyle: { color: colors[decision] },
    data: libraries.map((library) =>
      Number(
        rows
        .filter((row) => String(row.library || "").trim() === library && decisionKey(row) === decision)
        .reduce((acc, row) => acc + (parseMaybeNumber(row.file_size_gb) ?? 0), 0)
        .toFixed(1)
      )
    )
  }));

  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: {
      type: "category",
      data: libraries,
      axisLabel: { color: theme.muted, rotate: 24 }
    },
    yAxis: { type: "value", name: intl.t("chart.metric.gb") },
    series
  };
}

function decadeDistribution(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const decades = Array.from(
    new Set(rows.map((row) => row.decade).filter((value): value is number => typeof value === "number"))
  ).sort((left, right) => left - right);
  const series = ["KEEP", "MAYBE", "DELETE"].map((decision) => ({
    name: translateDecision(decision, intl.t),
    type: "bar" as const,
    stack: "count",
    itemStyle: { color: colors[decision] },
    data: decades.map(
      (decade) =>
        rows.filter((row) => row.decade === decade && decisionKey(row) === decision).length
    )
  }));
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: {
      type: "category",
      data: decades.map((decade) => intl.t("data.decade_label", { decade })),
      axisLabel: { color: theme.muted }
    },
    yAxis: { type: "value", name: intl.t("chart.metric.titles") },
    series
  };
}

function genreDistribution(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const genres = collectGenres(rows);
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: {
      type: "category",
      data: genres.map((item) => item.genre),
      axisLabel: { rotate: 28, color: theme.muted }
    },
    yAxis: { type: "value", name: intl.t("chart.metric.titles") },
    series: [
      { name: translateDecision("KEEP", intl.t), type: "bar" as const, stack: "genre", itemStyle: { color: colors.KEEP }, data: genres.map((item) => item.keep) },
      { name: translateDecision("MAYBE", intl.t), type: "bar" as const, stack: "genre", itemStyle: { color: colors.MAYBE }, data: genres.map((item) => item.maybe) },
      { name: translateDecision("DELETE", intl.t), type: "bar" as const, stack: "genre", itemStyle: { color: colors.DELETE }, data: genres.map((item) => item.delete) }
    ]
  };
}

function directorRanking(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const directors = collectDirectors(rows);
  return {
    ...baseChart(),
    xAxis: { type: "value", name: intl.t("chart.metric.average_imdb") },
    yAxis: {
      type: "category",
      data: directors.map((item) => item.director),
      axisLabel: { color: theme.muted }
    },
    series: [
      {
        type: "bar" as const,
        data: directors.map((item) => ({
          value: Number(item.mean.toFixed(2)),
          itemStyle: { color: theme.accent2 }
        }))
      }
    ]
  };
}

function wordRanking(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const words = collectTitleWords(rows);
  return {
    ...baseChart(),
    xAxis: { type: "value", name: intl.t("chart.metric.frequency") },
    yAxis: { type: "category", data: words.map((item) => item.word), axisLabel: { color: theme.muted } },
    series: [
      {
        type: "bar" as const,
        data: words.map((item) => ({ value: item.count, itemStyle: { color: theme.maybe } }))
      }
    ]
  };
}

function imdbByDecision(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const decisions = ["KEEP", "MAYBE", "DELETE"];
  const data = decisions.map((decision) => {
    const bucket = rows
      .filter((row) => decisionKey(row) === decision)
      .map((row) => parseMaybeNumber(row.imdb_rating))
      .filter((value): value is number => value !== null);
    const average = bucket.length ? bucket.reduce((acc, value) => acc + value, 0) / bucket.length : 0;
    return {
      value: Number(average.toFixed(2)),
      itemStyle: { color: colors[decision] }
    };
  });
  return {
    ...baseChart(),
    xAxis: {
      type: "category",
      data: decisions.map((decision) => translateDecision(decision, intl.t)),
      axisLabel: { color: theme.muted }
    },
    yAxis: { type: "value", name: intl.t("chart.metric.average_imdb") },
    series: [{ type: "bar" as const, data }]
  };
}

export function buildChartOption(
  viewKey: DashboardViewKey,
  rows: ReportRow[],
  intl: ChartIntl
): EChartsOption {
  switch (viewKey) {
    case "imdb-metacritic":
      return scatterOption(rows, "imdb_rating", "metacritic_score", intl.t("chart.metric.imdb"), intl.t("chart.metric.metacritic"), intl);
    case "decision-distribution":
      return decisionDistribution(rows, intl);
    case "boxplot-library":
      return boxplotByLibrary(rows, intl);
    case "imdb-rt":
      return scatterOption(rows, "imdb_rating", "rt_score", intl.t("chart.metric.imdb"), intl.t("chart.metric.rt"), intl);
    case "waste-map":
      return wasteMap(rows, intl);
    case "value-per-gb":
      return valuePerGb(rows, intl);
    case "space-library":
      return spaceByLibrary(rows, intl);
    case "decade-distribution":
      return decadeDistribution(rows, intl);
    case "genre-distribution":
      return genreDistribution(rows, intl);
    case "director-ranking":
      return directorRanking(rows, intl);
    case "word-ranking":
      return wordRanking(rows, intl);
    case "imdb-by-decision":
      return imdbByDecision(rows, intl);
    default:
      return decisionDistribution(rows, intl);
  }
}
