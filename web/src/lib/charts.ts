import type { EChartsOption } from "echarts";

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
  yName: string
): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const groups = ["KEEP", "MAYBE", "DELETE", "UNKNOWN"].map((decision) => ({
    name: decision,
    type: "scatter" as const,
    symbolSize: 12,
    itemStyle: { color: colors[decision] },
    data: rows
      .filter((row) => decisionKey(row) === decision)
      .map((row) => {
        const x = parseMaybeNumber(row[xKey]);
        const y = parseMaybeNumber(row[yKey]);
        if (x === null || y === null) return null;
        return [x, y, String(row.title || "Sin título"), String(row.library || "Sin biblioteca")];
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
          return "Dato no disponible";
        }
        return `${value[2]}<br/>${value[3]}<br/>${xName}: ${value[0]}<br/>${yName}: ${value[1]}`;
      }
    },
    series: groups
  };
}

function decisionDistribution(rows: ReportRow[]): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const counters = new Map<string, number>();
  rows.forEach((row) => {
    const key = decisionKey(row);
    counters.set(key, (counters.get(key) ?? 0) + 1);
  });
  const data = Array.from(counters.entries()).map(([name, value]) => ({
    name,
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

function boxplotByLibrary(rows: ReportRow[]): EChartsOption {
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
    yAxis: { type: "value", name: "IMDb", nameTextStyle: { color: theme.muted } },
    series: [
      {
        type: "boxplot" as const,
        itemStyle: { color: theme.accent, borderColor: theme.line },
        data: libraries.map((item) => item.stats)
      }
    ]
  };
}

function wasteMap(rows: ReportRow[]): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: { type: "value", name: "IMDb" },
    yAxis: { type: "value", name: "GB" },
    series: ["DELETE", "MAYBE", "KEEP"].map((decision) => ({
      name: decision,
      type: "scatter" as const,
      itemStyle: { color: colors[decision] },
      symbolSize: (value: Array<number | string>) => Math.max(10, Number(value[2] || 0) * 1.4),
      data: rows
        .filter((row) => decisionKey(row) === decision)
        .map((row) => {
          const imdb = parseMaybeNumber(row.imdb_rating);
          const size = parseMaybeNumber(row.file_size_gb);
          if (imdb === null || size === null) return null;
          return [imdb, size, size, String(row.title || "Sin título")];
        })
        .filter((value): value is WasteValue => Array.isArray(value))
    }))
  };
}

function valuePerGb(rows: ReportRow[]): EChartsOption {
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
    xAxis: { type: "value", name: "IMDb medio / GB" },
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

function spaceByLibrary(rows: ReportRow[]): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const libraries = Array.from(
    new Set(rows.map((row) => String(row.library || "").trim()).filter(Boolean))
  );
  const series = ["KEEP", "MAYBE", "DELETE"].map((decision) => ({
    name: decision,
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
    yAxis: { type: "value", name: "GB" },
    series
  };
}

function decadeDistribution(rows: ReportRow[]): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const decades = Array.from(
    new Set(rows.map((row) => row.decade_label).filter((value): value is string => typeof value === "string"))
  ).sort();
  const series = ["KEEP", "MAYBE", "DELETE"].map((decision) => ({
    name: decision,
    type: "bar" as const,
    stack: "count",
    itemStyle: { color: colors[decision] },
    data: decades.map(
      (decade) =>
        rows.filter((row) => row.decade_label === decade && decisionKey(row) === decision).length
    )
  }));
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted } },
    xAxis: { type: "category", data: decades, axisLabel: { color: theme.muted } },
    yAxis: { type: "value", name: "Títulos" },
    series
  };
}

function genreDistribution(rows: ReportRow[]): EChartsOption {
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
    yAxis: { type: "value", name: "Títulos" },
    series: [
      { name: "KEEP", type: "bar" as const, stack: "genre", itemStyle: { color: colors.KEEP }, data: genres.map((item) => item.keep) },
      { name: "MAYBE", type: "bar" as const, stack: "genre", itemStyle: { color: colors.MAYBE }, data: genres.map((item) => item.maybe) },
      { name: "DELETE", type: "bar" as const, stack: "genre", itemStyle: { color: colors.DELETE }, data: genres.map((item) => item.delete) }
    ]
  };
}

function directorRanking(rows: ReportRow[]): EChartsOption {
  const theme = chartTheme();
  const directors = collectDirectors(rows);
  return {
    ...baseChart(),
    xAxis: { type: "value", name: "IMDb medio" },
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

function wordRanking(rows: ReportRow[]): EChartsOption {
  const theme = chartTheme();
  const words = collectTitleWords(rows);
  return {
    ...baseChart(),
    xAxis: { type: "value", name: "Frecuencia" },
    yAxis: { type: "category", data: words.map((item) => item.word), axisLabel: { color: theme.muted } },
    series: [
      {
        type: "bar" as const,
        data: words.map((item) => ({ value: item.count, itemStyle: { color: theme.maybe } }))
      }
    ]
  };
}

function imdbByDecision(rows: ReportRow[]): EChartsOption {
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
    xAxis: { type: "category", data: decisions, axisLabel: { color: theme.muted } },
    yAxis: { type: "value", name: "IMDb medio" },
    series: [{ type: "bar" as const, data }]
  };
}

export function buildChartOption(viewKey: DashboardViewKey, rows: ReportRow[]): EChartsOption {
  switch (viewKey) {
    case "imdb-metacritic":
      return scatterOption(rows, "imdb_rating", "metacritic_score", "IMDb", "Metacritic");
    case "decision-distribution":
      return decisionDistribution(rows);
    case "boxplot-library":
      return boxplotByLibrary(rows);
    case "imdb-rt":
      return scatterOption(rows, "imdb_rating", "rt_score", "IMDb", "RT");
    case "waste-map":
      return wasteMap(rows);
    case "value-per-gb":
      return valuePerGb(rows);
    case "space-library":
      return spaceByLibrary(rows);
    case "decade-distribution":
      return decadeDistribution(rows);
    case "genre-distribution":
      return genreDistribution(rows);
    case "director-ranking":
      return directorRanking(rows);
    case "word-ranking":
      return wordRanking(rows);
    case "imdb-by-decision":
      return imdbByDecision(rows);
    default:
      return decisionDistribution(rows);
  }
}
