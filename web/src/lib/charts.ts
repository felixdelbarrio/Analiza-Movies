import type { EChartsOption } from "echarts";

import type { Translator } from "../i18n/catalog";
import { translateDecision } from "../i18n/helpers";
import { collectDirectors, collectGenres, collectTitleWords, parseMaybeNumber } from "./data";
import { readThemeTokens, type ThemeTokens } from "./theme";
import type { DashboardViewKey, ReportRow } from "./types";

type ChartTheme = ThemeTokens;

type ScatterValue = [number, number, string, string];
type WasteValue = [number, number, number, string];
type ValuePerGbRow = {
  deletePct: number;
  deleteSize: number;
  keepPct: number;
  keepSize: number;
  library: string;
  maybePct: number;
  maybeSize: number;
  metric: number;
  totalSize: number;
};

interface ChartIntl {
  locale: string;
  t: Translator;
}

function chartTheme(): ChartTheme {
  return readThemeTokens();
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

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function truncateLabel(value: string, maxLength = 18) {
  const normalized = String(value || "").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function axisInterval(length: number, target = 8) {
  if (length <= target) {
    return 0;
  }
  return Math.max(0, Math.ceil(length / target) - 1);
}

function tooltipCard(theme: ChartTheme, title: string, rows: Array<{ label: string; value: string; color?: string }>) {
  const body = rows
    .map(
      (row) =>
        `<div style="display:flex;justify-content:space-between;gap:16px;align-items:center;">
          <span style="display:inline-flex;align-items:center;gap:8px;color:${theme.tooltipMuted};">
            ${
              row.color
                ? `<span style="width:10px;height:10px;border-radius:999px;background:${row.color};display:inline-block;"></span>`
                : ""
            }
            ${escapeHtml(row.label)}
          </span>
          <strong style="color:${theme.tooltipText};font-weight:700;">${escapeHtml(row.value)}</strong>
        </div>`
    )
    .join("");

  return `<div style="display:grid;gap:10px;min-width:220px;">
    <div style="font-size:13px;font-weight:700;color:${theme.tooltipText};">${escapeHtml(title)}</div>
    <div style="display:grid;gap:6px;">${body}</div>
  </div>`;
}

function buildTooltip(theme: ChartTheme, option: Record<string, unknown> = {}) {
  return {
    backgroundColor: theme.tooltipBg,
    borderColor: theme.tooltipBorder,
    borderWidth: 1,
    padding: 12,
    textStyle: {
      color: theme.tooltipText,
      fontFamily: "Space Grotesk"
    },
    extraCssText: `border-radius:18px;box-shadow:${theme.tooltipShadow};backdrop-filter:blur(18px);`,
    ...option
  };
}

function firstTooltipParam(params: unknown) {
  return Array.isArray(params) ? params[0] : params;
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
    animationDuration: 420,
    animationDurationUpdate: 260,
    animationEasing: "cubicOut",
    textStyle: { color: theme.text, fontFamily: "Space Grotesk" },
    tooltip: buildTooltip(theme, { trigger: "axis" }),
    grid: { top: 64, left: 50, right: 28, bottom: 48, containLabel: true },
    xAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: theme.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: theme.line, type: "dashed" } }
    },
    yAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: theme.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: theme.line, type: "dashed" } }
    }
  };
}

function emptyChartOption(intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  return {
    backgroundColor: "transparent",
    animation: false,
    xAxis: { show: false, type: "value" },
    yAxis: { show: false, type: "value" },
    series: [],
    graphic: [
      {
        type: "group",
        left: "center",
        top: "middle",
        children: [
          {
            type: "text",
            style: {
              fill: theme.text,
              fontFamily: "Space Grotesk",
              fontSize: 16,
              fontWeight: 700,
              text: intl.t("chart.no_data")
            }
          }
        ]
      }
    ]
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
    symbolSize: 14,
    itemStyle: {
      color: colors[decision],
      borderColor: theme.panel,
      borderWidth: 1.5,
      opacity: 0.9
    },
    emphasis: {
      scale: 1.18,
      itemStyle: {
        shadowBlur: 18,
        shadowColor: colors[decision]
      }
    },
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
  if (!groups.some((group) => group.data.length)) {
    return emptyChartOption(intl);
  }

  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    xAxis: {
      name: xName,
      type: "value",
      nameTextStyle: { color: theme.muted, padding: [0, 0, 8, 0] }
    },
    yAxis: {
      name: yName,
      type: "value",
      nameTextStyle: { color: theme.muted, padding: [0, 0, 0, 8] }
    },
    tooltip: {
      formatter: (params: unknown) => {
        const current = Array.isArray(params) ? params[0] : params;
        const value = current && typeof current === "object" && "value" in current
          ? (current.value as ScatterValue)
          : undefined;
        if (!Array.isArray(value)) {
          return intl.t("chart.no_data");
        }
        return tooltipCard(theme, value[2], [
          { label: intl.t("column.library"), value: value[3] },
          { label: xName, value: formatValue(value[0], intl.locale, 1) },
          { label: yName, value: formatValue(value[1], intl.locale, 1) }
        ]);
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
  })).sort((left, right) => right.value - left.value);
  if (!data.length) {
    return emptyChartOption(intl);
  }
  const total = data.reduce((sum, item) => sum + item.value, 0);
  return {
    ...baseChart(),
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const current = firstTooltipParam(params);
        const name = current && typeof current === "object" && "name" in current ? String(current.name || "") : "";
        const value =
          current && typeof current === "object" && "value" in current ? Number(current.value ?? 0) : 0;
        const color =
          current && typeof current === "object" && "color" in current && typeof current.color === "string"
            ? current.color
            : undefined;
        return tooltipCard(theme, name, [
          {
            label: intl.t("chart.metric.titles"),
            value: formatValue(value, intl.locale, 0),
            color
          },
          {
            label: "%",
            value: total ? formatValue((value / total) * 100, intl.locale, 1) : "0"
          }
        ]);
      }
    },
    legend: { bottom: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    graphic: [
      {
        type: "group",
        left: "center",
        top: "39%",
        children: [
          {
            type: "text",
            style: {
              fill: theme.muted,
              fontFamily: "Space Grotesk",
              fontSize: 11,
              text: intl.t("chart.metric.titles")
            }
          },
          {
            type: "text",
            top: 18,
            style: {
              fill: theme.text,
              fontFamily: "Fraunces",
              fontSize: 26,
              fontWeight: 700,
              text: formatValue(total, intl.locale, 0)
            }
          }
        ]
      }
    ],
    series: [
      {
        type: "pie" as const,
        radius: ["54%", "78%"],
        center: ["50%", "42%"],
        minAngle: 6,
        label: {
          color: theme.text,
          formatter: (params: { percent?: number; name?: string }) =>
            Number(params.percent ?? 0) >= 9
              ? `${truncateLabel(String(params.name || ""), 12)}\n${formatValue(Number(params.percent ?? 0), intl.locale, 0)}%`
              : ""
        },
        labelLine: { length: 12, length2: 10, lineStyle: { color: theme.line } },
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
  if (!libraries.length) {
    return emptyChartOption(intl);
  }

  return {
    ...baseChart(),
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const current = firstTooltipParam(params);
        const axisValue =
          current && typeof current === "object" && "axisValue" in current
            ? String(current.axisValue || "")
            : "";
        const values =
          current && typeof current === "object" && "value" in current && Array.isArray(current.value)
            ? current.value
            : [];
        return tooltipCard(theme, axisValue, [
          { label: "Min", value: formatValue(Number(values[0] ?? 0), intl.locale, 1) },
          { label: "Q1", value: formatValue(Number(values[1] ?? 0), intl.locale, 1) },
          { label: "Median", value: formatValue(Number(values[2] ?? 0), intl.locale, 1) },
          { label: "Q3", value: formatValue(Number(values[3] ?? 0), intl.locale, 1) },
          { label: "Max", value: formatValue(Number(values[4] ?? 0), intl.locale, 1) }
        ]);
      }
    },
    xAxis: {
      type: "category",
      data: libraries.map((item) => item.library),
      axisLabel: {
        color: theme.muted,
        rotate: 18,
        interval: axisInterval(libraries.length, 9),
        formatter: (value: string) => truncateLabel(value, 16)
      }
    },
    yAxis: {
      type: "value",
      name: intl.t("chart.metric.imdb"),
      nameTextStyle: { color: theme.muted, padding: [0, 0, 0, 8] }
    },
    series: [
      {
        type: "boxplot" as const,
        itemStyle: { color: theme.accent2, borderColor: theme.accent, borderWidth: 1.2 },
        emphasis: { itemStyle: { shadowBlur: 18, shadowColor: theme.accent } },
        data: libraries.map((item) => item.stats)
      }
    ]
  };
}

function wasteMap(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const series = ["DELETE", "MAYBE", "KEEP"].map((decision) => ({
    name: translateDecision(decision, intl.t),
    type: "scatter" as const,
    itemStyle: {
      color: colors[decision],
      borderColor: theme.panel,
      borderWidth: 1.5,
      opacity: 0.86
    },
    symbolSize: (value: Array<number | string>) =>
      Math.max(12, Math.sqrt(Number(value[2] || 0)) * 7),
    data: rows
      .filter((row) => decisionKey(row) === decision)
      .map((row) => {
        const imdb = parseMaybeNumber(row.imdb_rating);
        const size = parseMaybeNumber(row.file_size_gb);
        if (imdb === null || size === null) return null;
        return [imdb, size, size, String(row.title || intl.t("column.title"))];
      })
      .filter((value): value is WasteValue => Array.isArray(value))
  }));
  if (!series.some((item) => item.data.length)) {
    return emptyChartOption(intl);
  }
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    tooltip: {
      formatter: (params: unknown) => {
        const current = firstTooltipParam(params);
        const seriesName =
          current && typeof current === "object" && "seriesName" in current
            ? String(current.seriesName || "")
            : "";
        const value =
          current && typeof current === "object" && "value" in current && Array.isArray(current.value)
            ? current.value
            : [];
        return tooltipCard(theme, String(value[3] || intl.t("column.title")), [
          { label: intl.t("column.decision"), value: seriesName },
          {
            label: intl.t("chart.metric.imdb"),
            value: formatValue(Number(value[0] ?? 0), intl.locale, 1)
          },
          {
            label: intl.t("chart.metric.gb"),
            value: formatValue(Number(value[1] ?? 0), intl.locale, 1)
          }
        ]);
      }
    },
    xAxis: { type: "value", name: intl.t("chart.metric.imdb") },
    yAxis: { type: "value", name: intl.t("chart.metric.gb") },
    series
  };
}

function valuePerGb(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const grouped = new Map<
    string,
    {
      size: number;
      score: number;
      total: number;
      decisionSizes: Record<string, number>;
    }
  >();
  rows.forEach((row) => {
    const library = String(row.library || "").trim();
    const size = parseMaybeNumber(row.file_size_gb);
    const imdb = parseMaybeNumber(row.imdb_rating);
    if (!library || size === null || imdb === null) return;
    const current = grouped.get(library) ?? {
      size: 0,
      score: 0,
      total: 0,
      decisionSizes: { KEEP: 0, MAYBE: 0, DELETE: 0, UNKNOWN: 0 }
    };
    current.size += size;
    current.score += imdb;
    current.total += 1;
    current.decisionSizes[decisionKey(row)] += size;
    grouped.set(library, current);
  });
  const data = Array.from(grouped.entries())
    .map(([library, value]) => ({
      library,
      metric: value.size > 0 ? (value.score / value.total) / value.size : 0,
      totalSize: value.size,
      keepSize: value.decisionSizes.KEEP,
      maybeSize: value.decisionSizes.MAYBE,
      deleteSize: value.decisionSizes.DELETE,
      keepPct: value.size > 0 ? (value.decisionSizes.KEEP / value.size) * 100 : 0,
      maybePct: value.size > 0 ? (value.decisionSizes.MAYBE / value.size) * 100 : 0,
      deletePct: value.size > 0 ? (value.decisionSizes.DELETE / value.size) * 100 : 0
    }))
    .sort((left, right) => right.metric - left.metric)
    .slice(0, 12) as ValuePerGbRow[];
  if (!data.length) {
    return emptyChartOption(intl);
  }
  const maxMetric = Math.max(...data.map((item) => item.metric), 0.1);
  const rowByLibrary = new Map(data.map((item) => [item.library, item]));

  return {
    ...baseChart(),
    grid: { top: 72, left: 44, right: 92, bottom: 48 },
    legend: { top: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: theme.panel,
      borderColor: theme.line,
      textStyle: { color: theme.text },
      formatter: (params: unknown) => {
        const items = Array.isArray(params) ? params : [params];
        const axisValue = items[0] && typeof items[0] === "object" && "axisValue" in items[0]
          ? String(items[0].axisValue)
          : "";
        const row = rowByLibrary.get(axisValue);
        if (!row) {
          return intl.t("chart.no_data");
        }
        return tooltipCard(theme, row.library, [
          {
            label: intl.t("chart.metric.average_imdb_per_gb"),
            value: formatValue(row.metric, intl.locale, 3)
          },
          {
            label: intl.t("chart.metric.gb"),
            value: formatValue(row.totalSize, intl.locale, 1)
          },
          {
            label: translateDecision("KEEP", intl.t),
            value: `${formatValue(row.keepPct, intl.locale, 1)}% · ${formatValue(row.keepSize, intl.locale, 1)} GB`,
            color: theme.keep
          },
          {
            label: translateDecision("MAYBE", intl.t),
            value: `${formatValue(row.maybePct, intl.locale, 1)}% · ${formatValue(row.maybeSize, intl.locale, 1)} GB`,
            color: theme.maybe
          },
          {
            label: translateDecision("DELETE", intl.t),
            value: `${formatValue(row.deletePct, intl.locale, 1)}% · ${formatValue(row.deleteSize, intl.locale, 1)} GB`,
            color: theme.danger
          }
        ]);
      }
    },
    xAxis: [
      {
        type: "value",
        min: 0,
        max: 100,
        axisLabel: {
          color: theme.muted,
          formatter: (value: number) => `${Math.round(value)}%`
        },
        splitLine: { lineStyle: { color: theme.line } }
      },
      {
        type: "value",
        min: 0,
        max: Number((maxMetric * 1.18).toFixed(3)),
        position: "top",
        name: intl.t("chart.metric.average_imdb_per_gb"),
        nameTextStyle: { color: theme.muted },
        axisLine: { lineStyle: { color: theme.line } },
        axisLabel: { color: theme.muted },
        splitLine: { show: false }
      }
    ],
    yAxis: {
      type: "category",
      data: data.map((item) => item.library),
      inverse: true,
      axisLabel: {
        color: theme.muted,
        formatter: (value: string) => truncateLabel(value, 18)
      }
    },
    series: [
      {
        name: translateDecision("KEEP", intl.t),
        type: "bar" as const,
        stack: "composition",
        barWidth: 18,
        itemStyle: { color: theme.keep, borderRadius: [999, 0, 0, 999] },
        emphasis: { focus: "series" },
        data: data.map((item) => Number(item.keepPct.toFixed(2)))
      },
      {
        name: translateDecision("MAYBE", intl.t),
        type: "bar" as const,
        stack: "composition",
        barWidth: 18,
        itemStyle: { color: theme.maybe },
        emphasis: { focus: "series" },
        data: data.map((item) => Number(item.maybePct.toFixed(2)))
      },
      {
        name: translateDecision("DELETE", intl.t),
        type: "bar" as const,
        stack: "composition",
        barWidth: 18,
        itemStyle: { color: theme.danger, borderRadius: [0, 999, 999, 0] },
        emphasis: { focus: "series" },
        data: data.map((item) => Number(item.deletePct.toFixed(2)))
      },
      {
        name: intl.t("chart.metric.average_imdb_per_gb"),
        type: "scatter" as const,
        xAxisIndex: 1,
        symbol: "diamond",
        symbolSize: 18,
        itemStyle: {
          color: theme.accent,
          borderColor: theme.panel,
          borderWidth: 2
        },
        label: {
          show: true,
          position: "right",
          color: theme.text,
          formatter: (params: { value?: unknown }) =>
            formatValue(Number(params.value ?? 0), intl.locale, 3)
        },
        data: data.map((item) => Number(item.metric.toFixed(3))),
        z: 4
      }
    ]
  };
}

function spaceByLibrary(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const grouped = new Map<string, { KEEP: number; MAYBE: number; DELETE: number }>();
  rows.forEach((row) => {
    const library = String(row.library || "").trim();
    if (!library) {
      return;
    }
    const current = grouped.get(library) ?? { KEEP: 0, MAYBE: 0, DELETE: 0 };
    const size = parseMaybeNumber(row.file_size_gb) ?? 0;
    const decision = decisionKey(row);
    if (decision === "KEEP" || decision === "MAYBE" || decision === "DELETE") {
      current[decision] += size;
    }
    grouped.set(library, current);
  });
  const libraries = Array.from(grouped.entries())
    .map(([library, value]) => ({
      library,
      keep: Number(value.KEEP.toFixed(1)),
      maybe: Number(value.MAYBE.toFixed(1)),
      delete: Number(value.DELETE.toFixed(1)),
      total: Number((value.KEEP + value.MAYBE + value.DELETE).toFixed(1))
    }))
    .sort((left, right) => right.total - left.total)
    .slice(0, 12);
  const series = ["KEEP", "MAYBE", "DELETE"].map((decision) => ({
    name: translateDecision(decision, intl.t),
    type: "bar" as const,
    stack: "size",
    barWidth: 18,
    itemStyle: {
      color: colors[decision],
      borderRadius:
        decision === "KEEP"
          ? [999, 0, 0, 999]
          : decision === "DELETE"
            ? [0, 999, 999, 0]
            : 0
    },
    data: libraries.map((library) =>
      decision === "KEEP"
        ? library.keep
        : decision === "MAYBE"
          ? library.maybe
          : library.delete
    )
  }));
  if (!libraries.length) {
    return emptyChartOption(intl);
  }
  const rowByLibrary = new Map(libraries.map((item) => [item.library, item]));

  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const items = Array.isArray(params) ? params : [params];
        const axisValue = items[0] && typeof items[0] === "object" && "axisValue" in items[0]
          ? String(items[0].axisValue)
          : "";
        const row = rowByLibrary.get(axisValue);
        if (!row) {
          return intl.t("chart.no_data");
        }
        return tooltipCard(theme, row.library, [
          {
            label: translateDecision("KEEP", intl.t),
            value: `${formatValue(row.keep, intl.locale, 1)} GB`,
            color: colors.KEEP
          },
          {
            label: translateDecision("MAYBE", intl.t),
            value: `${formatValue(row.maybe, intl.locale, 1)} GB`,
            color: colors.MAYBE
          },
          {
            label: translateDecision("DELETE", intl.t),
            value: `${formatValue(row.delete, intl.locale, 1)} GB`,
            color: colors.DELETE
          },
          { label: intl.t("chart.metric.gb"), value: formatValue(row.total, intl.locale, 1) }
        ]);
      }
    },
    xAxis: { type: "value", name: intl.t("chart.metric.gb") },
    yAxis: {
      type: "category",
      inverse: true,
      data: libraries.map((item) => item.library),
      axisLabel: { color: theme.muted, formatter: (value: string) => truncateLabel(value, 20) }
    },
    series
  };
}

function decadeDistribution(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const decades = Array.from(
    new Set(rows.map((row) => row.decade).filter((value): value is number => typeof value === "number"))
  ).sort((left, right) => left - right);
  const decadeRows = decades.map((decade) => {
    const keep = rows.filter((row) => row.decade === decade && decisionKey(row) === "KEEP").length;
    const maybe = rows.filter((row) => row.decade === decade && decisionKey(row) === "MAYBE").length;
    const deleteCount = rows.filter((row) => row.decade === decade && decisionKey(row) === "DELETE").length;
    return {
      label: intl.t("data.decade_label", { decade }),
      keep,
      maybe,
      delete: deleteCount,
      total: keep + maybe + deleteCount
    };
  });
  const series = ["KEEP", "MAYBE", "DELETE"].map((decision) => ({
    name: translateDecision(decision, intl.t),
    type: "bar" as const,
    stack: "count",
    barWidth: 22,
    itemStyle: {
      color: colors[decision],
      borderRadius:
        decision === "KEEP"
          ? [999, 999, 0, 0]
          : decision === "DELETE"
            ? [0, 0, 999, 999]
            : 0
    },
    emphasis: { focus: "series" as const },
    data: decadeRows.map((item) =>
      decision === "KEEP" ? item.keep : decision === "MAYBE" ? item.maybe : item.delete
    )
  }));
  if (!decades.length) {
    return emptyChartOption(intl);
  }
  const rowByLabel = new Map(decadeRows.map((item) => [item.label, item]));
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const items = Array.isArray(params) ? params : [params];
        const axisValue = items[0] && typeof items[0] === "object" && "axisValue" in items[0]
          ? String(items[0].axisValue)
          : "";
        const row = rowByLabel.get(axisValue);
        if (!row) {
          return intl.t("chart.no_data");
        }
        return tooltipCard(theme, row.label, [
          {
            label: translateDecision("KEEP", intl.t),
            value: formatValue(row.keep, intl.locale, 0),
            color: colors.KEEP
          },
          {
            label: translateDecision("MAYBE", intl.t),
            value: formatValue(row.maybe, intl.locale, 0),
            color: colors.MAYBE
          },
          {
            label: translateDecision("DELETE", intl.t),
            value: formatValue(row.delete, intl.locale, 0),
            color: colors.DELETE
          },
          {
            label: intl.t("chart.metric.titles"),
            value: formatValue(row.total, intl.locale, 0)
          }
        ]);
      }
    },
    xAxis: {
      type: "category",
      data: decadeRows.map((item) => item.label),
      axisLabel: { color: theme.muted, interval: axisInterval(decades.length, 10) }
    },
    yAxis: {
      type: "value",
      name: intl.t("chart.metric.titles"),
      nameTextStyle: { color: theme.muted, padding: [0, 0, 0, 8] }
    },
    series
  };
}

function genreDistribution(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const colors = decisionColors(theme);
  const genres = collectGenres(rows);
  if (!genres.length) {
    return emptyChartOption(intl);
  }
  return {
    ...baseChart(),
    legend: { top: 0, textStyle: { color: theme.muted, fontSize: 11 } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const items = Array.isArray(params) ? params : [params];
        const axisValue = items[0] && typeof items[0] === "object" && "axisValue" in items[0]
          ? String(items[0].axisValue)
          : "";
        const row = genres.find((item) => item.genre === axisValue);
        if (!row) {
          return intl.t("chart.no_data");
        }
        return tooltipCard(theme, row.genre, [
          {
            label: translateDecision("KEEP", intl.t),
            value: formatValue(row.keep, intl.locale, 0),
            color: colors.KEEP
          },
          {
            label: translateDecision("MAYBE", intl.t),
            value: formatValue(row.maybe, intl.locale, 0),
            color: colors.MAYBE
          },
          {
            label: translateDecision("DELETE", intl.t),
            value: formatValue(row.delete, intl.locale, 0),
            color: colors.DELETE
          },
          {
            label: intl.t("chart.metric.titles"),
            value: formatValue(row.total, intl.locale, 0)
          }
        ]);
      }
    },
    xAxis: {
      type: "value",
      name: intl.t("chart.metric.titles"),
      nameTextStyle: { color: theme.muted, padding: [0, 0, 8, 0] }
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: genres.map((item) => item.genre),
      axisLabel: { color: theme.muted, formatter: (value: string) => truncateLabel(value, 20) }
    },
    series: [
      {
        name: translateDecision("KEEP", intl.t),
        type: "bar" as const,
        stack: "genre",
        barWidth: 18,
        itemStyle: { color: colors.KEEP, borderRadius: [999, 0, 0, 999] },
        emphasis: { focus: "series" },
        data: genres.map((item) => item.keep)
      },
      {
        name: translateDecision("MAYBE", intl.t),
        type: "bar" as const,
        stack: "genre",
        barWidth: 18,
        itemStyle: { color: colors.MAYBE },
        emphasis: { focus: "series" },
        data: genres.map((item) => item.maybe)
      },
      {
        name: translateDecision("DELETE", intl.t),
        type: "bar" as const,
        stack: "genre",
        barWidth: 18,
        itemStyle: { color: colors.DELETE, borderRadius: [0, 999, 999, 0] },
        emphasis: { focus: "series" },
        data: genres.map((item) => item.delete)
      }
    ]
  };
}

function directorRanking(rows: ReportRow[], intl: ChartIntl): EChartsOption {
  const theme = chartTheme();
  const directors = collectDirectors(rows);
  if (!directors.length) {
    return emptyChartOption(intl);
  }
  return {
    ...baseChart(),
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const currentParam = firstTooltipParam(params);
        const dataIndex =
          currentParam && typeof currentParam === "object" && "dataIndex" in currentParam
            ? Number(currentParam.dataIndex ?? 0)
            : 0;
        const name =
          currentParam && typeof currentParam === "object" && "name" in currentParam
            ? String(currentParam.name || "")
            : "";
        const value =
          currentParam && typeof currentParam === "object" && "value" in currentParam
            ? Number(currentParam.value ?? 0)
            : 0;
        const current = directors[dataIndex];
        return tooltipCard(theme, name, [
          {
            label: intl.t("chart.metric.average_imdb"),
            value: formatValue(value, intl.locale, 2),
            color: theme.accent2
          },
          {
            label: intl.t("column.count"),
            value: formatValue(current?.total ?? 0, intl.locale, 0)
          }
        ]);
      }
    },
    xAxis: {
      type: "value",
      name: intl.t("chart.metric.average_imdb"),
      nameTextStyle: { color: theme.muted, padding: [0, 0, 8, 0] }
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: directors.map((item) => item.director),
      axisLabel: { color: theme.muted, formatter: (value: string) => truncateLabel(value, 22) }
    },
    series: [
      {
        type: "bar" as const,
        barWidth: 18,
        itemStyle: { color: theme.accent2, borderRadius: 999 },
        label: {
          show: true,
          position: "right",
          color: theme.text,
          formatter: (params: { value?: unknown }) =>
            formatValue(Number(params.value ?? 0), intl.locale, 2)
        },
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
  if (!words.length) {
    return emptyChartOption(intl);
  }
  return {
    ...baseChart(),
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const current = firstTooltipParam(params);
        const name =
          current && typeof current === "object" && "name" in current ? String(current.name || "") : "";
        const value =
          current && typeof current === "object" && "value" in current ? Number(current.value ?? 0) : 0;
        return tooltipCard(theme, name, [
          {
            label: intl.t("chart.metric.frequency"),
            value: formatValue(value, intl.locale, 0),
            color: theme.maybe
          }
        ]);
      }
    },
    xAxis: {
      type: "value",
      name: intl.t("chart.metric.frequency"),
      nameTextStyle: { color: theme.muted, padding: [0, 0, 8, 0] }
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: words.map((item) => item.word),
      axisLabel: { color: theme.muted, formatter: (value: string) => truncateLabel(value, 18) }
    },
    series: [
      {
        type: "bar" as const,
        barWidth: 18,
        itemStyle: { color: theme.maybe, borderRadius: 999 },
        label: {
          show: true,
          position: "right",
          color: theme.text,
          formatter: (params: { value?: unknown }) =>
            formatValue(Number(params.value ?? 0), intl.locale, 0)
        },
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
      count: bucket.length,
      name: translateDecision(decision, intl.t),
      value: Number(average.toFixed(2)),
      itemStyle: { color: colors[decision] }
    };
  });
  if (!data.some((item) => item.value > 0)) {
    return emptyChartOption(intl);
  }
  return {
    ...baseChart(),
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const currentParam = firstTooltipParam(params);
        const dataIndex =
          currentParam && typeof currentParam === "object" && "dataIndex" in currentParam
            ? Number(currentParam.dataIndex ?? 0)
            : 0;
        const name =
          currentParam && typeof currentParam === "object" && "name" in currentParam
            ? String(currentParam.name || "")
            : "";
        const value =
          currentParam && typeof currentParam === "object" && "value" in currentParam
            ? Number(currentParam.value ?? 0)
            : 0;
        const current = data[dataIndex];
        return tooltipCard(theme, name, [
          {
            label: intl.t("chart.metric.average_imdb"),
            value: formatValue(value, intl.locale, 2),
            color:
              typeof current?.itemStyle?.color === "string"
                ? current.itemStyle.color
                : undefined
          },
          {
            label: intl.t("column.count"),
            value: formatValue(current?.count ?? 0, intl.locale, 0)
          }
        ]);
      }
    },
    xAxis: {
      type: "category",
      data: data.map((item) => item.name),
      axisLabel: { color: theme.muted }
    },
    yAxis: {
      type: "value",
      name: intl.t("chart.metric.average_imdb"),
      nameTextStyle: { color: theme.muted, padding: [0, 0, 0, 8] }
    },
    series: [
      {
        type: "bar" as const,
        barWidth: 28,
        label: {
          show: true,
          position: "top",
          color: theme.text,
          formatter: (params: { value?: unknown }) =>
            formatValue(Number(params.value ?? 0), intl.locale, 2)
        },
        data: data.map((item) => ({
          ...item,
          itemStyle: {
            color: item.itemStyle.color,
            borderRadius: [14, 14, 0, 0]
          }
        }))
      }
    ]
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
