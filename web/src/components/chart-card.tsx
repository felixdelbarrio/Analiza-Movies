import type { EChartsOption } from "echarts";
import ReactEChartsCore from "echarts-for-react/lib/core";

import { echarts } from "../lib/echarts";
import { SectionCard } from "./section-card";

interface ChartCardProps {
  title: string;
  eyebrow?: string;
  subtitle?: string;
  option: EChartsOption;
  height?: number;
}

export function ChartCard({
  title,
  eyebrow,
  subtitle,
  option,
  height = 360
}: ChartCardProps) {
  return (
    <SectionCard
      title={title}
      eyebrow={eyebrow}
      className="chart-card"
      actions={subtitle ? <p className="chart-card__subtitle">{subtitle}</p> : null}
    >
      <ReactEChartsCore
        echarts={echarts}
        lazyUpdate
        notMerge
        option={option}
        opts={{ renderer: "canvas" }}
        style={{ height }}
      />
    </SectionCard>
  );
}
