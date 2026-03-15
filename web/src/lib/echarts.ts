import * as echarts from "echarts/core";
import { BarChart, BoxplotChart, PieChart, ScatterChart } from "echarts/charts";
import {
  DatasetComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  TransformComponent
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  BoxplotChart,
  PieChart,
  ScatterChart,
  DatasetComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  TransformComponent,
  CanvasRenderer
]);

export { echarts };
