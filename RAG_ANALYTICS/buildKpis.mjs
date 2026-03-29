import { runAnalyticsPipeline } from "./lib/pipeline.mjs";

const result = await runAnalyticsPipeline({ scope: "kpis" });
console.log(`KPI outputs created. Orders: ${result.kpis.kpiSummary.orderDeals}, offers: ${result.kpis.kpiSummary.offerDeals}`);
