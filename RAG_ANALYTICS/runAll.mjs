import { runAnalyticsPipeline } from "./lib/pipeline.mjs";

const result = await runAnalyticsPipeline({ scope: "all" });
console.log(`Analytics pipeline completed. QA issues: ${result.qaReport.issues.length}.`);
console.log(`Pricing rows: ${result.pricingFacts.length}. KPI monthly rows: ${result.kpis.monthlyDirectionKpis.length}.`);
