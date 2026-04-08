import { runAnalyticsPipeline } from "./lib/pipeline.mjs";
import { spawnSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const result = await runAnalyticsPipeline({ scope: "all" });
console.log(`Analytics pipeline completed. QA issues: ${result.qaReport.issues.length}.`);
console.log(`Pricing rows: ${result.pricingFacts.length}. KPI monthly rows: ${result.kpis.monthlyDirectionKpis.length}.`);

// Build smeta templates (П7): categories + canonical smeta + price stats
console.log("\nBuilding smeta templates...");
const __dirname = dirname(fileURLToPath(import.meta.url));
const smetaResult = spawnSync("node", [join(__dirname, "buildSmetaTemplates.mjs")], {
  stdio: "inherit",
});
if (smetaResult.status !== 0) {
  console.error("buildSmetaTemplates failed with code", smetaResult.status);
  process.exit(smetaResult.status || 1);
}
