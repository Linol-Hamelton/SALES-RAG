import { runAnalyticsPipeline } from "./lib/pipeline.mjs";

const result = await runAnalyticsPipeline({ scope: "qa" });
console.log(`QA report created with ${result.qaReport.issues.length} issue(s).`);
console.log(`Orders direction rows repaired: ${result.qaReport.repairInfo.repairedOrdersDirectionRows}`);
