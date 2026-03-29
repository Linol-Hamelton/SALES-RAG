import { runAnalyticsPipeline } from "./lib/pipeline.mjs";

const result = await runAnalyticsPipeline({ scope: "pricing" });
console.log(`Pricing recommendations created for ${result.pricingFacts.length} products/signatures.`);
console.log(`Auto: ${result.pricingSummary.autoProducts}, guided: ${result.pricingSummary.guidedProducts}, manual: ${result.pricingSummary.manualProducts}`);
