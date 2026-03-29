import { runAnalyticsPipeline } from "./lib/pipeline.mjs";

const result = await runAnalyticsPipeline({ scope: "facts" });
console.log(`Fact tables created: ${result.productFacts.length} product facts, ${result.dealFacts.length} deal facts.`);
