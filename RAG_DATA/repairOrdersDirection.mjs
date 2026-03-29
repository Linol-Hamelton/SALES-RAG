import { repairOrdersDirectionFile } from "../RAG_ANALYTICS/lib/pipeline.mjs";

const result = await repairOrdersDirectionFile();
console.log(`Orders direction repair completed. Fixed rows: ${result.fixedRows}`);
