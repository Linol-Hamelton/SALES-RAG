import { access } from "fs/promises";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { readCsv } from "./RAG_ANALYTICS/lib/io.mjs";

const ROOT_DIR = dirname(fileURLToPath(import.meta.url));
const expected = {
  raw: {
    goods: 9095,
    offers: 11674,
    orders: 50404,
  },
  ordersDirectionFilledRows: 44742,
};

const criticalFiles = [
  "Bitrix24Webhook.mjs",
  "CurMonDat.mjs",
  "batchUtils.mjs",
  "refreshSalesRagData.mjs",
  "RAG_DATA/goods.csv",
  "RAG_DATA/offers.csv",
  "RAG_DATA/orders.csv",
  "RAG_DATA/csvUtils.mjs",
  "RAG_DATA/bitrixClient.mjs",
  "RAG_DATA/getDealsByStage.mjs",
  "RAG_DATA/getDealProductRows.mjs",
  "RAG_DATA/getFullCatalog.mjs",
  "RAG_DATA/generateRagData.mjs",
  "RAG_DATA/repairOrdersDirection.mjs",
  "RAG_ANALYTICS/lib/common.mjs",
  "RAG_ANALYTICS/lib/io.mjs",
  "RAG_ANALYTICS/lib/pipeline.mjs",
  "RAG_ANALYTICS/buildQaAndNormalizedData.mjs",
  "RAG_ANALYTICS/buildFacts.mjs",
  "RAG_ANALYTICS/buildPricing.mjs",
  "RAG_ANALYTICS/buildKpis.mjs",
  "RAG_ANALYTICS/runAll.mjs",
  "RAG_ANALYTICS/output/qa_report.json",
  "RAG_ANALYTICS/output/facts/product_facts.csv",
  "RAG_ANALYTICS/output/pricing/pricing_recommendations.csv",
  "RAG_ANALYTICS/output/kpis/kpi_summary.json"
];

async function exists(relativePath) {
  try {
    await access(resolve(ROOT_DIR, relativePath));
    return true;
  } catch {
    return false;
  }
}

for (const file of criticalFiles) {
  if (!(await exists(file))) {
    console.error(`Missing critical file: ${file}`);
    process.exit(1);
  }
}

const majorNodeVersion = Number(process.versions.node.split(".")[0]);
if (!Number.isFinite(majorNodeVersion) || majorNodeVersion < 18) {
  console.error(`Node.js 18+ is required. Current version: ${process.versions.node}`);
  process.exit(1);
}

if (typeof fetch !== "function") {
  console.error("Global fetch is not available. Use Node.js 18+.");
  process.exit(1);
}

const goods = await readCsv(resolve(ROOT_DIR, "RAG_DATA/goods.csv"));
const offers = await readCsv(resolve(ROOT_DIR, "RAG_DATA/offers.csv"));
const orders = await readCsv(resolve(ROOT_DIR, "RAG_DATA/orders.csv"));

const checks = {
  goodsRows: goods.length,
  offersRows: offers.length,
  ordersRows: orders.length,
  ordersHasDirectionColumn: orders.length ? Object.keys(orders[0]).includes("Направление") : false,
  ordersDirectionFilledRows: orders.filter((row) => String(row["Направление"] ?? "").trim() !== "").length,
  nodeVersion: process.versions.node,
  bitrixWebhookOverrideSet: Boolean(process.env.BITRIX24_WEBHOOK_URL),
};

const failures = [];
if (checks.goodsRows !== expected.raw.goods) failures.push(`goods row count mismatch: ${checks.goodsRows}`);
if (checks.offersRows !== expected.raw.offers) failures.push(`offers row count mismatch: ${checks.offersRows}`);
if (checks.ordersRows !== expected.raw.orders) failures.push(`orders row count mismatch: ${checks.ordersRows}`);
if (!checks.ordersHasDirectionColumn) failures.push("orders.csv missing direction column");
if (checks.ordersDirectionFilledRows !== expected.ordersDirectionFilledRows) failures.push(`orders direction filled rows mismatch: ${checks.ordersDirectionFilledRows}`);

if (failures.length) {
  console.error("SALES_RAG bundle verification failed.");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log("SALES_RAG bundle verification passed.");
console.log(JSON.stringify(checks, null, 2));
