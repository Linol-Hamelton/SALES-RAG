import { fileURLToPath } from "url";
import { writeFile } from "fs/promises";
import { CSV_HEADERS, withSchema, writeCsvAtomically } from "./csvUtils.mjs";
import getDealsByStage from "./getDealsByStage.mjs";
import getDealProductRows from "./getDealProductRows.mjs";
import getFullCatalog from "./getFullCatalog.mjs";

const OFFERS_STAGE_ID = process.env.BITRIX24_OFFERS_STAGE_ID || "UC_Y55PKR";
const ORDERS_STAGE_ID = process.env.BITRIX24_ORDERS_STAGE_ID || "WON";

function outputPath(fileName) {
  return fileURLToPath(new URL(`./${fileName}`, import.meta.url));
}

function buildCatalogMap(catalogItems) {
  return new Map(catalogItems.map((item) => [String(item.ID), item]));
}

function groupRowsByDealId(productRows) {
  const rowsByDealId = new Map();

  for (const row of productRows) {
    const dealId = String(row.ID ?? "");

    if (!rowsByDealId.has(dealId)) {
      rowsByDealId.set(dealId, []);
    }

    rowsByDealId.get(dealId).push(row);
  }

  return rowsByDealId;
}

function buildGoodsRows(catalogItems) {
  return catalogItems.map((item) =>
    withSchema({
      DATASET: "goods",
      PRODUCT_ID: item.ID ?? "",
      PRODUCT_NAME: item.NAME ?? "",
      BASE_PRICE: item.BASE_PRICE ?? "",
      CATALOG_ID: item.CATALOG_ID ?? "",
      SECTION_ID: item.SECTION_ID ?? "",
      SECTION_NAME: item.SECTION_NAME ?? "",
      PARENT_SECTION: item.PARENT_SECTION ?? "",
      NAME: item.NAME ?? "",
      "Направление": item["Направление"] ?? "",
      PRODUCT_DESCRIPTION: item.PRODUCT_DESCRIPTION ?? "",
      COEFFICIENT: item.COEFFICIENT ?? "",
      WORKSHOP_SALARY: item.WORKSHOP_SALARY ?? "",
      COST_PRICE: item.COST_PRICE ?? "",
      EXECUTOR: item.EXECUTOR ?? "",
    })
  );
}

function buildDealBaseRow(dataset, deal) {
  return {
    DATASET: dataset,
    ID: deal.ID ?? "",
    TITLE: deal.TITLE ?? "",
    OPPORTUNITY: deal.OPPORTUNITY ?? "",
    COMPANY_ID: deal.COMPANY_ID ?? "",
    CONTACT_ID: deal.CONTACT_ID ?? "",
    BEGINDATE: deal.BEGINDATE ?? "",
    CLOSEDATE: deal.CLOSEDATE ?? "",
    DESCRIPTION: deal.DESCRIPTION ?? "",
    COMMENTS: deal.COMMENTS ?? "",
    DIRECTION: deal.DIRECTION ?? "",
    SIGN_TYPE: deal.SIGN_TYPE ?? "",
    PAYMENT_TYPE: deal.PAYMENT_TYPE ?? "",
    SOURCE_ID: deal.SOURCE_ID ?? "",
    IS_RETURN_CUSTOMER: deal.IS_RETURN_CUSTOMER ?? "",
  };
}

function buildDealRows(dataset, deals, productRows, catalogMap) {
  const rowsByDealId = groupRowsByDealId(productRows);
  const result = [];

  for (const deal of deals) {
    const dealId = String(deal.ID ?? "");
    const baseRow = buildDealBaseRow(dataset, deal);
    const dealProductRows = rowsByDealId.get(dealId) ?? [];

    if (dealProductRows.length === 0) {
      result.push(withSchema(baseRow));
      continue;
    }

    for (const productRow of dealProductRows) {
      const catalogItem = catalogMap.get(String(productRow.PRODUCT_ID ?? "")) ?? {};

      result.push(
        withSchema({
          ...baseRow,
          GOOD_ID: productRow.GOOD_ID ?? "",
          PRODUCT_ID: productRow.PRODUCT_ID ?? "",
          PRODUCT_NAME: productRow.PRODUCT_NAME ?? "",
          PRICE: productRow.PRICE ?? "",
          PRICE_ACCOUNT: productRow.PRICE_ACCOUNT ?? "",
          QUANTITY: productRow.QUANTITY ?? "",
          BASE_PRICE: catalogItem.BASE_PRICE ?? "",
          CATALOG_ID: catalogItem.CATALOG_ID ?? "",
          SECTION_ID: catalogItem.SECTION_ID ?? "",
          SECTION_NAME: catalogItem.SECTION_NAME ?? "",
          PARENT_SECTION: catalogItem.PARENT_SECTION ?? "",
          NAME: catalogItem.NAME ?? "",
          "Направление": catalogItem["Направление"] ?? "",
          PRODUCT_DESCRIPTION: catalogItem.PRODUCT_DESCRIPTION ?? "",
          COEFFICIENT: catalogItem.COEFFICIENT ?? "",
          WORKSHOP_SALARY: catalogItem.WORKSHOP_SALARY ?? "",
          COST_PRICE: catalogItem.COST_PRICE ?? "",
          EXECUTOR: catalogItem.EXECUTOR ?? "",
        })
      );
    }
  }

  return result;
}

async function writeDataset(fileName, rows) {
  await writeCsvAtomically(outputPath(fileName), rows, CSV_HEADERS);
  console.log(`Created ${fileName}: ${rows.length} rows`);
}

async function exportDealDataset({ dataset, stageId, fileName, catalogMap }) {
  console.log(`Collecting ${dataset} from stage ${stageId}...`);

  const deals = await getDealsByStage(stageId);
  const dealIds = deals.map((deal) => deal.ID).filter(Boolean);
  const productRows = await getDealProductRows(dealIds);
  const rows = buildDealRows(dataset, deals, productRows, catalogMap);

  await writeDataset(fileName, rows);
  return deals;
}

async function main() {
  console.log("Collecting full catalog...");
  const catalogItems = await getFullCatalog();
  const catalogMap = buildCatalogMap(catalogItems);

  await writeDataset("goods.csv", buildGoodsRows(catalogItems));
  const offersDeals = await exportDealDataset({
    dataset: "offers",
    stageId: OFFERS_STAGE_ID,
    fileName: "offers.csv",
    catalogMap,
  });
  const ordersDeals = await exportDealDataset({
    dataset: "orders",
    stageId: ORDERS_STAGE_ID,
    fileName: "orders.csv",
    catalogMap,
  });

  // Export deals.json for image/vision pipeline
  const allDeals = [...offersDeals, ...ordersDeals];
  const dealsForVision = allDeals.map(d => ({
    ID: d.ID,
    TITLE: d.TITLE,
    DESCRIPTION: d.DESCRIPTION,
    DIRECTION: d.DIRECTION,
    IMAGE_URLS: d.IMAGE_URLS,
  }));
  const withImages = dealsForVision.filter(d => d.IMAGE_URLS.length > 0).length;
  await writeFile(outputPath("deals.json"), JSON.stringify(dealsForVision, null, 2), "utf-8");
  console.log(`Created deals.json: ${dealsForVision.length} deals (${withImages} with images)`);

  console.log("RAG export completed.");
}

main().catch((error) => {
  console.error(`RAG export failed: ${error.message}`);
  process.exit(1);
});
