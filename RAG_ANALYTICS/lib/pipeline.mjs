
import { resolve } from "path";
import {
  RAW_DATASET_FILES,
  RAW_HEADERS,
  NORMALIZED_HEADERS,
  OUTPUT_DIR,
  NORMALIZED_DIR,
  FACTS_DIR,
  PRICING_DIR,
  KPI_DIR,
  MISSING_CLIENT,
  MISSING_DIRECTION,
  MANUAL_DIRECTION,
  NOT_APPLICABLE,
  roundNumber,
  parseNumber,
  average,
  median,
  percentile,
  unique,
  cleanText,
  sanitizePlaceholder,
  buildProductKey,
  pickClient,
  getCloseMonth,
  getCloseYear,
  sumLineTotal,
  getPrimaryDirection,
  makeBundleKey,
  isFinancialModifier,
  overlapScore,
  tokenizeText,
  getDealDurationDays,
  extractMaterials,
  parseDate,
} from "./common.mjs";
import { ensureDirectories, readCsv, writeCsv, writeJsonAtomically } from "./io.mjs";

function groupBy(rows, keyFn) {
  const groups = new Map();

  for (const row of rows) {
    const key = keyFn(row);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(row);
  }

  return groups;
}

function firstNonEmpty(...values) {
  for (const value of values) {
    const text = cleanText(value);
    if (text) {
      return text;
    }
  }
  return "";
}

function extractHeightFromText(text) {
  // Extract letter height (cm) from free text. Handles см/cm, мм/mm, м/m.
  if (!text) return [];
  const out = [];
  const clean = String(text).toLowerCase();
  const reCm = /(\d{2,3})\s*(?:см|cm)\b/g;
  const reMm = /(\d{2,4})\s*(?:мм|mm)\b/g;
  const reM = /(\d(?:[.,]\d)?)\s*(?:м|m)\b(?!м)/g;
  let m;
  while ((m = reCm.exec(clean))) {
    const h = parseInt(m[1], 10);
    if (h >= 10 && h <= 200) out.push(h);
  }
  while ((m = reMm.exec(clean))) {
    const h = Math.round(parseInt(m[1], 10) / 10);
    if (h >= 10 && h <= 200) out.push(h);
  }
  while ((m = reM.exec(clean))) {
    const h = Math.round(parseFloat(m[1].replace(",", ".")) * 100);
    if (h >= 10 && h <= 200) out.push(h);
  }
  return out;
}

function extractHeightFromProducts(rows, fallbackText = "") {
  // Extract letter height (cm) from product names; fallback to deal title/description.
  const heights = [];
  for (const row of rows) {
    const name = cleanText(row.PRODUCT_NAME || row.NAME || row.PRODUCT_NAME_NORMALIZED || "");
    heights.push(...extractHeightFromText(name));
  }
  if (!heights.length && fallbackText) {
    heights.push(...extractHeightFromText(fallbackText));
  }
  if (!heights.length) return null;
  const freq = new Map();
  for (const h of heights) freq.set(h, (freq.get(h) ?? 0) + 1);
  return [...freq.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function createGoodsMap(goodsRows) {
  const map = new Map();

  for (const row of goodsRows) {
    const productId = cleanText(row.PRODUCT_ID);
    if (!productId) {
      continue;
    }

    map.set(productId, row);
  }

  return map;
}

function createGoodsNameIndex(goodsRows) {
  return goodsRows
    .map((row) => {
      const productId = cleanText(row.PRODUCT_ID);
      const name = firstNonEmpty(row.NAME, row.PRODUCT_NAME);
      const tokens = tokenizeText(name);

      if (!productId || !name || !tokens.length) {
        return null;
      }

      return {
        productId,
        name,
        tokens,
        direction: cleanText(row["Направление"]),
        basePrice: parseNumber(row.BASE_PRICE),
      };
    })
    .filter(Boolean);
}

function createGoodsAnalogIndex(goodsRows) {
  const records = createGoodsNameIndex(goodsRows);
  const tokenMap = new Map();

  for (const record of records) {
    for (const token of record.tokens) {
      if (!tokenMap.has(token)) {
        tokenMap.set(token, []);
      }
      tokenMap.get(token).push(record);
    }
  }

  return { records, tokenMap };
}

async function loadDataset(filePath, name) {
  const rows = await readCsv(filePath);
  const headers = rows.length ? Object.keys(rows[0]) : [...RAW_HEADERS];

  return {
    name,
    filePath,
    headers,
    rows,
  };
}

export async function loadRawDatasets() {
  const goods = await loadDataset(RAW_DATASET_FILES.goods, "goods");
  const offers = await loadDataset(RAW_DATASET_FILES.offers, "offers");
  const orders = await loadDataset(RAW_DATASET_FILES.orders, "orders");

  return { goods, offers, orders };
}

export function repairOrdersDirectionRows(goodsRows, ordersRows) {
  const goodsMap = createGoodsMap(goodsRows);
  let fixedRows = 0;

  const repairedRows = ordersRows.map((row) => {
    const currentDirection = cleanText(row["Направление"]);
    const productId = cleanText(row.PRODUCT_ID);

    if (currentDirection || !productId || productId === "0") {
      return row;
    }

    const goodsRow = goodsMap.get(productId);
    const repairedDirection = cleanText(goodsRow?.["Направление"]);

    if (!repairedDirection) {
      return row;
    }

    fixedRows += 1;
    return {
      ...row,
      "Направление": repairedDirection,
    };
  });

  return { repairedRows, fixedRows };
}

export async function repairOrdersDirectionFile(rawDatasets = null) {
  const datasets = rawDatasets ?? await loadRawDatasets();
  const { repairedRows, fixedRows } = repairOrdersDirectionRows(
    datasets.goods.rows,
    datasets.orders.rows
  );

  if (fixedRows > 0) {
    await writeCsv(RAW_DATASET_FILES.orders, repairedRows, RAW_HEADERS);
  }

  return {
    datasets: {
      ...datasets,
      orders: {
        ...datasets.orders,
        rows: repairedRows,
      },
    },
    fixedRows,
  };
}

function getCatalogMatchStatus(row, goodsMap) {
  const productId = cleanText(row.PRODUCT_ID);
  if (!productId || productId === "0") {
    return "manual_row";
  }

  return goodsMap.has(productId) ? "catalog" : "missing_catalog";
}

function normalizeGoodsRows(goodsRows) {
  return goodsRows.map((row) => {
    const direction = cleanText(row["Направление"]) || MISSING_DIRECTION;
    return {
      ...row,
      CLIENT_SOURCE: NOT_APPLICABLE,
      CLIENT_LABEL: NOT_APPLICABLE,
      CLIENT_KEY: NOT_APPLICABLE,
      CLIENT_CONTEXT_STATUS: NOT_APPLICABLE,
      PRODUCT_KEY: cleanText(row.PRODUCT_ID),
      CATALOG_MATCH_STATUS: "catalog",
      PRODUCT_NAME_NORMALIZED: firstNonEmpty(row.NAME, row.PRODUCT_NAME),
      LINE_TOTAL: "",
      PRICE_NUM: "",
      BASE_PRICE_NUM: parseNumber(row.BASE_PRICE) ?? "",
      QUANTITY_NUM: "",
      PRICE_TO_BASE_RATIO: "",
      NORMALIZED_DIRECTION: direction,
      DIRECTION_SOURCE: cleanText(row["Направление"]) ? "raw" : "missing",
      CLOSE_MONTH: "",
      CLOSE_YEAR: "",
      SECTION_NAME: cleanText(row.SECTION_NAME) || "",
      PARENT_SECTION: cleanText(row.PARENT_SECTION) || "",
      COST_PRICE_NUM: parseNumber(row.COST_PRICE) ?? "",
      COEFFICIENT_NUM: parseNumber(row.COEFFICIENT) ?? "",
    };
  });
}
function normalizeDealRows(rows, goodsMap) {
  return rows.map((row) => {
    const client = pickClient(row);
    const productId = cleanText(row.PRODUCT_ID);
    const priceNum = parseNumber(row.PRICE);
    const basePriceNum = parseNumber(row.BASE_PRICE);
    const quantityNum = parseNumber(row.QUANTITY);
    const lineTotal = priceNum !== null && quantityNum !== null ? priceNum * quantityNum : null;
    const goodsRow = goodsMap.get(productId);
    const rawDirection = cleanText(row["Направление"]);
    const catalogDirection = cleanText(goodsRow?.["Направление"]);
    const directionSource = rawDirection
      ? "raw"
      : catalogDirection
        ? "catalog_fallback"
        : productId && productId !== "0"
          ? "missing"
          : "manual";
    const normalizedDirection = rawDirection || catalogDirection || (productId && productId !== "0" ? MISSING_DIRECTION : MANUAL_DIRECTION);
    const ratio = priceNum !== null && basePriceNum !== null && basePriceNum !== 0 ? priceNum / basePriceNum : null;

    return {
      ...row,
      COMPANY_ID: client.companyId,
      CONTACT_ID: client.contactId,
      CLIENT_SOURCE: client.clientSource,
      CLIENT_LABEL: client.clientLabel,
      CLIENT_KEY: client.clientKey,
      CLIENT_CONTEXT_STATUS: client.clientStatus,
      PRODUCT_KEY: buildProductKey(row),
      CATALOG_MATCH_STATUS: getCatalogMatchStatus(row, goodsMap),
      PRODUCT_NAME_NORMALIZED: firstNonEmpty(row.PRODUCT_NAME, row.NAME, row.TITLE),
      LINE_TOTAL: lineTotal !== null ? roundNumber(lineTotal, 2) : "",
      PRICE_NUM: priceNum ?? "",
      BASE_PRICE_NUM: basePriceNum ?? "",
      QUANTITY_NUM: quantityNum ?? "",
      PRICE_TO_BASE_RATIO: ratio !== null ? roundNumber(ratio, 4) : "",
      NORMALIZED_DIRECTION: normalizedDirection,
      DIRECTION_SOURCE: directionSource,
      CLOSE_MONTH: getCloseMonth(row),
      CLOSE_YEAR: getCloseYear(row),
      DEAL_DURATION_DAYS: getDealDurationDays(row),
    };
  });
}

export function normalizeDatasets(rawDatasets) {
  const goodsMap = createGoodsMap(rawDatasets.goods.rows);
  const normalizedGoods = normalizeGoodsRows(rawDatasets.goods.rows);
  const normalizedOffers = normalizeDealRows(rawDatasets.offers.rows, goodsMap);
  const normalizedOrders = normalizeDealRows(rawDatasets.orders.rows, goodsMap);

  return {
    goods: normalizedGoods,
    offers: normalizedOffers,
    orders: normalizedOrders,
  };
}

function buildLineGapStats(rows) {
  const groups = groupBy(rows, (row) => cleanText(row.ID));
  const gapRatios = [];

  for (const [dealId, dealRows] of groups.entries()) {
    if (!dealId) {
      continue;
    }

    const opportunity = parseNumber(dealRows[0].OPPORTUNITY);
    const lineTotal = sumLineTotal(dealRows);
    if (opportunity === null || opportunity === 0 || lineTotal === 0) {
      continue;
    }

    gapRatios.push(Math.abs(lineTotal - opportunity) / opportunity);
  }

  return {
    comparableDeals: gapRatios.length,
    medianGapRatio: roundNumber(median(gapRatios), 4),
    p95GapRatio: roundNumber(percentile(gapRatios, 0.95), 4),
    dealsWithin5Pct: gapRatios.filter((value) => value <= 0.05).length,
  };
}

function buildDatasetQa(name, rows, normalizedRows, goodsProductIds) {
  const headers = rows.length ? Object.keys(rows[0]) : [...RAW_HEADERS];
  const uniqueProductIds = unique(rows.map((row) => cleanText(row.PRODUCT_ID)).filter((value) => value && value !== "0"));
  const productCoverage = uniqueProductIds.filter((productId) => goodsProductIds.has(productId)).length;
  const directionMissingCatalogRows = normalizedRows.filter(
    (row) => row.CATALOG_MATCH_STATUS === "catalog" && row.NORMALIZED_DIRECTION === MISSING_DIRECTION
  ).length;
  const placeholderClients = rows.filter(
    (row) => sanitizePlaceholder(row.COMPANY_ID) === "" && sanitizePlaceholder(row.CONTACT_ID) === ""
  ).length;

  return {
    name,
    rowCount: rows.length,
    headers,
    schemaMatchesExpected: JSON.stringify(headers) === JSON.stringify(RAW_HEADERS),
    uniqueIds: unique(rows.map((row) => cleanText(row.ID))).length,
    uniqueProducts: uniqueProductIds.length,
    coveredByGoods: productCoverage,
    missingFromGoods: uniqueProductIds.length - productCoverage,
    directionMissingCatalogRows,
    placeholderClientRows: placeholderClients,
    lineGapStats: name === "goods" ? null : buildLineGapStats(rows),
  };
}

export function buildQaReport(rawDatasets, normalizedDatasets, repairInfo) {
  const goodsProductIds = new Set(
    rawDatasets.goods.rows
      .map((row) => cleanText(row.PRODUCT_ID))
      .filter(Boolean)
  );
  const offerProductIds = new Set(
    rawDatasets.offers.rows
      .map((row) => cleanText(row.PRODUCT_ID))
      .filter((value) => value && value !== "0")
  );
  const orderProductIds = new Set(
    rawDatasets.orders.rows
      .map((row) => cleanText(row.PRODUCT_ID))
      .filter((value) => value && value !== "0")
  );

  const datasets = {
    goods: buildDatasetQa("goods", rawDatasets.goods.rows, normalizedDatasets.goods, goodsProductIds),
    offers: buildDatasetQa("offers", rawDatasets.offers.rows, normalizedDatasets.offers, goodsProductIds),
    orders: buildDatasetQa("orders", rawDatasets.orders.rows, normalizedDatasets.orders, goodsProductIds),
  };

  const issues = [];

  for (const dataset of Object.values(datasets)) {
    if (!dataset.schemaMatchesExpected) {
      issues.push({
        severity: "blocking",
        code: `${dataset.name}_schema_mismatch`,
        message: `${dataset.name} does not match the expected raw CSV contract.`,
      });
    }
  }

  if (datasets.orders.directionMissingCatalogRows > 0) {
    issues.push({
      severity: "blocking",
      code: "orders_missing_direction_after_repair",
      message: `orders still has ${datasets.orders.directionMissingCatalogRows} catalog-backed rows without direction after repair.`,
    });
  }

  if (datasets.orders.placeholderClientRows > 0 || datasets.offers.placeholderClientRows > 0) {
    issues.push({
      severity: "warning",
      code: "missing_client_context",
      message: "A large share of offer/order rows has missing client context; downstream client-aware pricing must degrade gracefully.",
    });
  }

  if (datasets.orders.missingFromGoods > 0) {
    issues.push({
      severity: "warning",
      code: "orders_missing_catalog_links",
      message: `orders has ${datasets.orders.missingFromGoods} catalog product ids not present in goods.`,
    });
  }

  return {
    generatedAt: new Date().toISOString(),
    repairInfo,
    datasets,
    crossDataset: {
      goodsProducts: goodsProductIds.size,
      offerProducts: offerProductIds.size,
      orderProducts: orderProductIds.size,
      sharedOfferOrderProducts: [...offerProductIds].filter((productId) => orderProductIds.has(productId)).length,
      offersCoveredByGoods: [...offerProductIds].filter((productId) => goodsProductIds.has(productId)).length,
      ordersCoveredByGoods: [...orderProductIds].filter((productId) => goodsProductIds.has(productId)).length,
    },
    issues,
  };
}

function classifyProductFact(fact) {
  if (fact.CATALOG_MATCH_STATUS !== "catalog") {
    return {
      priceMode: "manual",
      confidenceTier: "low",
      manualReviewReason: "manual_or_missing_catalog",
    };
  }

  if (isFinancialModifier(fact.PRODUCT_NAME)) {
    return {
      priceMode: "manual",
      confidenceTier: "low",
      manualReviewReason: "financial_modifier",
    };
  }

  if (fact.ORDER_ROWS < 10) {
    return {
      priceMode: "manual",
      confidenceTier: "low",
      manualReviewReason: "insufficient_history",
    };
  }

  if (fact.PRICE_RATIO_RANGE === null) {
    return {
      priceMode: "manual",
      confidenceTier: "low",
      manualReviewReason: "no_price_anchor",
    };
  }

  if (fact.ORDER_ROWS >= 30 && fact.PRICE_RATIO_RANGE <= 0.15) {
    return {
      priceMode: "auto",
      confidenceTier: "high",
      manualReviewReason: "",
    };
  }

  if (fact.ORDER_ROWS >= 10 && fact.PRICE_RATIO_RANGE <= 0.5) {
    return {
      priceMode: "guided",
      confidenceTier: "medium",
      manualReviewReason: "requires_context_lookup",
    };
  }

  return {
    priceMode: "manual",
    confidenceTier: "low",
    manualReviewReason: "high_price_variance",
  };
}
export function buildProductFacts(normalizedDatasets) {
  const goodsByProductKey = new Map(normalizedDatasets.goods.map((row) => [row.PRODUCT_KEY, row]));
  const offersByProductKey = groupBy(normalizedDatasets.offers, (row) => row.PRODUCT_KEY);
  const ordersByProductKey = groupBy(normalizedDatasets.orders, (row) => row.PRODUCT_KEY);
  const allProductKeys = unique([
    ...goodsByProductKey.keys(),
    ...offersByProductKey.keys(),
    ...ordersByProductKey.keys(),
  ]);

  const productFacts = allProductKeys.map((productKey) => {
    const goodsRow = goodsByProductKey.get(productKey);
    const offerRows = offersByProductKey.get(productKey) ?? [];
    const orderRows = ordersByProductKey.get(productKey) ?? [];
    const sampleRow = goodsRow ?? orderRows[0] ?? offerRows[0] ?? {};
    // Exclude barter deals from price statistics — they distort medians
    const pricingOrderRows = orderRows.filter((row) => cleanText(row.PAYMENT_TYPE) !== "Бартер");
    const orderPrices = pricingOrderRows.map((row) => parseNumber(row.PRICE)).filter((value) => value !== null);
    const orderQuantities = pricingOrderRows.map((row) => parseNumber(row.QUANTITY)).filter((value) => value !== null);
    const orderRatios = pricingOrderRows
      .map((row) => parseNumber(row.PRICE_TO_BASE_RATIO))
      .filter((value) => value !== null);
    const offerQuantities = unique(
      offerRows
        .map((row) => parseNumber(row.QUANTITY))
        .filter((value) => value !== null)
        .sort((a, b) => a - b)
    );
    const currentBasePrice =
      parseNumber(goodsRow?.BASE_PRICE) ??
      parseNumber(sampleRow.BASE_PRICE) ??
      null;

    const fact = {
      PRODUCT_KEY: productKey,
      PRODUCT_ID: cleanText(sampleRow.PRODUCT_ID),
      PRODUCT_NAME: firstNonEmpty(sampleRow.NAME, sampleRow.PRODUCT_NAME, sampleRow.PRODUCT_NAME_NORMALIZED),
      CURRENT_CATALOG_NAME: firstNonEmpty(goodsRow?.NAME, sampleRow.NAME, sampleRow.PRODUCT_NAME),
      NORMALIZED_DIRECTION: firstNonEmpty(
        goodsRow?.NORMALIZED_DIRECTION,
        sampleRow.NORMALIZED_DIRECTION,
        sampleRow["Направление"],
        MISSING_DIRECTION
      ),
      CATALOG_MATCH_STATUS: goodsRow ? "catalog" : firstNonEmpty(sampleRow.CATALOG_MATCH_STATUS, "missing_catalog"),
      GOODS_PRESENT: goodsRow ? "Y" : "N",
      SECTION_NAME: cleanText(goodsRow?.SECTION_NAME) || "",
      PARENT_SECTION: cleanText(goodsRow?.PARENT_SECTION) || "",
      COST_PRICE: parseNumber(goodsRow?.COST_PRICE_NUM ?? goodsRow?.COST_PRICE) ?? null,
      COEFFICIENT: parseNumber(goodsRow?.COEFFICIENT_NUM ?? goodsRow?.COEFFICIENT) ?? null,
      OFFER_ROWS: offerRows.length,
      OFFER_DEALS: unique(offerRows.map((row) => row.ID)).length,
      OFFER_QTY_LADDER: offerQuantities.slice(0, 8).join("|"),
      ORDER_ROWS: orderRows.length,
      ORDER_DEALS: unique(orderRows.map((row) => row.ID)).length,
      TOTAL_ORDER_QTY: roundNumber(orderQuantities.reduce((sum, value) => sum + value, 0), 2) ?? 0,
      CURRENT_BASE_PRICE: currentBasePrice,
      ORDER_PRICE_P25: roundNumber(percentile(orderPrices, 0.25), 4),
      ORDER_PRICE_P50: roundNumber(median(orderPrices), 4),
      ORDER_PRICE_P75: roundNumber(percentile(orderPrices, 0.75), 4),
      ORDER_QTY_P25: roundNumber(percentile(orderQuantities, 0.25), 4),
      ORDER_QTY_P50: roundNumber(median(orderQuantities), 4),
      ORDER_QTY_P75: roundNumber(percentile(orderQuantities, 0.75), 4),
      PRICE_RATIO_P25: roundNumber(percentile(orderRatios, 0.25), 4),
      PRICE_RATIO_P50: roundNumber(median(orderRatios), 4),
      PRICE_RATIO_P75: roundNumber(percentile(orderRatios, 0.75), 4),
      PRICE_RATIO_MIN: roundNumber(orderRatios.length ? Math.min(...orderRatios) : null, 4),
      PRICE_RATIO_MAX: roundNumber(orderRatios.length ? Math.max(...orderRatios) : null, 4),
      PRICE_RATIO_RANGE: roundNumber(
        orderRatios.length ? Math.max(...orderRatios) - Math.min(...orderRatios) : null,
        4
      ),
    };

    const classification = classifyProductFact(fact);

    return {
      ...fact,
      PRICE_MODE: classification.priceMode,
      CONFIDENCE_TIER: classification.confidenceTier,
      MANUAL_REVIEW_REASON: classification.manualReviewReason,
    };
  });

  return productFacts.sort((left, right) => {
    if (right.ORDER_ROWS !== left.ORDER_ROWS) {
      return right.ORDER_ROWS - left.ORDER_ROWS;
    }
    return left.PRODUCT_KEY.localeCompare(right.PRODUCT_KEY);
  });
}

export function buildDealFacts(normalizedDatasets) {
  const allRows = [...normalizedDatasets.offers, ...normalizedDatasets.orders];
  const grouped = groupBy(allRows, (row) => `${row.DATASET}:${row.ID}`);
  const dealFacts = [];

  for (const [key, rows] of grouped.entries()) {
    const firstRow = rows[0];
    const productKeys = unique(rows.map((row) => row.PRODUCT_KEY));
    const lineValues = rows.map((row) => parseNumber(row.LINE_TOTAL)).filter((value) => value !== null);
    const totalValue = lineValues.reduce((sum, value) => sum + value, 0);
    const opportunity = parseNumber(firstRow.OPPORTUNITY);
    const gapPct = opportunity && totalValue
      ? Math.abs(totalValue - opportunity) / opportunity
      : null;

    const paymentType = cleanText(firstRow.PAYMENT_TYPE) || "";

    dealFacts.push({
      DATASET: firstRow.DATASET,
      DEAL_ID: firstRow.ID,
      TITLE: cleanText(firstRow.TITLE),
      CLIENT_KEY: firstRow.CLIENT_KEY,
      CLIENT_LABEL: firstRow.CLIENT_LABEL,
      CLIENT_CONTEXT_STATUS: firstRow.CLIENT_CONTEXT_STATUS,
      BEGINDATE: cleanText(firstRow.BEGINDATE),
      CLOSEDATE: cleanText(firstRow.CLOSEDATE),
      CLOSE_MONTH: firstRow.CLOSE_MONTH,
      CLOSE_YEAR: firstRow.CLOSE_YEAR,
      DESCRIPTION: cleanText(firstRow.DESCRIPTION),
      COMMENTS: cleanText(firstRow.COMMENTS) || "",
      PAYMENT_TYPE: paymentType,
      DEAL_DURATION_DAYS: firstRow.DEAL_DURATION_DAYS ?? getDealDurationDays(firstRow),
      LINE_COUNT: rows.length,
      UNIQUE_PRODUCT_COUNT: productKeys.length,
      TOTAL_QUANTITY: roundNumber(
        rows
          .map((row) => parseNumber(row.QUANTITY_NUM))
          .filter((value) => value !== null)
          .reduce((sum, value) => sum + value, 0),
        2
      ) ?? 0,
      LINE_TOTAL: roundNumber(totalValue, 2) ?? 0,
      OPPORTUNITY: opportunity,
      GAP_PCT: roundNumber(gapPct, 4),
      PRIMARY_DIRECTION: getPrimaryDirection(rows),
      BUNDLE_KEY: makeBundleKey(productKeys),
      PRODUCT_KEYS: productKeys.join("|"),
      PARENT_SECTIONS: unique(rows.map((row) => cleanText(row.PARENT_SECTION)).filter(Boolean)).sort().join("|"),
      LETTER_HEIGHT_CM: extractHeightFromProducts(
        rows,
        `${cleanText(firstRow.TITLE) || ""} ${cleanText(firstRow.DESCRIPTION) || ""}`
      ),
      SAMPLE_PRODUCTS: unique(rows.map((row) => row.PRODUCT_NAME_NORMALIZED)).slice(0, 5).join(" | "),
    });
  }

  return dealFacts.sort((left, right) => {
    if (left.DATASET !== right.DATASET) {
      return left.DATASET.localeCompare(right.DATASET);
    }
    return left.DEAL_ID.localeCompare(right.DEAL_ID, undefined, { numeric: true });
  });
}

export function buildBundleFacts(dealFacts) {
  const grouped = groupBy(
    dealFacts.filter((deal) => cleanText(deal.BUNDLE_KEY)),
    (deal) => `${deal.DATASET}:${deal.BUNDLE_KEY}`
  );
  const bundleFacts = [];

  for (const [key, deals] of grouped.entries()) {
    const [dataset, bundleKey] = key.split(":", 2);
    // Exclude barter deals from price medians — they distort real market values
    const pricingDeals = deals.filter((deal) => deal.PAYMENT_TYPE !== "Бартер");
    const values = pricingDeals.map((deal) => parseNumber(deal.LINE_TOTAL)).filter((value) => value !== null);
    const durations = deals
      .map((deal) => parseNumber(deal.DEAL_DURATION_DAYS))
      .filter((value) => value !== null && value >= 0);
    // Extract letter height from deals (mode across all deals in bundle)
    const heights = deals.map((d) => d.LETTER_HEIGHT_CM).filter((h) => h !== null && h > 0);
    const heightMode = heights.length
      ? (() => { const f = new Map(); for (const h of heights) f.set(h, (f.get(h) ?? 0) + 1); return [...f.entries()].sort((a, b) => b[1] - a[1])[0][0]; })()
      : null;

    // Collect best description from deals (longest non-empty)
    const descriptions = deals.map((d) => d.DESCRIPTION || "").filter(Boolean);
    const bestDescription = descriptions.sort((a, b) => b.length - a.length)[0] || "";

    // Canonical bundle key by parent_section set (semantic grouping across variants)
    const parentSections = new Set();
    for (const d of deals) {
      for (const sec of (d.PARENT_SECTIONS || "").split("|").filter(Boolean)) {
        parentSections.add(sec);
      }
    }
    const canonicalKey = [...parentSections].sort().join("|") || bundleKey;

    // Sample deal IDs (up to 5, prefer highest-value)
    const sampleDeals = deals
      .slice()
      .sort((a, b) => (parseNumber(b.LINE_TOTAL) ?? 0) - (parseNumber(a.LINE_TOTAL) ?? 0))
      .slice(0, 5)
      .map((d) => d.DEAL_ID)
      .filter(Boolean);

    bundleFacts.push({
      DATASET: dataset,
      BUNDLE_KEY: bundleKey,
      CANONICAL_BUNDLE_KEY: canonicalKey,
      PRODUCT_COUNT: bundleKey ? bundleKey.split("|").length : 0,
      DEAL_COUNT: deals.length,
      MEDIAN_DEAL_VALUE: roundNumber(median(values), 2),
      AVG_DEAL_VALUE: roundNumber(average(values), 2),
      DURATION_P50: roundNumber(median(durations), 0),
      PRIMARY_DIRECTION: getPrimaryDirection(deals),
      LETTER_HEIGHT_CM: heightMode,
      DESCRIPTION: bestDescription.substring(0, 500),
      SAMPLE_TITLE: deals[0].TITLE,
      SAMPLE_PRODUCTS: deals[0].SAMPLE_PRODUCTS,
      SAMPLE_DEAL_IDS: sampleDeals.join("|"),
    });
  }

  return bundleFacts.sort((left, right) => {
    if (left.DATASET !== right.DATASET) {
      return left.DATASET.localeCompare(right.DATASET);
    }
    return right.DEAL_COUNT - left.DEAL_COUNT;
  });
}

export function buildClientFacts(dealFacts) {
  const grouped = groupBy(dealFacts, (deal) => deal.CLIENT_KEY);
  const clientFacts = [];

  for (const [clientKey, deals] of grouped.entries()) {
    const values = deals.map((deal) => parseNumber(deal.LINE_TOTAL)).filter((value) => value !== null);
    const orderDeals = deals.filter((deal) => deal.DATASET === "orders");
    const offerDeals = deals.filter((deal) => deal.DATASET === "offers");
    const directions = groupBy(deals, (deal) => deal.PRIMARY_DIRECTION);
    const dominantDirection = [...directions.entries()].sort((a, b) => b[1].length - a[1].length)[0]?.[0] ?? MISSING_DIRECTION;

    clientFacts.push({
      CLIENT_KEY: clientKey,
      CLIENT_LABEL: deals[0].CLIENT_LABEL,
      CLIENT_CONTEXT_STATUS: deals[0].CLIENT_CONTEXT_STATUS,
      DEAL_COUNT: deals.length,
      ORDER_DEAL_COUNT: orderDeals.length,
      OFFER_DEAL_COUNT: offerDeals.length,
      TOTAL_VALUE: roundNumber(values.reduce((sum, value) => sum + value, 0), 2) ?? 0,
      AVG_DEAL_VALUE: roundNumber(average(values), 2),
      MEDIAN_DEAL_VALUE: roundNumber(median(values), 2),
      FIRST_CLOSE_MONTH: unique(deals.map((deal) => deal.CLOSE_MONTH)).sort()[0] ?? "",
      LAST_CLOSE_MONTH: unique(deals.map((deal) => deal.CLOSE_MONTH)).sort().at(-1) ?? "",
      DOMINANT_DIRECTION: dominantDirection,
      DATASETS_PRESENT: unique(deals.map((deal) => deal.DATASET)).sort().join("|"),
    });
  }

  return clientFacts.sort((left, right) => {
    if (right.DEAL_COUNT !== left.DEAL_COUNT) {
      return right.DEAL_COUNT - left.DEAL_COUNT;
    }
    return left.CLIENT_KEY.localeCompare(right.CLIENT_KEY);
  });
}
function intersectSets(left, right) {
  const result = new Set();
  const [small, large] = left.size <= right.size ? [left, right] : [right, left];

  for (const value of small) {
    if (large.has(value)) {
      result.add(value);
    }
  }

  return result;
}

export function buildTemplateMatchFacts(dealFacts) {
  const offerDeals = dealFacts.filter((deal) => deal.DATASET === "offers" && deal.BUNDLE_KEY);
  const orderDeals = dealFacts.filter((deal) => deal.DATASET === "orders" && deal.BUNDLE_KEY);
  const offerBundles = groupBy(offerDeals, (deal) => deal.BUNDLE_KEY);
  const orderBundles = groupBy(orderDeals, (deal) => deal.BUNDLE_KEY);
  const orderDealLookup = new Map(orderDeals.map((deal) => [deal.DEAL_ID, deal]));
  const ordersByProduct = new Map();

  for (const deal of orderDeals) {
    for (const productKey of deal.PRODUCT_KEYS.split("|").filter(Boolean)) {
      if (!ordersByProduct.has(productKey)) {
        ordersByProduct.set(productKey, new Set());
      }
      ordersByProduct.get(productKey).add(deal.DEAL_ID);
    }
  }

  const templateMatchFacts = [];

  for (const [bundleKey, deals] of offerBundles.entries()) {
    const productKeys = bundleKey.split("|").filter(Boolean);
    let containedMatches = null;

    for (const productKey of productKeys) {
      const ids = ordersByProduct.get(productKey) ?? new Set();
      containedMatches = containedMatches === null ? new Set(ids) : intersectSets(containedMatches, ids);
      if (containedMatches.size === 0) {
        break;
      }
    }

    const exactMatches = orderBundles.get(bundleKey) ?? [];
    const matchedOrders = [...(containedMatches ?? new Set())]
      .map((dealId) => orderDealLookup.get(dealId))
      .filter(Boolean);
    const matchedValues = matchedOrders
      .map((deal) => parseNumber(deal.LINE_TOTAL))
      .filter((value) => value !== null);

    templateMatchFacts.push({
      TEMPLATE_BUNDLE_KEY: bundleKey,
      TEMPLATE_PRODUCT_COUNT: productKeys.length,
      TEMPLATE_DEAL_COUNT: deals.length,
      SAMPLE_TEMPLATE_TITLE: deals[0].TITLE,
      SAMPLE_TEMPLATE_PRODUCTS: deals[0].SAMPLE_PRODUCTS,
      EXACT_ORDER_MATCH_COUNT: exactMatches.length,
      CONTAINED_ORDER_MATCH_COUNT: matchedOrders.length,
      MATCHED_ORDER_VALUE_MEDIAN: roundNumber(median(matchedValues), 2),
      SAMPLE_ORDER_TITLE: matchedOrders[0]?.TITLE ?? "",
      MATCH_RATE_PER_TEMPLATE_DEAL: deals.length ? roundNumber(matchedOrders.length / deals.length, 4) : null,
    });
  }

  return templateMatchFacts.sort((left, right) => right.CONTAINED_ORDER_MATCH_COUNT - left.CONTAINED_ORDER_MATCH_COUNT);
}

export function buildTimelineFacts(dealFacts) {
  const timelineFacts = [];

  // Group by direction (all datasets combined)
  const byDirection = groupBy(
    dealFacts.filter((d) => d.PRIMARY_DIRECTION && d.PRIMARY_DIRECTION !== MISSING_DIRECTION),
    (d) => d.PRIMARY_DIRECTION
  );

  for (const [direction, deals] of byDirection.entries()) {
    const durations = deals
      .map((d) => parseNumber(d.DEAL_DURATION_DAYS))
      .filter((v) => v !== null && v >= 0);
    if (!durations.length) continue;

    timelineFacts.push({
      GROUP_TYPE: "direction",
      GROUP_KEY: direction,
      DEAL_COUNT: durations.length,
      DURATION_MIN: Math.min(...durations),
      DURATION_P25: roundNumber(percentile(durations, 0.25), 0),
      DURATION_P50: roundNumber(median(durations), 0),
      DURATION_P75: roundNumber(percentile(durations, 0.75), 0),
      DURATION_MAX: Math.max(...durations),
      SAMPLE_TITLE: deals[0].TITLE,
    });
  }

  // Group by bundle_key (top 200 by deal count, orders only)
  const orderDeals = dealFacts.filter(
    (d) => d.DATASET === "orders" && cleanText(d.BUNDLE_KEY)
  );
  const byBundle = groupBy(orderDeals, (d) => d.BUNDLE_KEY);
  const sortedBundles = [...byBundle.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 200);

  for (const [bundleKey, deals] of sortedBundles) {
    const durations = deals
      .map((d) => parseNumber(d.DEAL_DURATION_DAYS))
      .filter((v) => v !== null && v >= 0);
    if (!durations.length) continue;

    timelineFacts.push({
      GROUP_TYPE: "bundle",
      GROUP_KEY: bundleKey,
      DEAL_COUNT: durations.length,
      DURATION_MIN: Math.min(...durations),
      DURATION_P25: roundNumber(percentile(durations, 0.25), 0),
      DURATION_P50: roundNumber(median(durations), 0),
      DURATION_P75: roundNumber(percentile(durations, 0.75), 0),
      DURATION_MAX: Math.max(...durations),
      SAMPLE_TITLE: deals[0].TITLE,
    });
  }

  return timelineFacts;
}

export function buildDealProfiles(normalizedDatasets, dealFacts, { dataset = "orders" } = {}) {
  // Deals with ≥2 products and >500 rub in the requested dataset (orders or offers).
  const selectedDeals = dealFacts.filter(
    (d) => d.DATASET === dataset && d.UNIQUE_PRODUCT_COUNT >= 2 && (parseNumber(d.LINE_TOTAL) ?? 0) > 500
      && d.PAYMENT_TYPE !== "Бартер"
  );

  // Build lookup: dealId → normalized rows (for requested dataset)
  const sourceRows = normalizedDatasets[dataset] || [];
  const rowsByDeal = groupBy(sourceRows, (row) => cleanText(row.ID));

  const profiles = [];

  for (const deal of selectedDeals) {
    const rows = rowsByDeal.get(deal.DEAL_ID) ?? [];
    if (!rows.length) continue;

    // Build component summary: "Product x5 = 1234 руб; ..."
    const components = rows
      .filter((r) => cleanText(r.PRODUCT_NAME_NORMALIZED))
      .map((r) => {
        const name = cleanText(r.PRODUCT_NAME_NORMALIZED);
        const qty = parseNumber(r.QUANTITY_NUM) ?? 1;
        const total = parseNumber(r.LINE_TOTAL) ?? 0;
        const direction = cleanText(r.NORMALIZED_DIRECTION) || cleanText(r["Направление"]) || "";
        return { name, qty, total, direction };
      })
      .sort((a, b) => b.total - a.total);

    const componentSummary = components
      .slice(0, 10)
      .map((c) => `${c.name} x${c.qty} = ${Math.round(c.total)} руб`)
      .join("; ");

    // Direction breakdown within the deal
    const dirTotals = new Map();
    const grandTotal = components.reduce((sum, c) => sum + c.total, 0) || 1;
    for (const c of components) {
      const dir = c.direction && !c.direction.startsWith("__") ? c.direction : "Прочее";
      dirTotals.set(dir, (dirTotals.get(dir) ?? 0) + c.total);
    }
    const dirBreakdown = [...dirTotals.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([dir, total]) => `${dir}: ${Math.round(total / grandTotal * 100)}% (${Math.round(total)} руб)`)
      .join("; ");

    // Count unique directions in this deal
    const directionCount = dirTotals.size;

    // Extract materials from all product names
    const allNames = rows.map((r) => cleanText(r.PRODUCT_NAME_NORMALIZED)).join(" ");
    const materials = extractMaterials(allNames);

    profiles.push({
      DEAL_ID: deal.DEAL_ID,
      DATASET: dataset,
      TITLE: deal.TITLE,
      DIRECTION: deal.PRIMARY_DIRECTION,
      LINE_TOTAL: deal.LINE_TOTAL,
      DEAL_DURATION_DAYS: deal.DEAL_DURATION_DAYS,
      DESCRIPTION: deal.DESCRIPTION,
      COMMENTS: deal.COMMENTS || "",
      UNIQUE_PRODUCT_COUNT: deal.UNIQUE_PRODUCT_COUNT,
      DIRECTION_COUNT: directionCount,
      DIRECTION_BREAKDOWN: dirBreakdown,
      COMPONENT_SUMMARY: componentSummary,
      MATERIALS: materials.join(", "),
      BUNDLE_KEY: deal.BUNDLE_KEY,
      SAMPLE_PRODUCTS: deal.SAMPLE_PRODUCTS,
    });
  }

  return profiles.sort((a, b) => (parseNumber(b.LINE_TOTAL) ?? 0) - (parseNumber(a.LINE_TOTAL) ?? 0));
}

export function buildServiceCompositionFacts(normalizedDatasets, dealFacts) {
  // Aggregate across all qualifying order deals to build
  // "typical service composition" reference by direction.
  const orderDeals = dealFacts.filter(
    (d) => d.DATASET === "orders" && d.UNIQUE_PRODUCT_COUNT >= 2 && (parseNumber(d.LINE_TOTAL) ?? 0) > 500
  );
  const orderRows = groupBy(normalizedDatasets.orders, (row) => cleanText(row.ID));

  // Per-direction aggregation: product category → frequency + cost share
  const dirStats = new Map(); // direction → { dealCount, totalValue, productFreq, materialFreq, crossDirections }

  for (const deal of orderDeals) {
    const rows = orderRows.get(deal.DEAL_ID) ?? [];
    if (!rows.length) continue;

    const primaryDir = deal.PRIMARY_DIRECTION;
    if (!primaryDir || primaryDir.startsWith("__")) continue;

    if (!dirStats.has(primaryDir)) {
      dirStats.set(primaryDir, {
        dealCount: 0,
        totalValue: 0,
        productFreq: new Map(),    // productName → { count, totalValue }
        materialFreq: new Map(),   // material → count
        crossDirections: new Map(), // otherDirection → count
        values: [],
      });
    }
    const stats = dirStats.get(primaryDir);
    stats.dealCount += 1;
    const dealTotal = parseNumber(deal.LINE_TOTAL) ?? 0;
    stats.totalValue += dealTotal;
    stats.values.push(dealTotal);

    // Track cross-direction presence
    const dirsInDeal = new Set();
    for (const row of rows) {
      const rowDir = cleanText(row.NORMALIZED_DIRECTION) || cleanText(row["Направление"]);
      if (rowDir && !rowDir.startsWith("__") && rowDir !== primaryDir) {
        dirsInDeal.add(rowDir);
      }
    }
    for (const d of dirsInDeal) {
      stats.crossDirections.set(d, (stats.crossDirections.get(d) ?? 0) + 1);
    }

    // Track product frequency within this direction
    for (const row of rows) {
      const name = cleanText(row.PRODUCT_NAME_NORMALIZED);
      if (!name || isFinancialModifier(name)) continue;
      const lt = (parseNumber(row.PRICE_NUM) ?? 0) * (parseNumber(row.QUANTITY_NUM) ?? 0);

      // Normalize product name: strip quantities and specifics for grouping
      const normalized = name.replace(/\d+[-–]\d+\s*(мп|м²|шт|см)/gi, "")
        .replace(/более\s*\d+/gi, "")
        .replace(/\s+/g, " ")
        .trim()
        .substring(0, 60);

      if (!stats.productFreq.has(normalized)) {
        stats.productFreq.set(normalized, { count: 0, totalValue: 0 });
      }
      const pf = stats.productFreq.get(normalized);
      pf.count += 1;
      pf.totalValue += lt;
    }

    // Materials
    const allNames = rows.map((r) => cleanText(r.PRODUCT_NAME_NORMALIZED)).join(" ");
    for (const mat of extractMaterials(allNames)) {
      stats.materialFreq.set(mat, (stats.materialFreq.get(mat) ?? 0) + 1);
    }
  }

  // Build output facts
  const compositionFacts = [];

  for (const [direction, stats] of dirStats.entries()) {
    if (stats.dealCount < 3) continue; // skip ultra-rare directions (was 5)

    // Top products by frequency (appear in most deals)
    const topProducts = [...stats.productFreq.entries()]
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 15)
      .map(([name, info]) => ({
        name,
        dealShare: roundNumber(info.count / stats.dealCount, 2),
        avgCostShare: stats.totalValue > 0 ? roundNumber(info.totalValue / stats.totalValue, 4) : 0,
      }));

    // Core products (>5% of deals) vs optional (>2%)
    // Threshold is low because large directions (Цех, Печатная) have huge product diversity
    const coreProducts = topProducts.filter((p) => p.dealShare >= 0.05);
    const optionalProducts = topProducts.filter((p) => p.dealShare >= 0.02 && p.dealShare < 0.05);

    // Top materials by frequency
    const topMaterials = [...stats.materialFreq.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([mat, count]) => `${mat} (${Math.round(count / stats.dealCount * 100)}%)`);

    // Cross-direction combos
    const crossDirs = [...stats.crossDirections.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([dir, count]) => `${dir} (${Math.round(count / stats.dealCount * 100)}%)`);

    compositionFacts.push({
      DIRECTION: direction,
      DEAL_COUNT: stats.dealCount,
      MEDIAN_VALUE: roundNumber(median(stats.values), 0),
      AVG_VALUE: roundNumber(average(stats.values), 0),
      CORE_PRODUCTS: coreProducts
        .map((p) => `${p.name} [${Math.round(p.dealShare * 100)}% сделок, ${Math.round(p.avgCostShare * 100)}% стоимости]`)
        .join("; "),
      OPTIONAL_PRODUCTS: optionalProducts
        .map((p) => `${p.name} [${Math.round(p.dealShare * 100)}% сделок]`)
        .join("; "),
      MATERIALS: topMaterials.join(", "),
      CROSS_DIRECTIONS: crossDirs.join(", "),
    });
  }

  return compositionFacts.sort((a, b) => b.DEAL_COUNT - a.DEAL_COUNT);
}

function findNearestAnalogs(targetName, targetDirection, goodsAnalogIndex, limit = 3) {
  const targetTokens = tokenizeText(targetName);
  if (!targetTokens.length) {
    return [];
  }

  const candidates = new Map();

  for (const token of targetTokens) {
    for (const candidate of goodsAnalogIndex.tokenMap.get(token) ?? []) {
      if (!candidates.has(candidate.productId)) {
        candidates.set(candidate.productId, candidate);
      }
    }
  }

  const scored = [...candidates.values()]
    .map((candidate) => {
      let score = overlapScore(targetName, candidate.name);
      if (targetDirection && candidate.direction && targetDirection === candidate.direction) {
        score += 0.15;
      }

      return {
        ...candidate,
        score,
      };
    })
    .filter((candidate) => candidate.score > 0)
    .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));

  return scored.slice(0, limit).map((candidate) => `${candidate.productId}:${candidate.name}`);
}

export function buildPricingFacts(productFacts, normalizedDatasets) {
  const goodsAnalogIndex = createGoodsAnalogIndex(normalizedDatasets.goods);
  const analogCache = new Map();

  return productFacts.map((fact) => {
    const currentBasePrice = parseNumber(fact.CURRENT_BASE_PRICE);
    const p25 = parseNumber(fact.ORDER_PRICE_P25);
    const p50 = parseNumber(fact.ORDER_PRICE_P50);
    const p75 = parseNumber(fact.ORDER_PRICE_P75);
    let recommendedPrice = null;
    let suggestedMin = null;
    let suggestedMax = null;
    let analogs = [];

    if (fact.PRICE_MODE === "auto") {
      recommendedPrice = p50 ?? currentBasePrice;
      suggestedMin = p25 ?? recommendedPrice;
      suggestedMax = p75 ?? recommendedPrice;
    } else if (fact.PRICE_MODE === "guided") {
      recommendedPrice = p50 ?? currentBasePrice;
      suggestedMin = p25 ?? (recommendedPrice !== null ? recommendedPrice * 0.9 : null);
      suggestedMax = p75 ?? (recommendedPrice !== null ? recommendedPrice * 1.1 : null);
      if (currentBasePrice !== null) {
        suggestedMin = Math.min(suggestedMin ?? currentBasePrice, currentBasePrice * 0.9);
        suggestedMax = Math.max(suggestedMax ?? currentBasePrice, currentBasePrice * 1.1);
      }
    } else if (fact.MANUAL_REVIEW_REASON !== "financial_modifier") {
      const catalogAnchorName = firstNonEmpty(fact.CURRENT_CATALOG_NAME, fact.PRODUCT_NAME);
      const hasCatalogAnchor = fact.CATALOG_MATCH_STATUS === "catalog" && cleanText(fact.PRODUCT_ID);

      if (hasCatalogAnchor) {
        analogs = [`${fact.PRODUCT_ID}:${catalogAnchorName}`];
      } else {
        const cacheKey = `${fact.PRODUCT_NAME}|${fact.NORMALIZED_DIRECTION}`;
        if (!analogCache.has(cacheKey)) {
          analogCache.set(
            cacheKey,
            findNearestAnalogs(fact.PRODUCT_NAME, fact.NORMALIZED_DIRECTION, goodsAnalogIndex)
          );
        }
        analogs = analogCache.get(cacheKey) ?? [];
      }
    }

    // Compute real markup ratio: P50 order price / cost_price
    const costPrice = parseNumber(fact.COST_PRICE);
    const markupRatio = (costPrice && costPrice > 0 && p50 && p50 > 0)
      ? roundNumber(p50 / costPrice, 2)
      : null;

    return {
      PRODUCT_KEY: fact.PRODUCT_KEY,
      PRODUCT_ID: fact.PRODUCT_ID,
      PRODUCT_NAME: fact.PRODUCT_NAME,
      NORMALIZED_DIRECTION: fact.NORMALIZED_DIRECTION,
      SECTION_NAME: fact.SECTION_NAME ?? "",
      PARENT_SECTION: fact.PARENT_SECTION ?? "",
      COST_PRICE: fact.COST_PRICE,
      MARKUP_RATIO: markupRatio,
      PRICE_MODE: fact.PRICE_MODE,
      CONFIDENCE_TIER: fact.CONFIDENCE_TIER,
      ORDER_ROWS: fact.ORDER_ROWS,
      OFFER_ROWS: fact.OFFER_ROWS,
      CURRENT_BASE_PRICE: fact.CURRENT_BASE_PRICE,
      RECOMMENDED_PRICE: roundNumber(recommendedPrice, 2),
      SUGGESTED_MIN_PRICE: roundNumber(suggestedMin, 2),
      SUGGESTED_MAX_PRICE: roundNumber(suggestedMax, 2),
      OFFER_QTY_LADDER: fact.OFFER_QTY_LADDER,
      MANUAL_REVIEW_REASON: fact.MANUAL_REVIEW_REASON,
      NEAREST_ANALOG_1: analogs[0] ?? "",
      NEAREST_ANALOG_2: analogs[1] ?? "",
      NEAREST_ANALOG_3: analogs[2] ?? "",
    };
  });
}

export function buildKpis(productFacts, dealFacts, templateMatchFacts) {
  const monthlyDirectionKpis = [];
  const monthlyGroups = groupBy(
    dealFacts,
    (deal) => `${deal.DATASET}:${deal.CLOSE_MONTH}:${deal.PRIMARY_DIRECTION}`
  );

  for (const [key, deals] of monthlyGroups.entries()) {
    const [dataset, closeMonth, primaryDirection] = key.split(":", 3);
    const values = deals.map((deal) => parseNumber(deal.LINE_TOTAL)).filter((value) => value !== null);
    monthlyDirectionKpis.push({
      DATASET: dataset,
      CLOSE_MONTH: closeMonth,
      PRIMARY_DIRECTION: primaryDirection,
      DEAL_COUNT: deals.length,
      TOTAL_VALUE: roundNumber(values.reduce((sum, value) => sum + value, 0), 2) ?? 0,
      MEDIAN_DEAL_VALUE: roundNumber(median(values), 2),
      AVG_DEAL_VALUE: roundNumber(average(values), 2),
    });
  }

  const catalogUtilization = productFacts
    .filter((fact) => fact.GOODS_PRESENT === "Y")
    .map((fact) => ({
      PRODUCT_ID: fact.PRODUCT_ID,
      PRODUCT_NAME: fact.PRODUCT_NAME,
      NORMALIZED_DIRECTION: fact.NORMALIZED_DIRECTION,
      CURRENT_BASE_PRICE: fact.CURRENT_BASE_PRICE,
      OFFER_ROWS: fact.OFFER_ROWS,
      ORDER_ROWS: fact.ORDER_ROWS,
      UTILIZATION_STATUS:
        fact.OFFER_ROWS > 0 && fact.ORDER_ROWS > 0
          ? "used_in_both"
          : fact.OFFER_ROWS > 0
            ? "offers_only"
            : fact.ORDER_ROWS > 0
              ? "orders_only"
              : "unused",
    }))
    .sort((left, right) => left.UTILIZATION_STATUS.localeCompare(right.UTILIZATION_STATUS) || right.ORDER_ROWS - left.ORDER_ROWS);

  const clientConcentration = (() => {
    const orderDeals = dealFacts.filter((deal) => deal.DATASET === "orders" && deal.CLIENT_KEY !== MISSING_CLIENT);
    const groups = groupBy(orderDeals, (deal) => deal.CLIENT_KEY);
    const totalValue = orderDeals.reduce((sum, deal) => sum + (parseNumber(deal.LINE_TOTAL) ?? 0), 0);
    let cumulativeShare = 0;

    return [...groups.entries()]
      .map(([clientKey, deals]) => {
        const value = deals.reduce((sum, deal) => sum + (parseNumber(deal.LINE_TOTAL) ?? 0), 0);
        return {
          CLIENT_KEY: clientKey,
          CLIENT_LABEL: deals[0].CLIENT_LABEL,
          DEAL_COUNT: deals.length,
          TOTAL_VALUE: roundNumber(value, 2),
          DOMINANT_DIRECTION: getPrimaryDirection(deals),
        };
      })
      .sort((left, right) => right.TOTAL_VALUE - left.TOTAL_VALUE)
      .map((row) => {
        const valueShare = totalValue ? row.TOTAL_VALUE / totalValue : 0;
        cumulativeShare += valueShare;
        return {
          ...row,
          VALUE_SHARE: roundNumber(valueShare, 4),
          CUMULATIVE_SHARE: roundNumber(cumulativeShare, 4),
        };
      });
  })();

  const priceStability = productFacts
    .map((fact) => ({
      PRODUCT_ID: fact.PRODUCT_ID,
      PRODUCT_NAME: fact.PRODUCT_NAME,
      NORMALIZED_DIRECTION: fact.NORMALIZED_DIRECTION,
      ORDER_ROWS: fact.ORDER_ROWS,
      PRICE_MODE: fact.PRICE_MODE,
      CONFIDENCE_TIER: fact.CONFIDENCE_TIER,
      PRICE_RATIO_RANGE: fact.PRICE_RATIO_RANGE,
      CURRENT_BASE_PRICE: fact.CURRENT_BASE_PRICE,
      ORDER_PRICE_P50: fact.ORDER_PRICE_P50,
      MANUAL_REVIEW_REASON: fact.MANUAL_REVIEW_REASON,
    }))
    .sort((left, right) => {
      const leftRange = parseNumber(left.PRICE_RATIO_RANGE) ?? Number.POSITIVE_INFINITY;
      const rightRange = parseNumber(right.PRICE_RATIO_RANGE) ?? Number.POSITIVE_INFINITY;
      if (leftRange !== rightRange) {
        return leftRange - rightRange;
      }
      return (right.ORDER_ROWS ?? 0) - (left.ORDER_ROWS ?? 0);
    });
  const templateConversionProxy = templateMatchFacts.map((row) => ({
    TEMPLATE_BUNDLE_KEY: row.TEMPLATE_BUNDLE_KEY,
    TEMPLATE_PRODUCT_COUNT: row.TEMPLATE_PRODUCT_COUNT,
    TEMPLATE_DEAL_COUNT: row.TEMPLATE_DEAL_COUNT,
    EXACT_ORDER_MATCH_COUNT: row.EXACT_ORDER_MATCH_COUNT,
    CONTAINED_ORDER_MATCH_COUNT: row.CONTAINED_ORDER_MATCH_COUNT,
    MATCH_RATE_PER_TEMPLATE_DEAL: row.MATCH_RATE_PER_TEMPLATE_DEAL,
    SAMPLE_TEMPLATE_TITLE: row.SAMPLE_TEMPLATE_TITLE,
    SAMPLE_ORDER_TITLE: row.SAMPLE_ORDER_TITLE,
  }));

  const knownOrderClients = clientConcentration.length;
  const repeatOrderClients = clientConcentration.filter((row) => row.DEAL_COUNT >= 2).length;
  const priceModeCounts = groupBy(productFacts, (fact) => fact.PRICE_MODE);

  const kpiSummary = {
    generatedAt: new Date().toISOString(),
    orderDeals: dealFacts.filter((deal) => deal.DATASET === "orders").length,
    offerDeals: dealFacts.filter((deal) => deal.DATASET === "offers").length,
    knownOrderClients,
    repeatOrderClients,
    repeatOrderClientShare: knownOrderClients ? roundNumber(repeatOrderClients / knownOrderClients, 4) : null,
    autoProducts: (priceModeCounts.get("auto") ?? []).length,
    guidedProducts: (priceModeCounts.get("guided") ?? []).length,
    manualProducts: (priceModeCounts.get("manual") ?? []).length,
    exactTemplateMatches: templateMatchFacts.filter((row) => row.EXACT_ORDER_MATCH_COUNT > 0).length,
    containedTemplateMatches: templateMatchFacts.filter((row) => row.CONTAINED_ORDER_MATCH_COUNT > 0).length,
  };

  return {
    monthlyDirectionKpis: monthlyDirectionKpis.sort((left, right) => left.CLOSE_MONTH.localeCompare(right.CLOSE_MONTH) || left.DATASET.localeCompare(right.DATASET)),
    catalogUtilization,
    clientConcentration,
    priceStability,
    templateConversionProxy,
    kpiSummary,
  };
}

async function writeNormalizedArtifacts(normalizedDatasets) {
  await writeCsv(resolve(NORMALIZED_DIR, "goods.normalized.csv"), normalizedDatasets.goods, NORMALIZED_HEADERS);
  await writeCsv(resolve(NORMALIZED_DIR, "offers.normalized.csv"), normalizedDatasets.offers, NORMALIZED_HEADERS);
  await writeCsv(resolve(NORMALIZED_DIR, "orders.normalized.csv"), normalizedDatasets.orders, NORMALIZED_HEADERS);
}

async function writeFactsArtifacts(productFacts, dealFacts, bundleFacts, clientFacts, templateMatchFacts) {
  await writeCsv(resolve(FACTS_DIR, "product_facts.csv"), productFacts, Object.keys(productFacts[0] ?? {}));
  await writeCsv(resolve(FACTS_DIR, "deal_facts.csv"), dealFacts, Object.keys(dealFacts[0] ?? {}));
  await writeCsv(resolve(FACTS_DIR, "bundle_facts.csv"), bundleFacts, Object.keys(bundleFacts[0] ?? {}));
  await writeCsv(resolve(FACTS_DIR, "client_facts.csv"), clientFacts, Object.keys(clientFacts[0] ?? {}));
  await writeCsv(resolve(FACTS_DIR, "template_match_facts.csv"), templateMatchFacts, Object.keys(templateMatchFacts[0] ?? {}));
}

async function writePricingArtifacts(pricingFacts, pricingSummary) {
  await writeCsv(resolve(PRICING_DIR, "pricing_recommendations.csv"), pricingFacts, Object.keys(pricingFacts[0] ?? {}));
  await writeJsonAtomically(resolve(PRICING_DIR, "pricing_summary.json"), pricingSummary);
}

async function writeKpiArtifacts(kpis) {
  await writeCsv(resolve(KPI_DIR, "monthly_direction_kpis.csv"), kpis.monthlyDirectionKpis, Object.keys(kpis.monthlyDirectionKpis[0] ?? {}));
  await writeCsv(resolve(KPI_DIR, "catalog_utilization.csv"), kpis.catalogUtilization, Object.keys(kpis.catalogUtilization[0] ?? {}));
  await writeCsv(resolve(KPI_DIR, "client_concentration.csv"), kpis.clientConcentration, Object.keys(kpis.clientConcentration[0] ?? {}));
  await writeCsv(resolve(KPI_DIR, "price_stability.csv"), kpis.priceStability, Object.keys(kpis.priceStability[0] ?? {}));
  await writeCsv(resolve(KPI_DIR, "template_conversion_proxy.csv"), kpis.templateConversionProxy, Object.keys(kpis.templateConversionProxy[0] ?? {}));
  await writeJsonAtomically(resolve(KPI_DIR, "kpi_summary.json"), kpis.kpiSummary);
}

export async function runAnalyticsPipeline({ scope = "all", repairRawOrders = true } = {}) {
  await ensureDirectories([OUTPUT_DIR, NORMALIZED_DIR, FACTS_DIR, PRICING_DIR, KPI_DIR]);

  const rawDatasets = await loadRawDatasets();
  const repairResult = repairRawOrders
    ? await repairOrdersDirectionFile(rawDatasets)
    : { datasets: rawDatasets, fixedRows: 0 };
  const normalizedDatasets = normalizeDatasets(repairResult.datasets);
  const qaReport = buildQaReport(repairResult.datasets, normalizedDatasets, {
    repairedOrdersDirectionRows: repairResult.fixedRows,
  });

  await writeJsonAtomically(resolve(OUTPUT_DIR, "qa_report.json"), qaReport);
  await writeNormalizedArtifacts(normalizedDatasets);

  const productFacts = buildProductFacts(normalizedDatasets);
  const dealFacts = buildDealFacts(normalizedDatasets);
  const bundleFacts = buildBundleFacts(dealFacts);
  const clientFacts = buildClientFacts(dealFacts);
  const templateMatchFacts = buildTemplateMatchFacts(dealFacts);

  const timelineFacts = buildTimelineFacts(dealFacts);
  const dealProfiles = buildDealProfiles(normalizedDatasets, dealFacts, { dataset: "orders" });
  const offerProfiles = buildDealProfiles(normalizedDatasets, dealFacts, { dataset: "offers" });
  const serviceCompositionFacts = buildServiceCompositionFacts(normalizedDatasets, dealFacts);

  if (["all", "facts", "pricing", "kpis"].includes(scope)) {
    await writeFactsArtifacts(productFacts, dealFacts, bundleFacts, clientFacts, templateMatchFacts);
    if (timelineFacts.length) {
      await writeCsv(resolve(FACTS_DIR, "timeline_facts.csv"), timelineFacts, Object.keys(timelineFacts[0]));
    }
    if (dealProfiles.length) {
      await writeCsv(resolve(FACTS_DIR, "deal_profiles.csv"), dealProfiles, Object.keys(dealProfiles[0]));
    }
    if (offerProfiles.length) {
      await writeCsv(resolve(FACTS_DIR, "offer_profiles.csv"), offerProfiles, Object.keys(offerProfiles[0]));
    }
    if (serviceCompositionFacts.length) {
      await writeCsv(resolve(FACTS_DIR, "service_composition.csv"), serviceCompositionFacts, Object.keys(serviceCompositionFacts[0]));
    }
  }

  let pricingFacts = [];
  let pricingSummary = {
    generatedAt: new Date().toISOString(),
    autoProducts: 0,
    guidedProducts: 0,
    manualProducts: 0,
    manualWithAnalogs: 0,
  };

  if (["all", "pricing"].includes(scope)) {
    pricingFacts = buildPricingFacts(productFacts, normalizedDatasets);
    pricingSummary = {
      generatedAt: new Date().toISOString(),
      autoProducts: pricingFacts.filter((row) => row.PRICE_MODE === "auto").length,
      guidedProducts: pricingFacts.filter((row) => row.PRICE_MODE === "guided").length,
      manualProducts: pricingFacts.filter((row) => row.PRICE_MODE === "manual").length,
      manualWithAnalogs: pricingFacts.filter((row) => row.PRICE_MODE === "manual" && row.NEAREST_ANALOG_1).length,
    };
    await writePricingArtifacts(pricingFacts, pricingSummary);
  }

  let kpis = {
    monthlyDirectionKpis: [],
    catalogUtilization: [],
    clientConcentration: [],
    priceStability: [],
    templateConversionProxy: [],
    kpiSummary: {
      generatedAt: new Date().toISOString(),
      orderDeals: 0,
      offerDeals: 0,
      sharedTemplateMatches: 0,
      autoProducts: 0,
      guidedProducts: 0,
      manualProducts: 0,
    },
  };

  if (["all", "kpis"].includes(scope)) {
    kpis = buildKpis(productFacts, dealFacts, templateMatchFacts);
    await writeKpiArtifacts(kpis);
  }

  return {
    qaReport,
    normalizedDatasets,
    productFacts,
    dealFacts,
    bundleFacts,
    clientFacts,
    templateMatchFacts,
    timelineFacts,
    dealProfiles,
    offerProfiles,
    serviceCompositionFacts,
    pricingFacts,
    pricingSummary,
    kpis,
  };
}


