import { fileURLToPath } from "url";
import { resolve } from "path";

export const ROOT_DIR = resolve(fileURLToPath(new URL("../..", import.meta.url)));
export const RAW_DATA_DIR = resolve(ROOT_DIR, "RAG_DATA");
export const ANALYTICS_DIR = resolve(ROOT_DIR, "RAG_ANALYTICS");
export const OUTPUT_DIR = resolve(ANALYTICS_DIR, "output");
export const NORMALIZED_DIR = resolve(OUTPUT_DIR, "normalized");
export const FACTS_DIR = resolve(OUTPUT_DIR, "facts");
export const PRICING_DIR = resolve(OUTPUT_DIR, "pricing");
export const KPI_DIR = resolve(OUTPUT_DIR, "kpis");

export const RAW_HEADERS = [
  "DATASET",
  "ID",
  "TITLE",
  "OPPORTUNITY",
  "COMPANY_ID",
  "CONTACT_ID",
  "BEGINDATE",
  "CLOSEDATE",
  "DESCRIPTION",
  "GOOD_ID",
  "PRODUCT_ID",
  "PRODUCT_NAME",
  "PRICE",
  "PRICE_ACCOUNT",
  "QUANTITY",
  "BASE_PRICE",
  "CATALOG_ID",
  "SECTION_ID",
  "NAME",
  "Направление",
];

export const NORMALIZED_HEADERS = [
  ...RAW_HEADERS,
  "CLIENT_SOURCE",
  "CLIENT_LABEL",
  "CLIENT_KEY",
  "CLIENT_CONTEXT_STATUS",
  "PRODUCT_KEY",
  "CATALOG_MATCH_STATUS",
  "PRODUCT_NAME_NORMALIZED",
  "LINE_TOTAL",
  "PRICE_NUM",
  "BASE_PRICE_NUM",
  "QUANTITY_NUM",
  "PRICE_TO_BASE_RATIO",
  "NORMALIZED_DIRECTION",
  "DIRECTION_SOURCE",
  "CLOSE_MONTH",
  "CLOSE_YEAR",
  "DEAL_DURATION_DAYS",
];

export const MISSING_CLIENT = "__MISSING_CLIENT__";
export const MISSING_DIRECTION = "__MISSING_DIRECTION__";
export const MANUAL_DIRECTION = "__MANUAL_OR_UNCATEGORIZED__";
export const NOT_APPLICABLE = "__NOT_APPLICABLE__";

export const RAW_DATASET_FILES = {
  goods: resolve(RAW_DATA_DIR, "goods.csv"),
  offers: resolve(RAW_DATA_DIR, "offers.csv"),
  orders: resolve(RAW_DATA_DIR, "orders.csv"),
};

export function roundNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }

  return Number(value.toFixed(digits));
}

export function parseNumber(value) {
  if (value === null || value === undefined) {
    return null;
  }

  const text = String(value).trim().replace(/,/g, ".");
  if (!text) {
    return null;
  }

  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseDate(value) {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function average(numbers) {
  if (!numbers.length) {
    return null;
  }

  return numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
}

export function median(numbers) {
  return percentile(numbers, 0.5);
}

export function percentile(numbers, q) {
  if (!numbers.length) {
    return null;
  }

  const sorted = [...numbers].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sorted[lower];
  }

  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function unique(values) {
  return [...new Set(values.filter((value) => value !== null && value !== undefined && value !== ""))];
}

export function sanitizePlaceholder(value) {
  if (value === null || value === undefined) {
    return "";
  }

  const text = String(value).trim();
  return text === "0" ? "" : text;
}

export function cleanText(value) {
  return String(value ?? "").trim();
}

export function tokenizeText(value) {
  return unique(
    cleanText(value)
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .split(/\s+/)
      .filter((token) => token.length >= 2)
  );
}

export function buildProductKey(row) {
  const productId = cleanText(row.PRODUCT_ID);
  if (productId && productId !== "0") {
    return productId;
  }

  const manualName = cleanText(row.PRODUCT_NAME) || cleanText(row.NAME) || cleanText(row.TITLE) || "manual";
  return `manual:${manualName.toLowerCase()}`;
}

export function pickClient(row) {
  const company = sanitizePlaceholder(row.COMPANY_ID);
  const contact = sanitizePlaceholder(row.CONTACT_ID);

  if (company) {
    return {
      clientSource: "company",
      clientLabel: company,
      clientKey: `company:${company.toLowerCase()}`,
      clientStatus: "known",
      companyId: company,
      contactId: contact,
    };
  }

  if (contact) {
    return {
      clientSource: "contact",
      clientLabel: contact,
      clientKey: `contact:${contact.toLowerCase()}`,
      clientStatus: "known",
      companyId: company,
      contactId: contact,
    };
  }

  return {
    clientSource: "missing",
    clientLabel: MISSING_CLIENT,
    clientKey: MISSING_CLIENT,
    clientStatus: "missing",
    companyId: company,
    contactId: contact,
  };
}

export function getCloseMonth(row) {
  const closeDate = parseDate(row.CLOSEDATE);
  if (!closeDate) {
    return "";
  }

  return closeDate.toISOString().slice(0, 7);
}

export function getCloseYear(row) {
  const closeDate = parseDate(row.CLOSEDATE);
  if (!closeDate) {
    return "";
  }

  return String(closeDate.getUTCFullYear());
}

export function sumLineTotal(rows) {
  return rows.reduce((sum, row) => sum + (parseNumber(row.PRICE) ?? 0) * (parseNumber(row.QUANTITY) ?? 0), 0);
}

export function getPrimaryDirection(rows) {
  const counts = new Map();

  for (const row of rows) {
    const direction = cleanText(
      row.NORMALIZED_DIRECTION || row.PRIMARY_DIRECTION || row["Направление"]
    );
    if (!direction || direction === MISSING_DIRECTION) {
      continue;
    }

    counts.set(direction, (counts.get(direction) ?? 0) + 1);
  }

  if (!counts.size) {
    return MISSING_DIRECTION;
  }

  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

export function makeBundleKey(productKeys) {
  const normalized = unique(productKeys.map((value) => cleanText(value))).sort();
  return normalized.join("|");
}

export function isFinancialModifier(name) {
  const normalized = cleanText(name).toLowerCase();
  return /безнал|скидк|налог|комис|процент|%/.test(normalized);
}

export function getDealDurationDays(row) {
  const begin = parseDate(row.BEGINDATE);
  const close = parseDate(row.CLOSEDATE);
  if (!begin || !close) {
    return null;
  }
  const diffMs = close.getTime() - begin.getTime();
  const days = Math.round(diffMs / (1000 * 60 * 60 * 24));
  return days >= 0 ? days : null;
}

const MATERIAL_PATTERNS = [
  [/пвх/i, "ПВХ"],
  [/акрил/i, "Акрил"],
  [/led|лед\b|светодиод/i, "LED"],
  [/неон/i, "Неон"],
  [/композит/i, "Композит"],
  [/плёнк|пленк/i, "Пленка"],
  [/баннер/i, "Баннер"],
  [/алюмин/i, "Алюминий"],
  [/оцинков|металл/i, "Металл"],
  [/стекл|оргстекл/i, "Стекло"],
  [/винил/i, "Винил"],
  [/полистирол/i, "Полистирол"],
  [/пенопласт|пеноплекс/i, "Пенопласт"],
  [/дерев|фанер|мдф/i, "Дерево/МДФ"],
  [/текстил|ткан/i, "Текстиль"],
  [/магнит/i, "Магнит"],
];

export function extractMaterials(text) {
  if (!text) return [];
  const lower = String(text).toLowerCase();
  const found = [];
  for (const [pattern, label] of MATERIAL_PATTERNS) {
    if (pattern.test(lower)) {
      found.push(label);
    }
  }
  return found;
}

export function overlapScore(leftName, rightName) {
  const leftTokens = tokenizeText(leftName);
  const rightTokens = new Set(tokenizeText(rightName));

  if (!leftTokens.length || !rightTokens.size) {
    return 0;
  }

  let overlap = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      overlap += 1;
    }
  }

  return overlap / Math.max(leftTokens.length, rightTokens.size);
}

