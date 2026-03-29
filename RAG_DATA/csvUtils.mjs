import { access, mkdir, rename, rm, writeFile } from "fs/promises";
import { dirname } from "path";

export const CSV_HEADERS = [
  "DATASET",
  "ID",
  "TITLE",
  "OPPORTUNITY",
  "COMPANY_ID",
  "CONTACT_ID",
  "BEGINDATE",
  "CLOSEDATE",
  "DESCRIPTION",
  "COMMENTS",
  "DIRECTION",
  "SIGN_TYPE",
  "PAYMENT_TYPE",
  "SOURCE_ID",
  "IS_RETURN_CUSTOMER",
  "GOOD_ID",
  "PRODUCT_ID",
  "PRODUCT_NAME",
  "PRICE",
  "PRICE_ACCOUNT",
  "QUANTITY",
  "BASE_PRICE",
  "CATALOG_ID",
  "SECTION_ID",
  "SECTION_NAME",
  "PARENT_SECTION",
  "NAME",
  "Направление",
  "PRODUCT_DESCRIPTION",
  "COEFFICIENT",
  "WORKSHOP_SALARY",
  "COST_PRICE",
  "EXECUTOR",
];

export function createEmptyRow() {
  return Object.fromEntries(CSV_HEADERS.map((header) => [header, ""]));
}

export function withSchema(values = {}) {
  return {
    ...createEmptyRow(),
    ...values,
  };
}

function escapeCsvValue(value) {
  if (value === null || value === undefined) {
    return "";
  }

  const stringValue = String(value);

  if (
    stringValue.includes(";") ||
    stringValue.includes("\"") ||
    stringValue.includes("\n") ||
    stringValue.includes("\r")
  ) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }

  return stringValue;
}

export function toCsv(rows, headers = CSV_HEADERS) {
  const lines = [headers.join(";")];

  for (const row of rows) {
    lines.push(headers.map((header) => escapeCsvValue(row[header])).join(";"));
  }

  return lines.join("\n");
}

async function pathExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") {
      return false;
    }

    throw error;
  }
}

export async function writeCsvAtomically(filePath, rows, headers = CSV_HEADERS) {
  const directory = dirname(filePath);
  const tempPath = `${filePath}.tmp`;
  const backupPath = `${filePath}.bak`;
  const csvContent = toCsv(rows, headers);

  await mkdir(directory, { recursive: true });
  await writeFile(tempPath, csvContent, "utf8");

  const hadOriginal = await pathExists(filePath);

  try {
    await rm(backupPath, { force: true });

    if (hadOriginal) {
      await rename(filePath, backupPath);
    }

    await rename(tempPath, filePath);

    if (hadOriginal) {
      await rm(backupPath, { force: true });
    }
  } catch (error) {
    await rm(tempPath, { force: true }).catch(() => {});

    if (hadOriginal && !(await pathExists(filePath)) && (await pathExists(backupPath))) {
      await rename(backupPath, filePath).catch(() => {});
    }

    throw error;
  }
}
