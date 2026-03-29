import { mkdir, readFile, writeFile, rename, rm, access } from "fs/promises";
import { dirname } from "path";
import { writeCsvAtomically } from "../../RAG_DATA/csvUtils.mjs";

function stripBom(text) {
  return text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
}

function parseCsvText(text, delimiter = ";") {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const nextChar = text[index + 1];

    if (inQuotes) {
      if (char === '"' && nextChar === '"') {
        value += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        value += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }

    if (char === delimiter) {
      row.push(value);
      value = "";
      continue;
    }

    if (char === "\r") {
      continue;
    }

    if (char === "\n") {
      row.push(value);
      rows.push(row);
      row = [];
      value = "";
      continue;
    }

    value += char;
  }

  if (value.length > 0 || row.length > 0) {
    row.push(value);
    rows.push(row);
  }

  return rows;
}

export async function readCsv(filePath, delimiter = ";") {
  const text = stripBom(await readFile(filePath, "utf8"));
  const [headers = [], ...rows] = parseCsvText(text, delimiter);

  return rows
    .filter((row) => row.some((value) => value !== ""))
    .map((row) => Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ""])));
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

export async function writeJsonAtomically(filePath, value) {
  const directory = dirname(filePath);
  const tempPath = `${filePath}.tmp`;
  const backupPath = `${filePath}.bak`;
  const payload = `${JSON.stringify(value, null, 2)}\n`;

  await mkdir(directory, { recursive: true });
  await writeFile(tempPath, payload, "utf8");

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

export async function writeCsv(filePath, rows, headers) {
  await writeCsvAtomically(filePath, rows, headers);
}

export async function ensureDirectories(paths) {
  for (const directory of paths) {
    await mkdir(directory, { recursive: true });
  }
}
