import bxPostJson from "../Bitrix24Webhook.mjs";

const MIN_REQUEST_INTERVAL_MS = 650;
const DEFAULT_PAGE_SIZE = 50;
const MAX_RETRIES = 4;
const RETRY_DELAY_MS = 2500;

let nextAllowedAt = 0;

function sleep(delayMs) {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

function isRetryableError(error) {
  const retryableCodes = new Set([
    "ECONNRESET",
    "ECONNABORTED",
    "ETIMEDOUT",
    "EAI_AGAIN",
    "ERR_STREAM_PREMATURE_CLOSE",
  ]);

  if (retryableCodes.has(error?.code)) {
    return true;
  }

  const message = String(error?.message ?? "").toLowerCase();

  return [
    "premature close",
    "socket hang up",
    "network",
    "fetch failed",
    "invalid response body",
  ].some((fragment) => message.includes(fragment));
}

async function waitForBitrixSlot() {
  const waitMs = Math.max(0, nextAllowedAt - Date.now());
  if (waitMs > 0) {
    await sleep(waitMs);
  }
}

export async function bitrixPost(method, params = {}) {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt += 1) {
    await waitForBitrixSlot();

    try {
      const result = await bxPostJson(method, params);
      nextAllowedAt = Date.now() + MIN_REQUEST_INTERVAL_MS;
      return result;
    } catch (error) {
      nextAllowedAt = Date.now() + MIN_REQUEST_INTERVAL_MS;

      if (!isRetryableError(error) || attempt === MAX_RETRIES) {
        throw error;
      }

      const retryDelayMs = RETRY_DELAY_MS * attempt;
      console.warn(
        `Retrying ${method} after transient error (${attempt}/${MAX_RETRIES - 1}): ${error.message}`
      );
      await sleep(retryDelayMs);
    }
  }
}

export async function bitrixList(
  method,
  {
    order = { ID: "ASC" },
    filter = {},
    select = [],
    pageSize = DEFAULT_PAGE_SIZE,
  } = {}
) {
  const items = [];
  let start = 0;

  while (true) {
    const response = await bitrixPost(method, {
      order,
      filter,
      select,
      start,
      count: pageSize,
    });

    const batch = Array.isArray(response?.result)
      ? response.result
      : Array.isArray(response)
        ? response
        : [];

    items.push(...batch);

    if (response?.next === undefined || response?.next === null) {
      break;
    }

    start = response.next;
  }

  return items;
}

export const BITRIX_LIMITS = {
  MIN_REQUEST_INTERVAL_MS,
  DEFAULT_PAGE_SIZE,
  MAX_RETRIES,
  RETRY_DELAY_MS,
};
