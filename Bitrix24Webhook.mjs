const DEFAULT_BITRIX24_WEBHOOK_URL = "https://labus.bitrix24.ru/rest/5/a184a56co9ghrehs/";

function getWebhookUrl() {
  const rawValue = String(process.env.BITRIX24_WEBHOOK_URL ?? DEFAULT_BITRIX24_WEBHOOK_URL).trim();
  if (!rawValue) {
    throw new Error("BITRIX24_WEBHOOK_URL is empty.");
  }

  return rawValue.endsWith("/") ? rawValue : `${rawValue}/`;
}

export function getBitrix24WebhookUrl() {
  return getWebhookUrl();
}

export default async function bxPostJson(method, params = {}) {
  const url = `${getWebhookUrl()}${method}/`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    throw new Error(`Bitrix24 HTTP ${response.status} ${response.statusText}`);
  }

  const result = await response.json();
  if (result?.error) {
    throw new Error(result.error_description || result.error);
  }

  return result;
}
