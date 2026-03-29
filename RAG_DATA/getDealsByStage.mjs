import curdate from "../CurMonDat.mjs";
import { bitrixList } from "./bitrixClient.mjs";

const DEAL_FIELDS = [
  "ID",
  "TITLE",
  "OPPORTUNITY",
  "COMPANY_ID",
  "CONTACT_ID",
  "BEGINDATE",
  "CLOSEDATE",
  "COMMENTS",
  "SOURCE_ID",
  "IS_RETURN_CUSTOMER",
  "UF_CRM_1656758173690",   // описание товара/услуги
  "UF_CRM_1650729604012",   // направление: Печатка/Цех/Дизайн/Аутсорсинг
  "UF_CRM_1674567201228",   // тип вывески: Статика/Динамика/Экран
  "UF_CRM_1684069704806",   // способ оплаты (приход)
  "UF_CRM_1684069832365",   // способ оплаты (расход)
  "UF_CRM_1656757942019",   // ссылки на фото
];

const DIRECTION_MAP = { 108: "Печатка", 110: "Цех", 112: "Дизайн", 114: "Аутсорсинг" };
const SIGN_TYPE_MAP = { 140: "Статика", 142: "Динамика", 144: "Экран" };
const PAYMENT_MAP  = { 152: "Наличный", 154: "Безналичный", 156: "Наличный", 158: "Безналичный", 160: "Бартер", 164: "Инвестиция", 268: "Бартер" };

function resolveEnum(raw, map) {
  if (!raw) return "";
  const ids = Array.isArray(raw) ? raw : [raw];
  return ids.map((id) => map[id] || String(id)).filter(Boolean).join(", ");
}

function extractImageUrls(rawImages) {
  if (!rawImages) return [];
  // Картинки будут торчать снаружи через симлинк b24_images
  // Пример исхода: "images/obem_bukvi/19/1.jpg"
  // Пример результата: "https://labus.pro/b24_images/images/obem_bukvi/19/1.jpg"
  return String(rawImages).split(',').map(s => s.trim()).filter(Boolean).map(s => `https://labus.pro/b24_images/${s.replace(/^\/+/, '')}`);
}

function stripBbCode(text) {
  if (!text) return "";
  return text.replace(/\[.*?\]/g, "").replace(/\s+/g, " ").trim();
}

function uniqueIds(values) {
  return [...new Set(values.filter(Boolean))];
}

function buildLookup(items, idKey, valueKey) {
  const lookup = {};

  for (const item of items) {
    lookup[item[idKey]] = item[valueKey];
  }

  return lookup;
}

export default async function getDealsByStage(stageId) {
  const { firstDay, lastDay } = await curdate();

  const deals = await bitrixList("crm.deal.list", {
    order: { ID: "ASC" },
    filter: {
      STAGE_ID: stageId,
      ">=CLOSEDATE": firstDay,
      "<=CLOSEDATE": lastDay,
    },
    select: DEAL_FIELDS,
  });

  if (!deals.length) {
    return [];
  }

  const companyIds = uniqueIds(deals.map((deal) => deal.COMPANY_ID));
  const contactIds = uniqueIds(deals.map((deal) => deal.CONTACT_ID));

  const chunkArray = (arr, size) => Array.from({ length: Math.ceil(arr.length / size) }, (_, i) => arr.slice(i * size, i * size + size));

  const companies = [];
  for (const chunk of chunkArray(companyIds, 50)) {
    const res = await bitrixList("crm.company.list", {
      order: { ID: "ASC" },
      filter: { ID: chunk },
      select: ["ID", "TITLE", "INDUSTRY"],
    });
    companies.push(...res);
  }

  const contacts = [];
  for (const chunk of chunkArray(contactIds, 50)) {
    const res = await bitrixList("crm.contact.list", {
      order: { ID: "ASC" },
      filter: { ID: chunk },
      select: ["ID", "NAME", "LAST_NAME"],
    });
    contacts.push(...res);
  }

  const companyLookup = buildLookup(companies, "ID", "TITLE");
  const contactLookup = new Map(
    contacts.map((c) => [c.ID, [c.NAME, c.LAST_NAME].filter(Boolean).join(" ") || c.ID])
  );

  return deals.map((deal) => ({
    ...deal,
    COMPANY_ID: companyLookup[deal.COMPANY_ID] ?? deal.COMPANY_ID ?? "",
    CONTACT_ID: contactLookup.get(String(deal.CONTACT_ID)) ?? deal.CONTACT_ID ?? "",
    DESCRIPTION: deal.UF_CRM_1656758173690 ?? "",
    COMMENTS: stripBbCode(deal.COMMENTS),
    DIRECTION: resolveEnum(deal.UF_CRM_1650729604012, DIRECTION_MAP),
    SIGN_TYPE: resolveEnum(deal.UF_CRM_1674567201228, SIGN_TYPE_MAP),
    PAYMENT_TYPE: resolveEnum(deal.UF_CRM_1684069704806, PAYMENT_MAP),
    SOURCE_ID: deal.SOURCE_ID ?? "",
    IS_RETURN_CUSTOMER: deal.IS_RETURN_CUSTOMER === "Y" ? "Y" : "",
    IMAGE_URLS: extractImageUrls(deal.UF_CRM_1656757942019),
  }));
}
