/**
 * buildSmetaTemplates.mjs
 *
 * Builds canonical smeta templates + price statistics per category
 * from offers.csv and orders.csv, using goods.csv as the catalog.
 *
 * Output: RAG_ANALYTICS/output/smeta_templates.json
 *
 * Category extraction:
 *   - Group offers.csv / orders.csv rows by `ID` → full smeta per offer/order.
 *   - Each offer has a `TITLE` (e.g. 'Световая вывеска "ШАУРМА"').
 *   - Category name = TITLE with quoted content, dimensions and material
 *     specs (ПВХ3/ПВХ5/ПВХ8) stripped, first 2-3 words kept.
 *   - Offers with the same normalized category key are clustered together.
 *
 * Canonical smeta:
 *   - From each category's offers, pick the one with MIN unique GOOD_IDs
 *     (the cleanest / most canonical smeta). Prefer orders.csv over offers.csv
 *     when available (completed deals > proposals).
 *
 * Price statistics per GOOD_ID in category:
 *   - mean, median, weighted_mean (by quantity × freshness), trimmed_mean (10%)
 *   - final = arithmetic average of all four metrics
 *   - std, cv = std / final → confidence (cv < 0.15 → high, < 0.3 → medium)
 */
import { join, resolve } from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { readCsv, writeJsonAtomically } from "./lib/io.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, "..");
const DATA_DIR = join(ROOT, "RAG_DATA");
const OUTPUT_DIR = join(__dirname, "output");

const FRESHNESS_HALF_LIFE_DAYS = 365; // weight = 0.5 at 1 year old
const MIN_CATEGORY_DEALS = 3; // skip categories with < N deals

// Anchor heads = "голова" существительного категории. extractCategoryKey()
// сливает любой хвост после якоря, оставляя максимум [прилагательное, якорь].
// Цель: «Объемная Буква Лицом/Лицевой/Зеркального/Световая» → «Объемная Буква»,
//       «Титульная Вывеска Оргстекла/Композита/Пвх» → «Титульная Вывеска».
const ANCHOR_HEADS = new Set([
  // вывески / буквы / таблички
  "вывеска", "вывески",
  "буква", "буквы",
  "табличка", "таблички",
  "штендер", "штендеры",
  "стенд", "стенды",
  "логотип", "логотипы",
  "табло",
  "лайтбокс",
  "панель", "панели",
  // печатная продукция
  "листовка", "листовки",
  "визитка", "визитки",
  "каталог", "каталоги",
  "флаер", "флаеры",
  "флайер", "флайеры",
  "буклет", "буклеты",
  "брошюра", "брошюры",
  "открытка", "открытки",
  "календарь", "календари",
  "плакат", "плакаты",
  "постер", "постеры",
  "брендбук",
  "меню",
  "афиша", "афиши",
  "наклейка", "наклейки",
  "стикер", "стикеры",
  "этикетка", "этикетки",
  "ценник", "ценники",
  "бирка", "бирки",
  "конверт", "конверты",
  "папка", "папки",
  "блокнот", "блокноты",
  "ежедневник", "ежедневники",
  // оформление / широкий формат
  "баннер", "баннеры",
  "плёнка", "пленка", "плёнки", "пленки",
  "окл", "оклейка",
  "постамент",
  // мерч
  "футболка", "футболки",
  "кружка", "кружки",
  "ручка", "ручки",
  "значок", "значки",
  // услуги/работы
  "монтаж",
  "дизайн",
  "дизайн-макет",
  "макет", "макеты",
  "визуализация",
  "подготовка",
  "сборка",
  "печать",
  "доставка",
]);

/**
 * Нормализует грубый ключ категории до якорной формы [прилагательное, якорь].
 * Если якорь не найден — берём первые 2 слова как есть.
 */
function normalizeCategoryKey(rawKey) {
  if (!rawKey) return "";
  const words = rawKey.split(/\s+/).filter(Boolean);
  if (words.length === 0) return "";
  const anchorIdx = words.findIndex((w) => ANCHOR_HEADS.has(w));
  if (anchorIdx >= 0) {
    const start = Math.max(0, anchorIdx - 1);
    return words.slice(start, anchorIdx + 1).join(" ");
  }
  return words.slice(0, 2).join(" ");
}

// ─── Manual category map (labus.pro taxonomy) ────────────────────────────
// Точная иерархия с сайта labus.pro / sitemap.xml. Каждой категории заданы
// «обязательные» (allOf) и «любые» (anyOf) regex по lowercase TITLE.
// Порядок важен — первое совпадение побеждает. Более специфичные категории
// идут раньше общих («Объёмные буквы» раньше «Световые короба» раньше «Вывески»).
const LABUS_CATEGORIES = [
  // ── НАРУЖНАЯ РЕКЛАМА ─────────────────────────────────────────────
  { parent: "Наружная реклама", name: "Неоновые вывески",
    anyOf: [/неон/i] },
  { parent: "Наружная реклама", name: "Объемные буквы",
    allOf: [/буква|буквы|буквенн/i],
    anyOf: [/объ[её]мн/i, /световая\s+буква/i, /3d/i, /акрил/i, /пвх/i] },
  { parent: "Наружная реклама", name: "Титульные вывески",
    allOf: [/титульн/i, /вывеск/i] },
  { parent: "Наружная реклама", name: "Световые короба",
    anyOf: [/лайтбокс/i, /светов(ой|ые)\s+короб/i, /короб\s+светов/i] },
  { parent: "Наружная реклама", name: "Панель-кронштейн",
    anyOf: [/кронштейн/i, /панель[-\s]*кронштейн/i] },
  { parent: "Наружная реклама", name: "Штендеры",
    anyOf: [/штендер/i] },
  { parent: "Наружная реклама", name: "Флаги",
    anyOf: [/\bфлаг(и|ов|ом)?\b/i, /флагшток/i] },
  { parent: "Наружная реклама", name: "Брендирование авто",
    anyOf: [/брендирован.*авто/i, /оклейк.*авто/i, /авто.*оклейк/i, /car\s*wrap/i] },
  { parent: "Наружная реклама", name: "Реклама на щитах",
    anyOf: [/билборд/i, /щит\s+реклам/i, /реклам.*щит/i, /3х6/i] },
  // П7.5: «Вывески» как самостоятельная категория УДАЛЕНА. Запросы "вывеска"
  // без маркеров объёма/света/композита не должны закрываться коротким
  // шаблоном на 6000₽ (выдавали 2 позиции: макет + монтаж). Падают в LLM.
  { parent: "Наружная реклама", name: "Световые вывески",
    allOf: [/вывеск/i], anyOf: [/светов/i, /подсветк/i, /объ[её]мн/i, /композит/i, /контражур/i, /led|лед/i] },

  // ── ВНУТРЕННЯЯ РЕКЛАМА ───────────────────────────────────────────
  { parent: "Внутренняя реклама", name: "Таблички с аппликацией",
    allOf: [/табличк/i, /аппликац/i] },
  { parent: "Внутренняя реклама", name: "Таблички с гравировкой",
    allOf: [/табличк/i, /гравир/i] },
  { parent: "Внутренняя реклама", name: "Таблички прозрачные",
    allOf: [/табличк/i], anyOf: [/прозрачн/i, /оргстекл/i, /акрил.*прозрачн/i] },
  { parent: "Внутренняя реклама", name: "Полиграфические таблички",
    allOf: [/табличк/i], anyOf: [/полиграф/i, /бумаг/i, /картон/i] },
  { parent: "Внутренняя реклама", name: "Таблички",
    allOf: [/табличк/i] },
  { parent: "Внутренняя реклама", name: "Холст",
    anyOf: [/\bхолст/i] },
  { parent: "Внутренняя реклама", name: "Стенды настенные",
    allOf: [/стенд/i, /настенн/i] },
  { parent: "Внутренняя реклама", name: "Стенды напольные",
    allOf: [/стенд/i], anyOf: [/напольн/i, /мобильн/i] },
  { parent: "Внутренняя реклама", name: "Настольные стойки",
    anyOf: [/настольн.*стойк/i, /стойк.*настольн/i, /менюхолдер/i] },
  { parent: "Внутренняя реклама", name: "Стойки",
    anyOf: [/стойк(а|и|у)/i] },
  { parent: "Внутренняя реклама", name: "Пресс-волл",
    anyOf: [/пресс-?волл/i, /press[\s-]?wall/i] },
  { parent: "Внутренняя реклама", name: "Световые панели",
    allOf: [/панель|панел/i, /светов/i] },
  { parent: "Внутренняя реклама", name: "Стенды",
    allOf: [/стенд/i] },

  // ── ТИПОГРАФИЯ ───────────────────────────────────────────────────
  { parent: "Типография", name: "Визитки",
    anyOf: [/визитк/i] },
  { parent: "Типография", name: "Листовки",
    anyOf: [/листовк/i] },
  { parent: "Типография", name: "Флаеры",
    anyOf: [/флаер|флайер/i] },
  { parent: "Типография", name: "Брошюры",
    anyOf: [/брошюр/i] },
  { parent: "Типография", name: "Буклеты",
    anyOf: [/буклет/i] },
  { parent: "Типография", name: "Открытки",
    anyOf: [/открытк/i] },
  { parent: "Типография", name: "Календари",
    anyOf: [/календар/i] },
  { parent: "Типография", name: "Ежедневники",
    anyOf: [/ежедневник/i] },
  { parent: "Типография", name: "Блокноты",
    anyOf: [/блокнот/i] },
  { parent: "Типография", name: "Бейджи",
    anyOf: [/бейдж/i] },
  { parent: "Типография", name: "Бланки",
    anyOf: [/бланк/i] },
  { parent: "Типография", name: "Папки",
    anyOf: [/\bпапк(а|и|у|ой)/i] },
  { parent: "Типография", name: "Планшеты",
    anyOf: [/планшет/i] },
  { parent: "Типография", name: "Пакеты крафт",
    allOf: [/пакет/i, /крафт/i] },
  { parent: "Типография", name: "Пакеты ПВД",
    allOf: [/пакет/i, /пвд/i] },
  { parent: "Типография", name: "Пакеты майка",
    allOf: [/пакет/i, /майк/i] },
  { parent: "Типография", name: "Пакеты",
    anyOf: [/пакет/i] },
  { parent: "Типография", name: "Меню",
    anyOf: [/\bменю\b/i] },
  { parent: "Типография", name: "Этикетки и стикеры",
    anyOf: [/этикетк/i, /стикер/i, /наклейк/i] },
  { parent: "Типография", name: "Пластиковые карты",
    allOf: [/карт(а|ы|очк)/i, /пластик/i] },
  { parent: "Типография", name: "Скотч",
    anyOf: [/скотч/i] },

  // ── ДИЗАЙН ───────────────────────────────────────────────────────
  { parent: "Дизайн", name: "Баннеры",
    anyOf: [/баннер/i] },
  { parent: "Дизайн", name: "Плакаты",
    anyOf: [/плакат|постер/i] },
  { parent: "Дизайн", name: "Веб-дизайн",
    anyOf: [/веб-?дизайн|web-?design/i] },
  { parent: "Дизайн", name: "3D моделирование",
    anyOf: [/3d.*модел/i, /моделир/i] },
  { parent: "Дизайн", name: "Упаковка",
    anyOf: [/упаковк/i] },
  { parent: "Дизайн", name: "Дизайн-макет",
    allOf: [/макет/i], anyOf: [/дизайн/i] },
  { parent: "Дизайн", name: "Полиграфия",
    anyOf: [/полиграф/i] },

  // ── БРЕНДИНГ ─────────────────────────────────────────────────────
  // П7.5: split «Логотип» — дорогие under-key проекты (брифинг, концепции,
  // фирменный стиль) отделены от дешёвой отрисовки. Порядок важен: более
  // специфичная «под ключ» идёт первой.
  { parent: "Брендинг", name: "Логотип под ключ",
    allOf: [/логотип/i],
    anyOf: [/под\s*ключ/i, /бриф/i, /концепц/i, /фирменн.*стил/i, /брендбук/i,
            /айдентик/i, /разработк.*с\s*нул/i, /креатив/i, /нейминг/i] },
  { parent: "Брендинг", name: "Брендбук",
    anyOf: [/брендбук|brand[\s-]?book/i] },
  { parent: "Брендинг", name: "Логотип",
    anyOf: [/логотип/i],
    // П7.6: исключаем мерч-носители («Ручка с логотипом», «Кружка с логотипом»),
    // которые иначе контаминируют keywords_text и эмбеддинг категории.
    notAny: [/ручк|кружк|термокружк|футболк|кепк|бейсболк|шоппер|худи|свитшот|толстовк|поло|кофт|майк|магнит|шоколад|сахар|бутылк|повербанк|значок|значк|брелок|пакет|ежедневник|блокнот|зажигалк|флешк|наклейк|стикер|этикет|визитк|баннер|табличк|вывеск|футляр|плед|шапк|зонт|носок|носк|ложк|вилк|нож/i] },
  { parent: "Брендинг", name: "Фирменный стиль",
    anyOf: [/фирменн.*стил/i, /стил.*фирм/i] },
  { parent: "Брендинг", name: "Иллюстрации",
    anyOf: [/иллюстрац/i] },
  { parent: "Брендинг", name: "Презентации",
    anyOf: [/презентац/i] },
  { parent: "Брендинг", name: "Фотосъемка",
    anyOf: [/фотосъ[её]мк|фотосессия/i] },
  { parent: "Брендинг", name: "Разработка сайтов",
    anyOf: [/разработк.*сайт|сайт.*разработк|landing|лендинг/i] },
  { parent: "Брендинг", name: "SMM",
    anyOf: [/\bsmm\b|соц.*сет/i] },

  // ── МЕРЧ ─────────────────────────────────────────────────────────
  { parent: "Мерч", name: "Кружки",
    anyOf: [/кружк/i] },
  { parent: "Мерч", name: "Термокружки",
    anyOf: [/термокружк|термос/i] },
  { parent: "Мерч", name: "Бутылки для воды",
    allOf: [/бутылк/i] },
  { parent: "Мерч", name: "Ручки",
    anyOf: [/\bручк(а|и|у|ой)\b/i] },
  { parent: "Мерч", name: "Футболки",
    anyOf: [/футболк/i] },
  { parent: "Мерч", name: "Бейсболки",
    anyOf: [/бейсболк|кепк/i] },
  { parent: "Мерч", name: "Шопперы",
    anyOf: [/шоппер/i] },
  { parent: "Мерч", name: "Свитшоты",
    anyOf: [/свитшот|худи/i] },
  { parent: "Мерч", name: "Повербанки",
    anyOf: [/повербанк|power[\s-]?bank/i] },
  { parent: "Мерч", name: "Магниты",
    anyOf: [/магнит/i] },
  { parent: "Мерч", name: "Шоколад с логотипом",
    anyOf: [/шоколад/i] },
];

/**
 * Match a TITLE against the labus.pro manual taxonomy.
 * Returns category name (e.g. "Объемные буквы") or null.
 */
function mapToLabusCategory(title) {
  if (!title) return null;
  const t = String(title).toLowerCase();
  for (const cat of LABUS_CATEGORIES) {
    const allOk = !cat.allOf || cat.allOf.every((re) => re.test(t));
    if (!allOk) continue;
    const anyOk = !cat.anyOf || cat.anyOf.some((re) => re.test(t));
    if (!anyOk) continue;
    const notOk = !cat.notAny || !cat.notAny.some((re) => re.test(t));
    if (!notOk) continue;
    return cat.name;
  }
  return null;
}

// ─── Category extraction ─────────────────────────────────────────────────

/**
 * Extract a normalized category key from an offer TITLE.
 * Priority: (1) manual labus.pro mapping → (2) anchor-head normalization.
 */
function extractCategoryKey(title) {
  if (!title) return "";

  // (1) Manual labus.pro taxonomy — точный матч по полному TITLE
  const manual = mapToLabusCategory(title);
  if (manual) return "labus:" + manual; // prefix to bypass prettify lowercasing

  // (2) Fallback: anchor-head normalization
  let s = String(title).trim();

  // Strip all quoted content (both « » and "")
  s = s.replace(/«[^»]*»/g, " ");
  s = s.replace(/"[^"]*"/g, " ");
  s = s.replace(/""[^"]*""/g, " ");

  // Strip material specs: ПВХ3, ПВХ5, ПВХ8, Акрил3, etc.
  s = s.replace(/\b(ПВХ|Акрил|Композит)\d+\b/gi, " ");

  // Strip dimensions: 85х100 см, 10x20, 50 см
  s = s.replace(/\b\d+[xх×]\d+(\s*см)?\b/gi, " ");
  s = s.replace(/\b\d+\s*см\b/gi, " ");
  s = s.replace(/\b\d+\s*мм\b/gi, " ");
  s = s.replace(/\b\d+\s*мп\b/gi, " ");

  // Strip pure numbers and punctuation
  s = s.replace(/[,;:()]/g, " ");
  s = s.replace(/\s+/g, " ").trim().toLowerCase();

  // Keep meaningful words, then normalize to anchor head form
  const words = s.split(/\s+/).filter((w) => w.length >= 3);
  const rawKey = words.slice(0, 5).join(" ");
  return normalizeCategoryKey(rawKey);
}

/** Pretty name: capitalize first letter of each word, or use labus: prefix as-is. */
function prettifyCategoryName(key) {
  if (key.startsWith("labus:")) return key.slice(6);
  return key
    .split(" ")
    .map((w) => (w.length > 0 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

// ─── Deal grouping ───────────────────────────────────────────────────────

/**
 * Group CSV rows by `ID` column → reconstruct full smetas.
 * Returns: Map<dealId, {id, title, direction, closeDate, lineItems: []}>
 */
function groupByDeal(rows, datasetName) {
  const deals = new Map();
  for (const row of rows) {
    const id = row.ID;
    if (!id) continue;
    if (!deals.has(id)) {
      deals.set(id, {
        id,
        dataset: datasetName,
        title: row.TITLE || "",
        direction: row.DIRECTION || row["Направление"] || "",
        closeDate: row.CLOSEDATE || row.BEGINDATE || "",
        opportunity: parseFloat(row.OPPORTUNITY) || 0,
        lineItems: [],
      });
    }
    // PRODUCT_ID is the FK into goods catalog (shared across line items for
    // the same goods record). GOOD_ID is per-line Bitrix24 internal ID.
    const good_id = row.PRODUCT_ID || "";
    const product_name = row.PRODUCT_NAME || row.NAME || "";
    const price = parseFloat(row.PRICE) || 0;
    const quantity = parseFloat(row.QUANTITY) || 0;
    if (!good_id || !product_name) continue;
    deals.get(id).lineItems.push({
      good_id,
      product_name,
      price,
      quantity,
      section_id: row.SECTION_ID || "",
      section_name: row.SECTION_NAME || "",
      parent_section: row.PARENT_SECTION || "",
      direction: row["Направление"] || row.DIRECTION || "",
    });
  }
  return deals;
}

// ─── Freshness weighting ─────────────────────────────────────────────────

function freshnessWeight(dateStr, nowMs) {
  if (!dateStr) return 0.5;
  const t = Date.parse(dateStr);
  if (isNaN(t)) return 0.5;
  const ageDays = Math.max(0, (nowMs - t) / (1000 * 60 * 60 * 24));
  // exponential decay: half-life = FRESHNESS_HALF_LIFE_DAYS
  return Math.pow(0.5, ageDays / FRESHNESS_HALF_LIFE_DAYS);
}

// ─── Statistics ──────────────────────────────────────────────────────────

function computePriceStats(samples) {
  // samples: [{price, quantity, freshness}]
  if (samples.length === 0) return null;
  const prices = samples.map((s) => s.price).filter((p) => p > 0);
  if (prices.length === 0) return null;

  prices.sort((a, b) => a - b);
  const n = prices.length;

  const mean = prices.reduce((a, b) => a + b, 0) / n;
  const median =
    n % 2 === 0
      ? (prices[n / 2 - 1] + prices[n / 2]) / 2
      : prices[Math.floor(n / 2)];

  // Weighted mean: by quantity × freshness
  let totalWeight = 0;
  let weightedSum = 0;
  for (const s of samples) {
    if (!s.price || s.price <= 0) continue;
    const w = (s.quantity || 1) * (s.freshness || 0.5);
    weightedSum += s.price * w;
    totalWeight += w;
  }
  const weighted_mean = totalWeight > 0 ? weightedSum / totalWeight : mean;

  // Trimmed mean: drop 10% from each side
  const trimK = Math.floor(n * 0.1);
  const trimmed = prices.slice(trimK, n - trimK);
  const trimmed_mean =
    trimmed.length > 0
      ? trimmed.reduce((a, b) => a + b, 0) / trimmed.length
      : mean;

  const final = (mean + median + weighted_mean + trimmed_mean) / 4;

  const variance =
    prices.reduce((sum, p) => sum + (p - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const cv = final > 0 ? std / final : 0;

  let confidence = "low";
  if (cv < 0.15) confidence = "high";
  else if (cv < 0.3) confidence = "medium";

  return {
    mean: round2(mean),
    median: round2(median),
    weighted_mean: round2(weighted_mean),
    trimmed_mean: round2(trimmed_mean),
    final: round2(final),
    std: round2(std),
    cv: round4(cv),
    confidence,
    sample_size: n,
  };
}

const round2 = (x) => Math.round(x * 100) / 100;
const round4 = (x) => Math.round(x * 10000) / 10000;

// ─── П7.6: SEED_DEALS — hand-curated canonical templates ────────────────
// Для категорий, где дефолтный p60-picker даёт нерепрезентативный canonical
// (например "Логотип" — под ключ vs дешёвая отрисовка). Ключ = category_name
// после prettifyCategoryName; значение = {dataset, seed_ids, min_overlap}.
//
// Логика: canonical positions = union line_items из seed сделок; для каждого
// good_id берём медианное quantity и наиболее частое product_name среди seeds.
// Price stats — из ВСЕХ сделок кластера (offers+orders), как раньше, но
// дополнительно из самих seed сделок если good_id отсутствует в кластере.
const SEED_DEALS = {
  "Логотип": {
    dataset: "offers",
    // canonical_id = конкретная сделка-образец (4–6 позиций, 16–30к).
    // Остальные seed_ids идут в price stats и keywords.
    canonical_id: "21208",
    seed_ids: ["21208", "46416", "46414", "21210", "46428", "46426", "21214", "46424", "46422"],
  },
};

/**
 * Builds a synthetic canonical "deal" from seed deal IDs.
 * Returns {id, dataset, title, lineItems} or null if no seeds found.
 */
function buildCanonicalFromSeeds(cluster, seedConfig, offerDeals, orderDeals) {
  const sourceMap = seedConfig.dataset === "orders" ? orderDeals : offerDeals;
  const seeds = [];
  for (const sid of seedConfig.seed_ids) {
    const d = sourceMap.get(sid);
    if (d) seeds.push(d);
  }
  if (seeds.length === 0) return null;

  // canonical_id = явно указанная сделка-образец. Остальные seeds
  // инжектятся в cluster.deals для обогащения price_stats и frequency.
  const canonical = sourceMap.get(seedConfig.canonical_id);
  if (!canonical || canonical.lineItems.length === 0) return null;

  return { ...canonical, _isSeed: true, _seedDeals: seeds };
}

// ─── Main build ──────────────────────────────────────────────────────────

async function main() {
  console.log("Loading CSVs...");
  const offersRows = await readCsv(join(DATA_DIR, "offers.csv"));
  const ordersRows = await readCsv(join(DATA_DIR, "orders.csv"));
  const goodsRows = await readCsv(join(DATA_DIR, "goods.csv"));
  console.log(
    `  offers: ${offersRows.length}, orders: ${ordersRows.length}, goods: ${goodsRows.length}`,
  );

  // Build goods lookup by PRODUCT_ID (goods.csv uses PRODUCT_ID, offers/orders use GOOD_ID)
  const goodsByProductId = new Map();
  for (const g of goodsRows) {
    const pid = g.PRODUCT_ID;
    if (!pid) continue;
    goodsByProductId.set(pid, {
      product_id: pid,
      product_name: g.PRODUCT_NAME || g.NAME || "",
      section_name: g.SECTION_NAME || "",
      parent_section: g.PARENT_SECTION || "",
      direction: g["Направление"] || g.DIRECTION || "",
      base_price: parseFloat(g.BASE_PRICE) || 0,
      cost_price: parseFloat(g.COST_PRICE) || 0,
    });
  }
  console.log(`  goods indexed: ${goodsByProductId.size}`);

  // Group offers/orders by deal
  const offerDeals = groupByDeal(offersRows, "offers");
  const orderDeals = groupByDeal(ordersRows, "orders");
  console.log(
    `  offer deals: ${offerDeals.size}, order deals: ${orderDeals.size}`,
  );

  // Cluster deals by category key (from TITLE)
  const categoryClusters = new Map(); // key → {key, name, deals: []}
  const nowMs = Date.now();

  const collectDeals = (dealMap) => {
    for (const deal of dealMap.values()) {
      const key = extractCategoryKey(deal.title);
      if (!key) continue;
      if (!categoryClusters.has(key)) {
        categoryClusters.set(key, {
          key,
          name: prettifyCategoryName(key),
          deals: [],
        });
      }
      categoryClusters.get(key).deals.push(deal);
    }
  };
  collectDeals(offerDeals);
  collectDeals(orderDeals);

  console.log(`  unique category keys: ${categoryClusters.size}`);

  // Build categories with stats
  const categories = [];
  let totalDealsAnalyzed = 0;

  for (const cluster of categoryClusters.values()) {
    if (cluster.deals.length < MIN_CATEGORY_DEALS) continue;

    // П7.6: SEED_DEALS override — hand-curated canonical for specific categories
    let seedCanonical = null;
    const seedCfg = SEED_DEALS[cluster.name];
    if (seedCfg) {
      seedCanonical = buildCanonicalFromSeeds(cluster, seedCfg, offerDeals, orderDeals);
      if (seedCanonical) {
        console.log(
          `  [SEED] ${cluster.name}: ${seedCanonical.lineItems.length} canonical positions from ${seedCfg.seed_ids.length} seed ids`,
        );
        // Inject seed deals into cluster.deals if missing — чтобы их line_items
        // участвовали в price_stats и good_id frequency.
        const haveIds = new Set(cluster.deals.map((d) => d.id));
        for (const sd of seedCanonical._seedDeals) {
          if (!haveIds.has(sd.id)) cluster.deals.push(sd);
        }
      }
    }

    // П7.5: canonical smeta = MEDIAN deal by unique positions count
    // (было: минимальная сделка — давала вырожденные шаблоны типа
    // «Макет+Монтаж=6000₽» для «Вывесок»). Берём сделку с числом позиций
    // ближе к медиане пула — это репрезентативный реальный заказ.
    const MIN_CANONICAL_POSITIONS = 2;
    const dealsWithEnoughPos = cluster.deals.filter(
      (d) => new Set(d.lineItems.map((li) => li.good_id)).size >= MIN_CANONICAL_POSITIONS,
    );
    const candidatePool = dealsWithEnoughPos.length > 0 ? dealsWithEnoughPos : cluster.deals;
    const byPosAsc = [...candidatePool].sort((a, b) => {
      const uniqA = new Set(a.lineItems.map((li) => li.good_id)).size;
      const uniqB = new Set(b.lineItems.map((li) => li.good_id)).size;
      return uniqA - uniqB;
    });
    // П7.5: canonical = сделка с positions ≥ p60 по числу позиций и
    // opportunity ближе всего к p50 по сумме. Это отсекает
    // вырожденные 2-позиционные сделки И гигантские тиражи-выбросы.
    const sortedByPos = [...candidatePool].map((d) => ({
      deal: d,
      pos: new Set(d.lineItems.map((li) => li.good_id)).size,
    })).sort((a, b) => a.pos - b.pos);
    const p60Idx = Math.floor(sortedByPos.length * 0.6);
    const p60Pos = sortedByPos[Math.min(p60Idx, sortedByPos.length - 1)].pos;
    const oppsSorted = candidatePool
      .map((d) => d.opportunity || 0)
      .filter((v) => v > 0)
      .sort((a, b) => a - b);
    const medianOpp =
      oppsSorted.length > 0
        ? oppsSorted[Math.floor(oppsSorted.length * 0.5)]
        : 0;
    const eligible = sortedByPos
      .filter((x) => x.pos >= p60Pos)
      .map((x) => x.deal);
    eligible.sort((a, b) => {
      if (a.dataset !== b.dataset) return a.dataset === "orders" ? -1 : 1;
      const da = Math.abs((a.opportunity || 0) - medianOpp);
      const db = Math.abs((b.opportunity || 0) - medianOpp);
      return da - db;
    });
    const canonicalDeal =
      seedCanonical ||
      eligible[0] ||
      sortedByPos[sortedByPos.length - 1].deal;
    if (!canonicalDeal || canonicalDeal.lineItems.length === 0) continue;
    // П7.5: hard minimum — канонический шаблон с одной позицией бесполезен
    // (даёт вырожденные оценки типа «Логотип под ключ: 1 поз, 4000₽»).
    const canonicalUniq = new Set(
      canonicalDeal.lineItems.map((li) => li.good_id),
    ).size;
    if (canonicalUniq < 2) continue;

    // Aggregate prices by GOOD_ID across ALL deals in this category
    const priceSamplesByGoodId = new Map();
    for (const deal of cluster.deals) {
      const freshness = freshnessWeight(deal.closeDate, nowMs);
      for (const li of deal.lineItems) {
        if (!li.good_id || !li.price || li.price <= 0) continue;
        if (!priceSamplesByGoodId.has(li.good_id)) {
          priceSamplesByGoodId.set(li.good_id, []);
        }
        priceSamplesByGoodId.get(li.good_id).push({
          price: li.price,
          quantity: li.quantity,
          freshness,
        });
      }
    }

    // Frequency: in how many % of category deals does each GOOD_ID appear
    const goodIdFrequency = new Map();
    for (const deal of cluster.deals) {
      const seen = new Set(deal.lineItems.map((li) => li.good_id));
      for (const gid of seen) {
        goodIdFrequency.set(gid, (goodIdFrequency.get(gid) || 0) + 1);
      }
    }

    // Build canonical template positions
    const positions = [];
    for (const li of canonicalDeal.lineItems) {
      if (!li.good_id) continue;
      const samples = priceSamplesByGoodId.get(li.good_id) || [];
      const stats = computePriceStats(samples);
      if (!stats) continue;
      const goodMeta = goodsByProductId.get(li.good_id) || {};
      const freq = goodIdFrequency.get(li.good_id) || 0;
      // Fix 13: strip trailing parenthesized client/author names
      const cleanName = String(li.product_name || "")
        .replace(/\s*\([^)]*\)\s*$/, "")
        .trim();
      positions.push({
        good_id: li.good_id,
        product_name: cleanName,
        section_name: li.section_name || goodMeta.section_name || "",
        parent_section: li.parent_section || goodMeta.parent_section || "",
        direction: li.direction || goodMeta.direction || "",
        quantity_typical: round2(li.quantity || 1),
        unit: inferUnit(li.product_name),
        b24_section: mapDirectionToB24(li.direction || goodMeta.direction),
        frequency: round2(freq / cluster.deals.length),
        price_stats: stats,
      });
    }

    // All goods seen in category (for fallback/augmentation)
    const allGoods = [];
    for (const [gid, samples] of priceSamplesByGoodId.entries()) {
      const stats = computePriceStats(samples);
      if (!stats) continue;
      const goodMeta = goodsByProductId.get(gid) || {};
      allGoods.push({
        good_id: gid,
        product_name:
          goodMeta.product_name ||
          (cluster.deals
            .flatMap((d) => d.lineItems)
            .find((li) => li.good_id === gid)?.product_name ?? ""),
        frequency: round2((goodIdFrequency.get(gid) || 0) / cluster.deals.length),
        final: stats.final,
        sample_size: stats.sample_size,
      });
    }
    allGoods.sort((a, b) => b.frequency - a.frequency);

    const avgDealValue =
      cluster.deals.reduce((sum, d) => sum + (d.opportunity || 0), 0) /
      cluster.deals.length;

    // Keywords: top non-stopword tokens across all titles
    const keywords = extractKeywords(cluster.deals.map((d) => d.title));

    // Top-20 raw deal TITLEs by frequency (Fix 2: дать BGE-M3 живую лексику —
    // в том числе бренды/имена клиентов вроде «ШАУРМА», которых нет в keywords).
    const titleCounts = new Map();
    for (const deal of cluster.deals) {
      const t = (deal.title || "").trim();
      if (!t) continue;
      titleCounts.set(t, (titleCounts.get(t) || 0) + 1);
    }
    const topTitles = [...titleCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([t]) => t);

    // Top product names from canonical positions (по 5)
    const topProducts = positions
      .slice(0, 5)
      .map((p) => p.product_name)
      .filter(Boolean);

    // Fix 9: примеры заказов идут ПЕРВЫМИ — BGE-M3 чувствительнее к началу
    // строки, и реальные TITLE содержат бренд/контекст («ШАУРМА», «КОФЕ»),
    // чего нет в служебных компонентах вроде «лента 4-10 мп Цех».
    const keywordsText = [
      cluster.name + ".",
      topTitles.length ? "примеры заказов: " + topTitles.slice(0, 10).join(" | ") + "." : "",
      keywords.length ? "ключевые слова: " + keywords.join(", ") + "." : "",
      topProducts.length ? "компоненты: " + topProducts.join("; ") + "." : "",
    ]
      .filter(Boolean)
      .join(" ");

    categories.push({
      category_id: cluster.key,
      category_name: cluster.name,
      keywords,
      keywords_text: keywordsText,
      sample_titles: topTitles,
      deals_count: cluster.deals.length,
      avg_deal_value: round2(avgDealValue),
      canonical_smeta: {
        source_deal_id: canonicalDeal.id,
        source_deal_type: canonicalDeal.dataset,
        source_title: canonicalDeal.title,
        positions_count: positions.length,
        total:
          round2(
            positions.reduce(
              (s, p) => s + (p.price_stats.final || 0) * (p.quantity_typical || 1),
              0,
            ),
          ),
        positions,
      },
      all_goods_in_category: allGoods.slice(0, 50),
    });

    totalDealsAnalyzed += cluster.deals.length;
  }

  // Sort categories by deals_count desc
  categories.sort((a, b) => b.deals_count - a.deals_count);

  const output = {
    built_at: new Date().toISOString(),
    total_categories: categories.length,
    total_deals_analyzed: totalDealsAnalyzed,
    min_category_deals: MIN_CATEGORY_DEALS,
    freshness_half_life_days: FRESHNESS_HALF_LIFE_DAYS,
    categories,
  };

  const outPath = join(OUTPUT_DIR, "smeta_templates.json");
  await writeJsonAtomically(outPath, output);
  console.log(
    `\n✓ Wrote ${categories.length} categories (${totalDealsAnalyzed} deals) → ${outPath}`,
  );

  // Print top 10 categories summary
  console.log("\nTop 10 categories by deal count:");
  for (const c of categories.slice(0, 10)) {
    console.log(
      `  ${c.deals_count.toString().padStart(4)} · ${c.category_name.padEnd(40)} ` +
        `canonical=${c.canonical_smeta.positions_count} pos, ~${c.canonical_smeta.total}₽`,
    );
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

function inferUnit(productName) {
  if (!productName) return "шт";
  const s = productName.toLowerCase();
  if (/\bмп\b/.test(s)) return "мп";
  if (/\bкв\.?м\b|\bм2\b/.test(s)) return "кв.м";
  if (/\bкомпл\b/.test(s)) return "компл";
  return "шт";
}

function mapDirectionToB24(dir) {
  if (!dir) return "";
  const d = dir.toLowerCase();
  if (d.includes("цех")) return "Цех";
  if (d.includes("рик")) return "РИК";
  if (d.includes("дизайн")) return "Дизайн";
  if (d.includes("печат")) return "Печатная";
  if (d.includes("мерч")) return "Мерч";
  if (d.includes("сольвент")) return "Сольвент";
  return dir;
}

const STOPWORDS = new Set([
  "для", "с", "из", "на", "в", "и", "по", "до", "от", "под",
  "шт", "см", "мм", "мп", "м2", "см2", "ПВХ", "пвх",
  "простой", "стандарт", "базовый", "новый",
]);

function extractKeywords(titles) {
  const counts = new Map();
  for (const title of titles) {
    const clean = String(title || "")
      .replace(/«[^»]*»/g, " ")
      .replace(/"[^"]*"/g, " ")
      .replace(/[,;:()0-9]/g, " ")
      .toLowerCase();
    for (const word of clean.split(/\s+/)) {
      if (word.length < 3) continue;
      if (STOPWORDS.has(word)) continue;
      counts.set(word, (counts.get(word) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([w]) => w);
}

main().catch((err) => {
  console.error("buildSmetaTemplates failed:", err);
  process.exit(1);
});
