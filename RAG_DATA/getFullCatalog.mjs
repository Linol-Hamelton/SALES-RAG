import { bitrixList, bitrixPost } from "./bitrixClient.mjs";

function unwrapBitrixValue(value, depth = 0) {
  if (depth > 5 || value === null || value === undefined) {
    return "";
  }

  if (Array.isArray(value) && value.length > 0) {
    return unwrapBitrixValue(value[0], depth + 1);
  }

  if (typeof value === "object") {
    if ("value" in value) {
      return unwrapBitrixValue(value.value, depth + 1);
    }

    const keys = Object.keys(value);
    if (keys.length === 1) {
      return unwrapBitrixValue(value[keys[0]], depth + 1);
    }

    return JSON.stringify(value);
  }

  return String(value);
}

async function fetchSectionMap() {
  const sections = await bitrixList("crm.productsection.list", {
    order: { ID: "ASC" },
    select: ["ID", "NAME", "SECTION_ID", "CATALOG_ID"],
  });

  const map = new Map();
  for (const s of sections) {
    map.set(String(s.ID), { name: s.NAME ?? "", parentId: s.SECTION_ID ? String(s.SECTION_ID) : "" });
  }
  return map;
}

function resolveSectionPath(sectionId, sectionMap) {
  if (!sectionId) return { section: "", parentSection: "" };
  const entry = sectionMap.get(String(sectionId));
  if (!entry) return { section: "", parentSection: "" };
  const parent = entry.parentId ? sectionMap.get(entry.parentId) : null;
  return {
    section: entry.name,
    parentSection: parent ? parent.name : "",
  };
}

async function fetchProductsBatch() {
  const products = [];
  let start = 0;
  
  while (true) {
    const cmd = {};
    for (let i = 0; i < 50; i++) {
      cmd[`page_${i}`] = `crm.product.list?order[ID]=ASC&filter[ACTIVE]=Y&select[]=ID&select[]=NAME&select[]=CATALOG_ID&select[]=SECTION_ID&select[]=PRICE&select[]=DESCRIPTION&select[]=MEASURE&select[]=PROPERTY_478&select[]=PROPERTY_480&select[]=PROPERTY_570&select[]=PROPERTY_574&select[]=PROPERTY_580&start=${start}`;
      start += 50;
    }
    
    const response = await bitrixPost("batch", { halt: 0, cmd });
    const results = response?.result?.result || {};
    
    let addedCount = 0;
    for (let i = 0; i < 50; i++) {
      const pageData = results[`page_${i}`] || [];
      products.push(...pageData);
      addedCount += pageData.length;
    }
    
    // Если вернулось меньше 50 * 50 товаров, значит, мы достигли конца каталога
    if (addedCount < 2500) {
      break; 
    }
  }
  
  return products;
}

export default async function getFullCatalog() {
  const [products, sectionMap] = await Promise.all([
    fetchProductsBatch(),
    fetchSectionMap(),
  ]);

  return products.map((product) => {
    const { section, parentSection } = resolveSectionPath(product.SECTION_ID, sectionMap);
    return {
      ID: product.ID ?? "",
      NAME: product.NAME ?? "",
      CATALOG_ID: product.CATALOG_ID ?? "",
      SECTION_ID: product.SECTION_ID ?? "",
      BASE_PRICE: product.PRICE ?? "",
      PRODUCT_DESCRIPTION: (product.DESCRIPTION ?? "").replace(/\s+/g, " ").trim(),
      "Направление": unwrapBitrixValue(product.PROPERTY_570),
      SECTION_NAME: section,
      PARENT_SECTION: parentSection,
      COEFFICIENT: unwrapBitrixValue(product.PROPERTY_478),
      WORKSHOP_SALARY: unwrapBitrixValue(product.PROPERTY_480),
      COST_PRICE: unwrapBitrixValue(product.PROPERTY_574),
      EXECUTOR: unwrapBitrixValue(product.PROPERTY_580),
    };
  });
}
