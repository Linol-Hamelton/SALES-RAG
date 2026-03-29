import { batchProcess } from "../batchUtils.mjs";
import { bitrixPost } from "./bitrixClient.mjs";

const MAX_BATCH_SIZE = 50;
const BATCH_DELAY_MS = 0; // Искусственная задержка не нужна, bitrixClient уже балансирует запросы (650мс)

export default async function getDealProductRows(dealIds) {
  if (!Array.isArray(dealIds) || dealIds.length === 0) {
    return [];
  }

  const uniqueDealIds = [...new Set(dealIds.filter(Boolean))];

  const goodsList = await batchProcess(
    uniqueDealIds,
    MAX_BATCH_SIZE,
    async (batch) => {
      const cmd = batch.reduce((accumulator, id) => {
        accumulator[id] = `crm.deal.productrows.get?id=${id}`;
        return accumulator;
      }, {});

      const response = await bitrixPost("batch", { halt: 0, cmd });
      const batchRows = [];

      for (const id of batch) {
        const productRows = response?.result?.result?.[id] || [];

        for (const row of productRows) {
          batchRows.push({
            ...row,
            OWNER_ID: id,
          });
        }
      }

      return batchRows;
    },
    BATCH_DELAY_MS
  );

  return goodsList.map(
    ({
      OWNER_TYPE,
      ORIGINAL_PRODUCT_NAME,
      PRICE_EXCLUSIVE,
      PRICE_NETTO,
      PRICE_BRUTTO,
      PRODUCT_DESCRIPTION,
      DISCOUNT_TYPE_ID,
      DISCOUNT_RATE,
      DISCOUNT_SUM,
      TAX_RATE,
      TAX_INCLUDED,
      CUSTOMIZED,
      MEASURE_CODE,
      MEASURE_NAME,
      SORT,
      XML_ID,
      TYPE,
      STORE_ID,
      RESERVE_ID,
      DATE_RESERVE_END,
      RESERVE_QUANTITY,
      ID,
      OWNER_ID,
      ...rest
    }) => ({
      GOOD_ID: ID ?? "",
      ID: OWNER_ID ?? "",
      ...rest,
    })
  );
}
