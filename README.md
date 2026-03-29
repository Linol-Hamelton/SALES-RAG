# SALES_RAG

Минимальный self-contained пакет для переноса на другую машину.

## Что включено
- `RAG_DATA/goods.csv`
- `RAG_DATA/offers.csv`
- `RAG_DATA/orders.csv`
- `RAG_DATA/csvUtils.mjs`
- `RAG_DATA/bitrixClient.mjs`
- `RAG_DATA/getDealsByStage.mjs`
- `RAG_DATA/getDealProductRows.mjs`
- `RAG_DATA/getFullCatalog.mjs`
- `RAG_DATA/generateRagData.mjs`
- `RAG_DATA/repairOrdersDirection.mjs`
- `Bitrix24Webhook.mjs`
- `CurMonDat.mjs`
- `batchUtils.mjs`
- `refreshSalesRagData.mjs`
- весь `RAG_ANALYTICS` с `lib`, build-скриптами и `output`
- `package.json`
- `bundle_manifest.json`
- `verifySalesRag.mjs`
- `.env.example`

## Что специально не включено
- старые модули проекта, не связанные с RAG
- прочие файлы исходного проекта, не участвующие в цепочке Bitrix24 -> raw CSV -> analytics outputs

Этот пакет предназначен для переноса данных, обновления raw CSV из Bitrix24 и запуска аналитики на другой машине для локального RAG / ML / pricing pipeline.

## Требования
- Node.js 18+ 
- доступ к Bitrix24 webhook URL

## Быстрый старт
Из корня `SALES_RAG`:

```powershell
node verifySalesRag.mjs
node RAG_ANALYTICS/runAll.mjs
```

или

```powershell
npm run verify
npm run all
```

## Обновление raw данных из Bitrix24
Если нужен полный refresh из той же папки:

```powershell
node refreshSalesRagData.mjs
```

или

```powershell
npm run refresh
```

Этот сценарий последовательно делает:
1. `RAG_DATA/generateRagData.mjs`
2. `RAG_DATA/repairOrdersDirection.mjs`
3. `RAG_ANALYTICS/runAll.mjs`
4. `verifySalesRag.mjs`

## Конфигурация Bitrix24
По умолчанию используется встроенный webhook URL из пакета.
При необходимости можно переопределить переменные окружения:

```powershell
$env:BITRIX24_WEBHOOK_URL='https://labus.bitrix24.ru/rest/5/a184a56co9ghrehs/'
$env:BITRIX24_OFFERS_STAGE_ID='UC_Y55PKR'
$env:BITRIX24_ORDERS_STAGE_ID='WON'
node refreshSalesRagData.mjs
```

Шаблон переменных лежит в `.env.example`.

## Что проверяет verify
- наличие критичных raw/analytics файлов
- наличие всей Bitrix24 refresh chain
- row counts raw CSV
- наличие `Направление` в `orders.csv`
- минимальные runtime требования для запуска

## Примечание
Структура `RAG_DATA` и `RAG_ANALYTICS` внутри `SALES_RAG` сохранена такой же, как в исходном проекте. Благодаря этому существующие relative imports продолжают работать без переписывания основной логики.
