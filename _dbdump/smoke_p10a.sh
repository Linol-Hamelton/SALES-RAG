#!/usr/bin/env bash
# P10-A smoke tests (бежит на сервере, где bash и UTF-8 работают нормально)
set -u
API="https://ai.labus.pro/query_structured"
HDR='-H "Content-Type: application/json"'

run() {
  local name="$1"; local body="$2"
  echo
  echo "=============================================================="
  echo "Smoke: $name"
  echo "--------------------------------------------------------------"
  curl -sS -X POST "$API" -H "Content-Type: application/json" -d "$body" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print("confidence:", d.get("confidence")); print("price_band:", d.get("price_band")); print("estimated_price:", d.get("estimated_price")); print("flags:", d.get("flags")); print("summary:", (d.get("summary") or "")[:800])'
}

run "1 Logo" '{"query":"Сколько стоит логотип?"}'
run "2 Световые буквы БЕЗ размера" '{"query":"Световые буквы для вывески"}'
run "3 Брендбук" '{"query":"Сколько стоит брендбук?"}'
run "4 Световые буквы 45 см 7 штук" '{"query":"Световые буквы 45 см 7 штук для кафе"}'
run "5 Фирменный стиль" '{"query":"Сколько стоит фирменный стиль?"}'

echo
echo "=============================================================="
echo "Tail trace logs (последние 5 минут):"
echo "--------------------------------------------------------------"
docker logs --since 5m labus_api 2>&1 | grep -E 'request_trace|feedback_context_built|bridge force-injected|SmetaEngine bridge-category|Intent classified' | tail -80
