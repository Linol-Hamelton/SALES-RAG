#!/usr/bin/env bash
# P13.3: Clean full reindex on the CURRENT corpus (no Bitrix pull).
#
# Runs analytics → all ingest scripts → full build_index re-embed → verify.
# `set -e` halts on first failure. Each phase logged separately.
#
# Usage:
#   ./scripts/full_reindex_clean.sh
#
# Output: D:/tmp/reindex_<timestamp>/{phase1,phase2.*,phase3,phase4}.log

set -euo pipefail

ROOT="D:/SALES_RAG"
PYTHON="$ROOT/RAG_RUNTIME/.venv/Scripts/python.exe"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="D:/tmp/reindex_${TS}"
mkdir -p "$LOG_DIR"

export PYTHONIOENCODING=utf-8

cd "$ROOT"

echo "=========================================="
echo "P13.3 clean full reindex"
echo "Timestamp: $TS"
echo "Logs:      $LOG_DIR"
echo "Python:    $PYTHON"
echo "=========================================="

step_start() {
    echo ""
    echo ">>> [$(date +%H:%M:%S)] $1"
}

run_step() {
    local label="$1"
    local logfile="$2"
    shift 2
    step_start "$label"
    if "$@" > "$logfile" 2>&1; then
        local lines=$(wc -l < "$logfile")
        local tail=$(tail -1 "$logfile" 2>/dev/null | head -c 200)
        echo "    OK ($lines lines logged)  tail: $tail"
    else
        local code=$?
        echo "    FAIL exit=$code"
        echo "    --- last 30 lines of log ---"
        tail -30 "$logfile"
        echo "    --- end log ---"
        exit $code
    fi
}

# ---- Phase 1: Analytics (regenerate derived files from current CSVs) ----
run_step "Phase 1: analytics (RAG_ANALYTICS/runAll.mjs)" \
    "$LOG_DIR/phase1_analytics.log" \
    node RAG_ANALYTICS/runAll.mjs

# ---- Phase 2: All ingest scripts in dependency order ----
# Order matches refreshSalesRagData.mjs (sans generateRagData):
#   roadmaps → ingest (reads roadmap_docs.jsonl) → knowledge → offer_compositions
#   → service_pages (T1, new) → orders (T7, new) → bridges
run_step "Phase 2.1: ingest_roadmaps" \
    "$LOG_DIR/phase2_1_roadmaps.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest_roadmaps.py

run_step "Phase 2.2: ingest core (product/bundle/deal_profile/offer_profile/...)" \
    "$LOG_DIR/phase2_2_core.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest.py

run_step "Phase 2.3: ingest_knowledge (FAQ + structured MD)" \
    "$LOG_DIR/phase2_3_knowledge.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest_knowledge.py

run_step "Phase 2.4: ingest_offer_compositions" \
    "$LOG_DIR/phase2_4_offer_compositions.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest_offer_compositions.py

run_step "Phase 2.5: ingest_service_pages (T1)" \
    "$LOG_DIR/phase2_5_service_pages.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest_service_pages.py

run_step "Phase 2.6: ingest_orders (T7)" \
    "$LOG_DIR/phase2_6_orders.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest_orders.py

run_step "Phase 2.7: ingest_bridges" \
    "$LOG_DIR/phase2_7_bridges.log" \
    "$PYTHON" RAG_RUNTIME/scripts/ingest_bridges.py

# ---- Phase 3: build_index full re-embed + upsert ----
run_step "Phase 3: build_index --recreate --batch-size 8 (full re-embed)" \
    "$LOG_DIR/phase3_build_index.log" \
    "$PYTHON" RAG_RUNTIME/scripts/build_index.py --recreate --batch-size 8

# ---- Phase 4: Verify Qdrant counts vs jsonl line counts ----
step_start "Phase 4: verify Qdrant counts vs jsonl"
{
    echo "Verification at $(date)"
    echo ""
    declare -A EXPECTED
    EXPECTED[product]="product_docs.jsonl"
    EXPECTED[bundle]="bundle_docs.jsonl"
    EXPECTED[deal_profile]="deal_profile_docs.jsonl"
    EXPECTED[offer_profile]="offer_profile_docs.jsonl"
    EXPECTED[offer_composition]="offer_composition_docs.jsonl"
    # knowledge bucket aggregates knowledge_docs.jsonl + roi_anchor_docs.jsonl
    EXPECTED[knowledge]="knowledge_docs.jsonl roi_anchor_docs.jsonl"
    EXPECTED[faq]="faq_docs.jsonl"
    EXPECTED[roadmap]="roadmap_docs.jsonl"
    EXPECTED[service_pricing_bridge]="bridge_docs.jsonl"
    EXPECTED[pricing_policy]="pricing_policy_docs.jsonl"
    EXPECTED[service_composition]="service_composition_docs.jsonl"
    EXPECTED[retrieval_support]="retrieval_support_docs.jsonl"
    EXPECTED[photo_analysis]="photo_analysis_docs.jsonl"
    EXPECTED[timeline_fact]="timeline_docs.jsonl"
    # NB: roi_anchor_docs.jsonl ingested as doc_type="knowledge" (rolled into
    # knowledge count); no separate roi_anchor key in Qdrant.
    EXPECTED[service_page]="service_page_docs.jsonl"
    EXPECTED[historical_deal]="historical_deal_docs.jsonl"

    total_jsonl=0
    fails=0
    printf "%-26s | %10s | %10s | %s\n" "doc_type" "jsonl" "qdrant" "status"
    printf "%-26s-+-%10s-+-%10s-+-%s\n" "--------------------------" "----------" "----------" "------"
    for dt in "${!EXPECTED[@]}"; do
        # EXPECTED[$dt] may list multiple jsonl files (space-separated) — sum them.
        jcount=0
        for fname in ${EXPECTED[$dt]}; do
            f="RAG_RUNTIME/data/$fname"
            if [[ -f "$f" ]]; then
                jcount=$((jcount + $(wc -l < "$f" | tr -d ' ')))
            fi
        done
        total_jsonl=$((total_jsonl + jcount))
        qcount=$(curl -s -X POST http://localhost:6333/collections/labus_docs/points/count \
            -H "Content-Type: application/json" \
            -d "{\"filter\":{\"must\":[{\"key\":\"doc_type\",\"match\":{\"value\":\"$dt\"}}]},\"exact\":true}" \
            | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('count','?'))" 2>/dev/null || echo "?")
        if [[ "$jcount" == "$qcount" ]]; then
            status="OK"
        elif [[ "$qcount" == "?" ]] || [[ "$qcount" == "0" && "$jcount" != "0" ]]; then
            status="FAIL"
            fails=$((fails+1))
        else
            status="DRIFT"
            fails=$((fails+1))
        fi
        printf "%-26s | %10s | %10s | %s\n" "$dt" "$jcount" "$qcount" "$status"
    done
    echo ""
    qtotal=$(curl -s http://localhost:6333/collections/labus_docs | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('points_count','?'))" 2>/dev/null || echo "?")
    echo "TOTAL jsonl rows:  $total_jsonl"
    echo "TOTAL Qdrant pts:  $qtotal"
    echo "Mismatched types:  $fails"
} | tee "$LOG_DIR/phase4_verify.log"

echo ""
echo "=========================================="
echo "Reindex completed at $(date)"
echo "Logs: $LOG_DIR"
echo "=========================================="
