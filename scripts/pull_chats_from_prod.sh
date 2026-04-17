#!/usr/bin/env bash
# Pull the SQLite chat/feedback DB from prod (labus_api container) and
# materialise it into this repo under _dbdump/<YYYY-MM-DD>/.
#
# Usage:  bash scripts/pull_chats_from_prod.sh
# Env overrides:
#   PROD_HOST (default: root@62.217.178.117)
#   API_CONTAINER (default: labus_api)
#   CONTAINER_DB (default: /app/data/labus_rag.db)
#
# После пулла запусти экспорт:
#   python RAG_RUNTIME/scripts/export_chats.py _dbdump/<date>/labus_rag.db
set -euo pipefail

PROD_HOST="${PROD_HOST:-root@62.217.178.117}"
API_CONTAINER="${API_CONTAINER:-labus_api}"
CONTAINER_DB="${CONTAINER_DB:-/app/data/labus_rag.db}"

TODAY="$(date +%F)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/_dbdump/${TODAY}"
mkdir -p "${OUT_DIR}"

REMOTE_TMP="/tmp/labus_rag_${TODAY}.db"
REMOTE_DUMP="/tmp/labus_rag_${TODAY}.sql"

echo "[1/4] Snapshotting DB inside container ${API_CONTAINER}…"
ssh "${PROD_HOST}" "docker exec ${API_CONTAINER} sqlite3 ${CONTAINER_DB} \".backup ${CONTAINER_DB}.pull\" && docker cp ${API_CONTAINER}:${CONTAINER_DB}.pull ${REMOTE_TMP} && docker exec ${API_CONTAINER} rm -f ${CONTAINER_DB}.pull"

echo "[2/4] Dumping SQL text on remote…"
ssh "${PROD_HOST}" "sqlite3 ${REMOTE_TMP} .dump > ${REMOTE_DUMP}"

echo "[3/4] Copying to ${OUT_DIR}/…"
scp "${PROD_HOST}:${REMOTE_TMP}" "${OUT_DIR}/labus_rag.db"
scp "${PROD_HOST}:${REMOTE_DUMP}" "${OUT_DIR}/labus_rag.sql"

echo "[4/4] Cleaning remote tmp…"
ssh "${PROD_HOST}" "rm -f ${REMOTE_TMP} ${REMOTE_DUMP}"

echo
echo "OK. Pulled to ${OUT_DIR}/"
ls -la "${OUT_DIR}/"
echo
echo "Next:"
echo "  python RAG_RUNTIME/scripts/export_chats.py ${OUT_DIR}/labus_rag.db"
