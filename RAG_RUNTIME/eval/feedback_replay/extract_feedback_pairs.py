#!/usr/bin/env python3
"""P12.1.2: Extract (user_query, assistant_response, feedback) triplets from SQLite dump.

Reads labus_rag.db (from _dbdump/YYYY-MM-DD/) and serializes all rated exchanges
into feedback_pairs.json — the format consumed by replay.py / prepare_r4_pairs.py.

Includes both positive (rating=1) and negative (rating=-1) feedback; prior dumps
only captured negatives.

Usage:
    python extract_feedback_pairs.py --db _dbdump/2026-04-15/labus_rag.db
    python extract_feedback_pairs.py --db <path> --since 2026-04-09
    python extract_feedback_pairs.py --db <path> --output feedback_pairs.json
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def fetch_pairs(db_path: Path, since: str | None) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    since_clause = "AND mf.created_at >= ?" if since else ""
    params: tuple = (since,) if since else ()

    # Assistant messages that have feedback, with their chat metadata.
    rows = conn.execute(f"""
        SELECT
            mf.id                      AS feedback_id,
            mf.rating                  AS rating,
            mf.comment                 AS expert_comment,
            mf.created_at              AS feedback_created_at,
            m_asst.id                  AS assistant_msg_id,
            m_asst.chat_id             AS chat_id,
            m_asst.content             AS assistant_content,
            m_asst.mode                AS assistant_mode,
            m_asst.structured_data     AS assistant_structured_raw,
            m_asst.latency_ms          AS assistant_latency_ms,
            m_asst.created_at          AS assistant_created_at,
            c.title                    AS chat_title
        FROM message_feedback mf
        JOIN messages m_asst ON m_asst.id = mf.message_id
        JOIN chats    c      ON c.id      = m_asst.chat_id
        WHERE m_asst.role = 'assistant'
          {since_clause}
        ORDER BY mf.created_at ASC
    """, params).fetchall()

    pairs: list[dict] = []
    for r in rows:
        # Find the latest user message in the same chat before the assistant message.
        user_row = conn.execute("""
            SELECT id, content
            FROM messages
            WHERE chat_id = ? AND role = 'user' AND created_at <= ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """, (r["chat_id"], r["assistant_created_at"])).fetchone()

        if user_row is None:
            continue

        structured = None
        if r["assistant_structured_raw"]:
            try:
                structured = json.loads(r["assistant_structured_raw"])
            except json.JSONDecodeError:
                structured = None

        pairs.append({
            "feedback_id": r["feedback_id"],
            "chat_id": r["chat_id"],
            "chat_title": r["chat_title"],
            "user_query": user_row["content"],
            "user_msg_id": user_row["id"],
            "assistant_msg_id": r["assistant_msg_id"],
            "assistant_content": r["assistant_content"],
            "assistant_mode": r["assistant_mode"],
            "assistant_latency_ms": r["assistant_latency_ms"],
            "assistant_structured": structured,
            "rating": r["rating"],
            "expert_comment": r["expert_comment"] or "",
            "feedback_created_at": r["feedback_created_at"],
        })

    conn.close()
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to labus_rag.db")
    ap.add_argument("--since", default=None, help="Only pairs with feedback_created_at >= YYYY-MM-DD")
    ap.add_argument("--output", default=str(ROOT / "feedback_pairs.json"))
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found.")
        return 1

    pairs = fetch_pairs(db_path, args.since)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    pos = sum(1 for p in pairs if p["rating"] == 1)
    neg = sum(1 for p in pairs if p["rating"] == -1)
    with_comment = sum(1 for p in pairs if p["expert_comment"].strip())

    print(f"Extracted {len(pairs)} pairs from {db_path}")
    print(f"  positive (rating=+1): {pos}")
    print(f"  negative (rating=-1): {neg}")
    print(f"  with expert comment: {with_comment}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
