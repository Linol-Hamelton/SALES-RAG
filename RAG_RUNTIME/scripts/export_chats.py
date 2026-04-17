"""Export chats / messages / feedback / lessons from a labus_rag.db dump
into human-readable JSON + Markdown files next to the DB.

Usage:
  python RAG_RUNTIME/scripts/export_chats.py _dbdump/2026-04-17/labus_rag.db
  python RAG_RUNTIME/scripts/export_chats.py _dbdump/2026-04-17/labus_rag.db --since 2026-04-01

Output layout (sibling to the DB):
  chats/             - one markdown per chat (full conversation + ratings + comments)
  chats.json         - structured dump of all chats (for downstream analysis)
  feedback.json      - all message_feedback rows joined with the message text
  feedback.md        - human review: all -1/+1 entries grouped by rating
  lessons.json       - feedback_lessons (without the embedding BLOB)
  SUMMARY.md         - aggregate stats: totals, ratings distribution, top bad cases
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_structured(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return {"_parse_error": True, "_raw": raw[:500]}


def _format_money(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):,.0f}".replace(",", " ") + " ₽"
    except Exception:
        return str(v)


def _slugify(text: str, maxlen: int = 40) -> str:
    out = []
    for ch in text.strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
        if len("".join(out)) >= maxlen:
            break
    return "".join(out).strip("-") or "chat"


def export(db_path: Path, since: str | None = None) -> dict[str, Any]:
    out_dir = db_path.parent
    chats_dir = out_dir / "chats"
    chats_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    since_clause = ""
    since_args: tuple = ()
    if since:
        since_clause = " WHERE c.created_at >= ?"
        since_args = (since,)

    users = {
        r["id"]: dict(r)
        for r in conn.execute("SELECT id, username, email, full_name, role FROM users").fetchall()
    }

    chats = conn.execute(
        f"""
        SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
               COUNT(m.id) AS msg_count
          FROM chats c
          LEFT JOIN messages m ON m.chat_id = c.id
          {since_clause}
          GROUP BY c.id
          ORDER BY c.created_at DESC
        """,
        since_args,
    ).fetchall()

    chats_json: list[dict] = []
    rating_stats = Counter()
    by_user = Counter()
    bad_cases: list[dict] = []
    good_cases: list[dict] = []

    for c in chats:
        chat = dict(c)
        chat["user"] = users.get(c["user_id"], {})
        msgs = conn.execute(
            """
            SELECT m.id, m.role, m.content, m.mode, m.structured_data,
                   m.latency_ms, m.created_at,
                   f.rating AS feedback_rating,
                   f.comment AS feedback_comment,
                   f.created_at AS feedback_at
              FROM messages m
              LEFT JOIN message_feedback f
                     ON f.message_id = m.id
              WHERE m.chat_id = ?
              ORDER BY m.id
            """,
            (c["id"],),
        ).fetchall()
        chat["messages"] = []
        for m in msgs:
            mrow = dict(m)
            mrow["structured_data"] = _load_structured(m["structured_data"])
            chat["messages"].append(mrow)
            if m["feedback_rating"] is not None:
                rating_stats[m["feedback_rating"]] += 1
                by_user[c["user_id"]] += 1
                case = {
                    "chat_id": c["id"],
                    "chat_title": c["title"],
                    "message_id": m["id"],
                    "user_query": _find_preceding_user_query(chat["messages"]),
                    "assistant": m["content"][:1000],
                    "rating": m["feedback_rating"],
                    "comment": m["feedback_comment"] or "",
                    "created_at": m["created_at"],
                    "feedback_at": m["feedback_at"],
                    "username": chat["user"].get("username"),
                }
                if m["feedback_rating"] == -1:
                    bad_cases.append(case)
                else:
                    good_cases.append(case)
        chats_json.append(chat)
        _write_chat_md(chats_dir, chat)

    # feedback.json — one entry per rating, with full context
    feedback_rows = conn.execute(
        """
        SELECT f.id, f.message_id, f.user_id, f.rating, f.comment,
               f.created_at AS feedback_at,
               m.chat_id, m.content AS assistant, m.created_at AS msg_at,
               m.mode, m.latency_ms
          FROM message_feedback f
          JOIN messages m ON m.id = f.message_id
          ORDER BY f.created_at DESC
        """
    ).fetchall()
    feedback_json = []
    for r in feedback_rows:
        row = dict(r)
        # Pull the user query that triggered this assistant message
        prev_user = conn.execute(
            """
            SELECT content FROM messages
             WHERE chat_id = ? AND id < ? AND role = 'user'
             ORDER BY id DESC LIMIT 1
            """,
            (r["chat_id"], r["message_id"]),
        ).fetchone()
        row["user_query"] = prev_user["content"] if prev_user else ""
        row["username"] = users.get(r["user_id"], {}).get("username", "")
        feedback_json.append(row)

    # lessons.json — drop the BLOB
    lessons = conn.execute(
        """
        SELECT id, feedback_id, user_query, direction, lesson_text, rating,
               is_active, match_count, created_at
          FROM feedback_lessons
          ORDER BY created_at DESC
        """
    ).fetchall()
    lessons_json = [dict(r) for r in lessons]

    rules = conn.execute(
        "SELECT id, rule_text, direction, priority, source_ids, is_active, created_at FROM feedback_rules ORDER BY priority DESC, id"
    ).fetchall()
    rules_json = [dict(r) for r in rules]

    # Write outputs
    (out_dir / "chats.json").write_text(
        json.dumps(chats_json, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "feedback.json").write_text(
        json.dumps(feedback_json, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "lessons.json").write_text(
        json.dumps(lessons_json, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "rules.json").write_text(
        json.dumps(rules_json, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    _write_feedback_md(out_dir / "feedback.md", bad_cases, good_cases)
    _write_summary_md(
        out_dir / "SUMMARY.md",
        db_path=db_path,
        chats=chats_json,
        rating_stats=rating_stats,
        by_user=by_user,
        users=users,
        bad_cases=bad_cases,
        good_cases=good_cases,
        lessons_json=lessons_json,
        rules_json=rules_json,
    )

    conn.close()
    return {
        "chats": len(chats_json),
        "messages": sum(len(c["messages"]) for c in chats_json),
        "feedback": len(feedback_json),
        "lessons": len(lessons_json),
        "rules": len(rules_json),
        "rating_stats": dict(rating_stats),
        "out_dir": str(out_dir),
    }


def _find_preceding_user_query(messages: list[dict]) -> str:
    """For display in feedback dump — what triggered the assistant turn."""
    last_user = ""
    for m in messages:
        if m["role"] == "user":
            last_user = m["content"]
        elif m.get("feedback_rating") is not None:
            return last_user
    return last_user


def _write_chat_md(chats_dir: Path, chat: dict) -> None:
    slug = _slugify(chat["title"] or f"chat-{chat['id']}")
    fname = chats_dir / f"{chat['id']:04d}-{slug}.md"
    user = chat.get("user") or {}
    lines = [
        f"# Чат #{chat['id']}: {chat['title']}",
        "",
        f"- Пользователь: **{user.get('username','?')}** "
        f"({user.get('role','')}, {user.get('full_name','')})",
        f"- Создан: {chat['created_at']}",
        f"- Последнее обновление: {chat['updated_at']}",
        f"- Сообщений: {chat['msg_count']}",
        "",
        "---",
        "",
    ]
    for m in chat["messages"]:
        who = "👤 Пользователь" if m["role"] == "user" else "🤖 Ассистент"
        lines.append(f"## {who} · {m['created_at']}")
        if m["role"] == "assistant" and m["latency_ms"]:
            lines.append(f"_latency: {m['latency_ms']} ms · mode: {m['mode']}_")
        lines.append("")
        lines.append(m["content"])
        lines.append("")
        sd = m.get("structured_data")
        if isinstance(sd, dict) and sd:
            ep = sd.get("estimated_price") or {}
            pb = sd.get("price_band") or {}
            tags = []
            if sd.get("confidence"):
                tags.append(f"confidence={sd['confidence']}")
            if ep.get("value"):
                tags.append(f"est={_format_money(ep['value'])}")
            if pb.get("min") is not None or pb.get("max") is not None:
                tags.append(f"band={_format_money(pb.get('min'))}—{_format_money(pb.get('max'))}")
            if sd.get("flags"):
                tags.append(f"flags={'; '.join(sd['flags'][:3])}")
            if sd.get("risks"):
                tags.append(f"risks={'; '.join(sd['risks'][:3])}")
            if tags:
                lines.append("<details><summary>Structured metadata</summary>")
                lines.append("")
                for t in tags:
                    lines.append(f"- {t}")
                refs = sd.get("references") or []
                if refs:
                    lines.append("")
                    lines.append("References:")
                    for r in refs[:5]:
                        lines.append(
                            f"  - {r.get('doc_type','?')} · {r.get('product_name') or r.get('article_id') or '-'}"
                            f" · score={r.get('score',0)}"
                            + (f" · [Bitrix]({r['bitrix_url']})" if r.get('bitrix_url') else "")
                        )
                lines.append("")
                lines.append("</details>")
                lines.append("")
        if m.get("feedback_rating") is not None:
            icon = "👎" if m["feedback_rating"] == -1 else "👍"
            lines.append(f"> **{icon} Оценка менеджера ({m['feedback_at']}):**")
            lines.append(f"> {m['feedback_comment'] or '(без комментария)'}")
            lines.append("")
        lines.append("---")
        lines.append("")
    fname.write_text("\n".join(lines), encoding="utf-8")


def _write_feedback_md(path: Path, bad: list[dict], good: list[dict]) -> None:
    lines = [
        "# Feedback dump",
        "",
        f"Сгенерировано: {datetime.now().isoformat(timespec='seconds')}",
        f"- 👎 Негативные: **{len(bad)}**",
        f"- 👍 Позитивные: **{len(good)}**",
        "",
        "---",
        "",
        "## 👎 Негативные оценки (все)",
        "",
    ]
    for i, c in enumerate(bad, 1):
        lines.extend(_feedback_block(i, c))
    lines.append("---")
    lines.append("")
    lines.append("## 👍 Позитивные оценки (все)")
    lines.append("")
    for i, c in enumerate(good, 1):
        lines.extend(_feedback_block(i, c))
    path.write_text("\n".join(lines), encoding="utf-8")


def _feedback_block(i: int, c: dict) -> list[str]:
    q = (c.get("user_query") or "").replace("\n", " ")[:500]
    a = (c.get("assistant") or "").replace("\n", " ")[:800]
    return [
        f"### {i}. Чат #{c['chat_id']} · msg #{c['message_id']} · {c.get('username')}",
        f"- Дата оценки: {c.get('feedback_at')}",
        f"- Заголовок чата: **{c.get('chat_title')}**",
        f"- Вопрос: _{q}_",
        f"- Ответ (до 800 симв.): {a}",
        f"- Комментарий менеджера: **{c.get('comment') or '(пусто)'}**",
        "",
    ]


def _write_summary_md(
    path: Path,
    *,
    db_path: Path,
    chats: list[dict],
    rating_stats: Counter,
    by_user: Counter,
    users: dict,
    bad_cases: list[dict],
    good_cases: list[dict],
    lessons_json: list[dict],
    rules_json: list[dict],
) -> None:
    total_msg = sum(len(c["messages"]) for c in chats)
    pos = rating_stats.get(1, 0)
    neg = rating_stats.get(-1, 0)
    total_rated = pos + neg
    csat = (pos / total_rated * 100) if total_rated else 0.0

    lines = [
        f"# SUMMARY — {db_path.name}",
        "",
        f"Сгенерировано: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Объёмы",
        f"- Чатов: **{len(chats)}**",
        f"- Сообщений: **{total_msg}**",
        f"- Оценок (📨 feedback): **{total_rated}** (👍 {pos} / 👎 {neg})",
        f"- CSAT: **{csat:.1f}%**",
        f"- Lessons: **{len(lessons_json)}**",
        f"- Rules: **{len(rules_json)}**",
        "",
        "## Оценки по пользователям",
    ]
    for uid, cnt in by_user.most_common():
        u = users.get(uid, {})
        lines.append(f"- {u.get('username','?')} ({u.get('role','')}): {cnt}")

    lines += ["", "## Топ-10 самых свежих 👎 негативных кейсов", ""]
    for c in sorted(bad_cases, key=lambda x: x.get("feedback_at") or "", reverse=True)[:10]:
        q = (c.get("user_query") or "").replace("\n", " ")[:180]
        cm = (c.get("comment") or "").replace("\n", " ")[:180]
        lines.append(f"- #{c['chat_id']}/msg{c['message_id']} · _{q}_ — комментарий: **{cm or '(пусто)'}**")

    lines += ["", "## Топ-10 самых свежих 👍 позитивных кейсов", ""]
    for c in sorted(good_cases, key=lambda x: x.get("feedback_at") or "", reverse=True)[:10]:
        q = (c.get("user_query") or "").replace("\n", " ")[:180]
        cm = (c.get("comment") or "").replace("\n", " ")[:180]
        lines.append(f"- #{c['chat_id']}/msg{c['message_id']} · _{q}_ — {cm or '(пусто)'}")

    # Cluster by direction for actionable analysis
    dir_counts = Counter()
    for l in lessons_json:
        if l.get("direction"):
            dir_counts[l["direction"]] += 1
    if dir_counts:
        lines += ["", "## Lessons по направлениям", ""]
        for d, n in dir_counts.most_common():
            lines.append(f"- {d}: {n}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path, help="Path to labus_rag.db (a snapshot)")
    ap.add_argument("--since", help="ISO date (YYYY-MM-DD) to filter chats by created_at")
    args = ap.parse_args()
    stats = export(args.db, since=args.since)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
