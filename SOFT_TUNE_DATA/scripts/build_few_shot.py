"""P14 Phase F-alt-1: соберёт топ-20 RAG-wins из scores.jsonl + answers_rag.jsonl
и сохранит как few-shot examples для инжекта в prompts.yaml.

Top-20 определяется по: winner=rag AND high (rag_completeness + rag_accuracy + rag_format_quality).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
SCORES = DATA / "scores.jsonl"
RAG = DATA / "answers_rag.jsonl"
OUT = DATA / "few_shot_examples.jsonl"


def main():
    scores = {}
    for line in SCORES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            if "error" in s:
                continue
            sid = s["id"]
            total = (s.get("rag_completeness", 0)
                     + s.get("rag_accuracy", 0)
                     + s.get("rag_format_quality", 0))
            grounded = 1 if s.get("rag_pricing_grounded") else 0
            scores[sid] = {
                "score": total,
                "winner": s.get("winner"),
                "grounded": grounded,
                "topic": s.get("topic"),
                "persona": s.get("persona"),
                "winner_reason": s.get("winner_reason", ""),
            }

    rag_answers = {}
    for line in RAG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            if not a.get("error"):
                rag_answers[a["id"]] = a

    # Pick wins, prefer grounded
    wins = [
        (sid, s, rag_answers.get(sid))
        for sid, s in scores.items()
        if s["winner"] == "rag" and sid in rag_answers
    ]
    # Sort: grounded desc, then total score desc
    wins.sort(key=lambda x: (x[1]["grounded"], x[1]["score"]), reverse=True)

    # Diversify by topic — take at most 3 per topic
    topic_count: dict[str, int] = {}
    picked = []
    for sid, s, ans in wins:
        if len(picked) >= 20:
            break
        t = s["topic"]
        if topic_count.get(t, 0) >= 3:
            continue
        topic_count[t] = topic_count.get(t, 0) + 1
        picked.append((sid, s, ans))

    examples = []
    for sid, s, ans in picked:
        examples.append({
            "id": sid,
            "topic": s["topic"],
            "persona": s["persona"],
            "score_total": s["score"],
            "grounded": bool(s["grounded"]),
            "winner_reason": s["winner_reason"],
            "query": ans["query"],
            "summary": (ans.get("summary") or "")[:1500],
            "deal_items_count": ans.get("deal_items_count", 0),
            "references_count": ans.get("references_count", 0),
            "historical_deals_count": ans.get("historical_deals_count", 0),
            "total_value": ans.get("total_value"),
            "intent": ans.get("intent"),
        })

    with OUT.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"✓ {len(examples)} few-shot examples → {OUT}")
    print("\nBreakdown by topic:")
    for t, n in sorted(topic_count.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
