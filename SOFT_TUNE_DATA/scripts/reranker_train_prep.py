"""P15.D-prep: extract training pairs for reranker fine-tune from accumulated
v3-v11 judge scores + answers_rag references.

Output: SOFT_TUNE_DATA/reranker_training_pairs.jsonl
  Each line: {"query": str, "passage": str, "label": 0|1, "source": "vN_topicM"}

Labels (weak supervision):
  - winner=rag  → top-5 refs labeled 1 (relevant)
  - winner=no_rag → top-5 refs labeled 0 (less relevant)
  - tie / error → skip

Each query contributes up to 5 (query, doc) pairs.
Total expected: ~20-25k examples.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
OUT = DATA / "reranker_training_pairs.jsonl"

VERSIONS = ["", "_v2", "_v3", "_v4", "_v5", "_v6", "_v7", "_v10", "_v11"]
TOP_REFS_PER_QUERY = 5
MAX_SNIPPET_CHARS = 400


def main():
    out_rows = []
    total_seen = 0
    for ver in VERSIONS:
        scores_path = DATA / f"scores{ver}.jsonl"
        ans_path = DATA / f"answers_rag{ver}.jsonl"
        if not scores_path.exists() or not ans_path.exists():
            print(f"skip v{ver}: files missing")
            continue
        scores = {}
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            s = json.loads(line)
            if "error" in s: continue
            if not s.get("winner"): continue
            if s["winner"] == "tie": continue
            scores[s["id"]] = s["winner"]

        used = 0
        for line in ans_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            a = json.loads(line)
            if a.get("error"): continue
            winner = scores.get(a["id"])
            if winner is None: continue
            fr = a.get("full_response") or {}
            refs = (fr.get("references") or [])[:TOP_REFS_PER_QUERY]
            if not refs: continue
            label = 1 if winner == "rag" else 0
            for r in refs:
                snippet = (r.get("snippet") or "").strip()
                if not snippet or len(snippet) < 20:
                    continue
                out_rows.append({
                    "query": a["query"],
                    "passage": snippet[:MAX_SNIPPET_CHARS],
                    "label": label,
                    "source": f"v{ver or '1'}_{a.get('topic', 'unk')}",
                })
                used += 1
        total_seen += used
        print(f"  v{ver or '1'}: scored={len(scores)}, training pairs={used}")

    # Dedup by (query, passage)
    seen = set()
    deduped = []
    for r in out_rows:
        k = (r["query"][:200], r["passage"][:200])
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    print(f"\nTotal pairs: {total_seen}, deduped: {len(deduped)}")
    print(f"  positive (label=1): {sum(1 for r in deduped if r['label']==1)}")
    print(f"  negative (label=0): {sum(1 for r in deduped if r['label']==0)}")

    with OUT.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ wrote {len(deduped)} → {OUT}")


if __name__ == "__main__":
    main()
