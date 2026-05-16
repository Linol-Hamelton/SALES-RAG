"""P16.B.2-analysis: compare v12 (v1 judge) vs v12_recalibrated (v2 judge).

Run AFTER `RUN_SUFFIX=_v12 OUT_SUFFIX=_v12_recalibrated JUDGE_PROMPT_VERSION=v2
python judge_and_recommend.py` completes.

Output: stdout report with:
  - Aggregate win rates before/after
  - pricing_grounded TRUE rate change (the key recalibration metric)
  - Per-topic delta
  - Number of pairs that flipped winner (rag ↔ no_rag)
  - Hand-pickable sample of flipped pairs for manual sanity check
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
V1 = DATA / "scores_v12.jsonl"
V2 = DATA / "scores_v12_recalibrated.jsonl"


def load(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("error") or not d.get("winner"):
            continue
        out[d["id"]] = d
    return out


def main():
    if not V2.exists():
        print(f"ERROR: {V2} not found. Run recalibrated judge first.")
        sys.exit(1)

    s_v1 = load(V1)
    s_v2 = load(V2)
    common = sorted(set(s_v1) & set(s_v2))
    print(f"Loaded v1: {len(s_v1)}, v2: {len(s_v2)}, common: {len(common)}")

    # Aggregate winners
    def wr(scores):
        w = Counter(s["winner"] for s in scores.values())
        n = len(scores)
        return n, w.get("rag", 0)/n*100, w.get("no_rag", 0)/n*100, w.get("tie", 0)/n*100

    n1, rag1, no1, tie1 = wr(s_v1)
    n2, rag2, no2, tie2 = wr(s_v2)
    print()
    print("=== Aggregate win rates ===")
    print(f"  v12 (v1 judge):           rag={rag1:.1f}%  no_rag={no1:.1f}%  tie={tie1:.1f}%  (n={n1})")
    print(f"  v12 (v2 recalibrated):    rag={rag2:.1f}%  no_rag={no2:.1f}%  tie={tie2:.1f}%  (n={n2})")
    print(f"  Δ RAG win:                {rag2-rag1:+.1f}pp")

    # pricing_grounded
    def grounded(scores, key):
        n = len(scores)
        if not n:
            return 0
        return sum(1 for s in scores.values() if s.get(key) is True) / n * 100

    pg_rag_v1 = grounded(s_v1, "rag_pricing_grounded")
    pg_rag_v2 = grounded(s_v2, "rag_pricing_grounded")
    pg_no_v1 = grounded(s_v1, "no_rag_pricing_grounded")
    pg_no_v2 = grounded(s_v2, "no_rag_pricing_grounded")
    print()
    print("=== pricing_grounded TRUE rate ===")
    print(f"  RAG:    v1={pg_rag_v1:.1f}% → v2={pg_rag_v2:.1f}%  Δ{pg_rag_v2-pg_rag_v1:+.1f}pp")
    print(f"  no_RAG: v1={pg_no_v1:.1f}% → v2={pg_no_v2:.1f}%  Δ{pg_no_v2-pg_no_v1:+.1f}pp")

    # Per-topic delta
    print()
    print("=== Per-topic Δ RAG win (top 8 most-affected) ===")
    topic_v1 = defaultdict(Counter)
    topic_v2 = defaultdict(Counter)
    for id_ in common:
        t1 = s_v1[id_].get("topic", "?")
        t2 = s_v2[id_].get("topic", "?")
        topic_v1[t1][s_v1[id_]["winner"]] += 1
        topic_v2[t2][s_v2[id_]["winner"]] += 1
    deltas = []
    for t in sorted(set(topic_v1) | set(topic_v2)):
        c1, c2 = topic_v1[t], topic_v2[t]
        n_t1, n_t2 = sum(c1.values()), sum(c2.values())
        p1 = c1.get("rag", 0)/max(1,n_t1)*100
        p2 = c2.get("rag", 0)/max(1,n_t2)*100
        deltas.append((t, p1, p2, p2-p1, n_t1))
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"  {'topic':<22}  v1%   v2%   Δpp")
    for t, p1, p2, d, n in deltas[:8]:
        print(f"  {t:<22}  {p1:>5.1f} {p2:>5.1f}  {d:>+6.1f}  (n={n})")

    # Winner flips
    print()
    print("=== Winner flips (v1 → v2) ===")
    flips = []
    for id_ in common:
        w1 = s_v1[id_]["winner"]
        w2 = s_v2[id_]["winner"]
        if w1 != w2:
            flips.append((id_, w1, w2, s_v1[id_].get("topic", "?"),
                         s_v1[id_].get("rag_pricing_grounded"),
                         s_v2[id_].get("rag_pricing_grounded")))
    flip_counter = Counter((w1, w2) for _, w1, w2, _, _, _ in flips)
    print(f"  Total flips: {len(flips)} ({len(flips)/len(common)*100:.1f}%)")
    for (w1, w2), n in flip_counter.most_common():
        print(f"    {w1:>6} → {w2:<6}  n={n}")

    # Sample flipped pairs for manual audit (5 from biggest category)
    if flips:
        print()
        print("=== Sample flips (no_rag→rag, key Phase B mechanism) ===")
        for id_, w1, w2, topic, pg1, pg2 in flips[:8]:
            if w1 == "no_rag" and w2 == "rag":
                print(f"  {id_} ({topic}): {w1}→{w2}  pg_rag: {pg1}→{pg2}")

    print()
    print("=== Verdict ===")
    if rag2 - rag1 >= 10:
        print(f"  STRONG signal: +{rag2-rag1:.1f}pp confirms recalibration is the lever.")
    elif rag2 - rag1 >= 5:
        print(f"  POSITIVE: +{rag2-rag1:.1f}pp — recalibration helps but RAG fixes (Phase A) also needed for 95%+.")
    elif rag2 - rag1 >= 0:
        print(f"  WEAK: only +{rag2-rag1:.1f}pp — recalibration not the dominant factor.")
    else:
        print(f"  REGRESSION: {rag2-rag1:+.1f}pp — recalibrated judge is stricter (unexpected).")


if __name__ == "__main__":
    main()
