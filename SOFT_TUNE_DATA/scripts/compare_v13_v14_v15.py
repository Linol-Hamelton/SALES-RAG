"""P17.alt analysis: compare v13 (chat baseline), v14 (all-reasoner regression),
v15 (hybrid intent routing).

Run after `RUN_SUFFIX=_v15 JUDGE_PROMPT_VERSION=v2 python judge_and_recommend.py`
completes.

Output: comprehensive verdict on whether P17.alt (intent-based hybrid) outperforms
both pure-chat (v13) and pure-reasoner (v14).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"


def load(suf: str) -> dict[str, dict]:
    p = DATA / f"scores{suf}.jsonl"
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


def avg(scores: dict[str, dict], ka: str, kb: str) -> float:
    vals = [
        s.get(ka, 0) - s.get(kb, 0)
        for s in scores.values()
        if isinstance(s.get(ka), (int, float))
        and isinstance(s.get(kb), (int, float))
    ]
    return sum(vals) / len(vals) if vals else 0.0


def grounded(scores: dict[str, dict], key: str) -> float:
    n = len(scores)
    if not n:
        return 0.0
    return sum(1 for s in scores.values() if s.get(key) is True) / n * 100


def main():
    v13 = load("_v13")
    v14 = load("_v14")
    v15 = load("_v15")

    if not v15:
        print(f"ERROR: scores_v15.jsonl not found. Run judge first.")
        sys.exit(1)

    print(f"Loaded: v13={len(v13)}, v14={len(v14)}, v15={len(v15)}")

    print()
    print("=" * 60)
    print("AGGREGATE RAG win (all under v2 judge — apples-to-apples)")
    print("=" * 60)
    for label, sc in [("v13 (chat)", v13), ("v14 (reasoner all)", v14), ("v15 (P17.alt hybrid)", v15)]:
        n = len(sc)
        if not n:
            continue
        w = Counter(s["winner"] for s in sc.values())
        rag = w.get("rag", 0) / n * 100
        no = w.get("no_rag", 0) / n * 100
        tie = w.get("tie", 0) / n * 100
        print(f"  {label:<22}  rag={rag:>5.1f}%  no={no:>5.1f}%  tie={tie:>5.1f}%  (n={n})")

    # Per-persona
    print()
    print("=" * 60)
    print("Per-persona RAG win")
    print("=" * 60)
    for label, sc in [("v13", v13), ("v14", v14), ("v15", v15)]:
        if not sc:
            continue
        by_p = defaultdict(Counter)
        for s in sc.values():
            by_p[s.get("persona", "?")][s["winner"]] += 1
        line = f"  {label:<5}  "
        for p in ["client", "manager"]:
            c = by_p.get(p, Counter())
            n = sum(c.values())
            if n:
                line += f"{p}={c.get('rag', 0) / n * 100:>5.1f}%  "
        print(line)

    # Dimension deltas
    print()
    print("=" * 60)
    print("Avg dimension deltas (RAG − no_RAG)")
    print("=" * 60)
    print(f"  {'ver':<5}  {'Δacc':>6}  {'Δcomp':>6}  {'Δfmt':>6}  {'gnd_rag':>8}  {'gnd_no':>7}")
    for label, sc in [("v13", v13), ("v14", v14), ("v15", v15)]:
        if not sc:
            continue
        dacc = avg(sc, "rag_accuracy", "no_rag_accuracy")
        dcomp = avg(sc, "rag_completeness", "no_rag_completeness")
        dfmt = avg(sc, "rag_format_quality", "no_rag_format_quality")
        gr = grounded(sc, "rag_pricing_grounded")
        gn = grounded(sc, "no_rag_pricing_grounded")
        print(f"  {label:<5}  {dacc:>+6.2f}  {dcomp:>+6.2f}  {dfmt:>+6.2f}  {gr:>7.1f}%  {gn:>6.1f}%")

    # Per-topic v15 vs baseline
    print()
    print("=" * 60)
    print("Per-topic Δpp (v15 vs v13 baseline) — главное измерение P17.alt")
    print("=" * 60)
    t13 = defaultdict(Counter)
    t15 = defaultdict(Counter)
    for s in v13.values():
        t13[s.get("topic", "?")][s["winner"]] += 1
    for s in v15.values():
        t15[s.get("topic", "?")][s["winner"]] += 1

    topics = sorted(set(t13.keys()) | set(t15.keys()))
    deltas = []
    for t in topics:
        c13, c15 = t13.get(t, Counter()), t15.get(t, Counter())
        n13, n15 = sum(c13.values()), sum(c15.values())
        if not n13 or not n15:
            continue
        p13 = c13.get("rag", 0) / n13 * 100
        p15 = c15.get("rag", 0) / n15 * 100
        deltas.append((t, p13, p15, p15 - p13))
    deltas.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'topic':<24}  {'v13%':>6}  {'v15%':>6}  {'Δpp':>6}")
    for t, p13, p15, d in deltas:
        marker = " ✓" if d >= 2 else (" ✗" if d <= -2 else "")
        print(f"  {t:<24}  {p13:>5.1f}%  {p15:>5.1f}%  {d:>+5.1f}{marker}")

    # P17 hypothesis check: did reasoner-routed intents improve while chat-routed stayed?
    print()
    print("=" * 60)
    print("P17.alt hypothesis: reasoner-routed intents should WIN, chat-routed stay")
    print("=" * 60)
    # Intent-level (using question intent metadata if available)
    # For simplicity, group topics by "reasoner-routed" vs "chat-routed"
    REASONER_TOPICS = {"objection_handling", "objections", "historical_links",
                       "smeta_assist", "pricing_justify", "discovery"}
    print("Reasoner-routed topics (objection/historical/smeta/discovery):")
    for t, p13, p15, d in deltas:
        if t in REASONER_TOPICS:
            print(f"  {t:<24}  v13={p13:>5.1f}% → v15={p15:>5.1f}%  {d:>+5.1f}pp")
    print("\nChat-routed topics (simple/direct):")
    for t, p13, p15, d in deltas:
        if t not in REASONER_TOPICS:
            print(f"  {t:<24}  v13={p13:>5.1f}% → v15={p15:>5.1f}%  {d:>+5.1f}pp")

    # Verdict
    print()
    print("=" * 60)
    print("VERDICT P17.alt")
    print("=" * 60)
    n13 = len(v13)
    n15 = len(v15)
    rag13 = sum(1 for s in v13.values() if s["winner"] == "rag") / max(1, n13) * 100
    rag15 = sum(1 for s in v15.values() if s["winner"] == "rag") / max(1, n15) * 100
    delta_15_13 = rag15 - rag13
    if v14:
        n14 = len(v14)
        rag14 = sum(1 for s in v14.values() if s["winner"] == "rag") / max(1, n14) * 100
        delta_15_14 = rag15 - rag14
    else:
        rag14 = None
        delta_15_14 = None

    print(f"  v13 (chat baseline):     {rag13:.1f}%")
    if rag14 is not None:
        print(f"  v14 (all-reasoner):      {rag14:.1f}%   ({delta_15_14:+.1f}pp gap closed by P17.alt)")
    print(f"  v15 (P17.alt hybrid):    {rag15:.1f}%   ({delta_15_13:+.1f}pp vs v13)")

    if delta_15_13 >= 2.0:
        print(f"\n  🟢 SUCCESS: P17.alt дал +{delta_15_13:.1f}pp над chat baseline.")
        print("  → Переходим к P18 (embedding swap, target +3pp)")
    elif delta_15_13 >= 0.5:
        print(f"\n  🟡 MARGINAL: +{delta_15_13:.1f}pp над v13. Hybrid дал умеренную выгоду.")
        print("  → P18 имеет смысл; рассмотреть refine intent boundaries.")
    elif delta_15_13 >= -1.0:
        print(f"\n  🟠 FLAT: {delta_15_13:+.1f}pp над v13. Hybrid не дал чистой выгоды.")
        print("  → P18 как next phase; revisit intent groups в P21.")
    else:
        print(f"\n  🔴 REGRESSION: {delta_15_13:+.1f}pp. Hybrid hurts.")
        print("  → Rollback на pure chat (v13). Пропустить P17 целиком.")


if __name__ == "__main__":
    main()
