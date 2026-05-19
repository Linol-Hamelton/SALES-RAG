"""P21.C final analysis: compare v13 (baseline), v13b (noise check), v18 (P21).

Output: definitive P21 verdict — did embedding fine-tune + subcategory filter
beat the noise band measured in v13b?

Run after v18 judge completes.
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


def agg(d):
    n = len(d)
    w = Counter(s["winner"] for s in d.values())
    return n, w.get("rag", 0) / max(1, n) * 100, w.get("no_rag", 0) / max(1, n) * 100, w.get("tie", 0) / max(1, n) * 100


def avg(d, ka, kb):
    vals = [s.get(ka, 0) - s.get(kb, 0) for s in d.values()
            if isinstance(s.get(ka), (int, float)) and isinstance(s.get(kb), (int, float))]
    return sum(vals) / len(vals) if vals else 0


def grounded(d, key):
    n = len(d)
    if not n:
        return 0
    return sum(1 for s in d.values() if s.get(key) is True) / n * 100


def main():
    v13 = load("_v13")
    v13b = load("_v13b")
    v18 = load("_v18")

    if not v18:
        print(f"ERROR: scores_v18.jsonl not found. Run judge first.")
        sys.exit(1)

    print(f"Loaded: v13={len(v13)}, v13b={len(v13b)}, v18={len(v18)}")

    # Compute noise band from v13 vs v13b
    n13, rag13, no13, tie13 = agg(v13)
    n13b, rag13b, no13b, tie13b = agg(v13b)
    n18, rag18, no18, tie18 = agg(v18)

    noise_band = abs(rag13b - rag13)
    baseline_avg = (rag13 + rag13b) / 2  # average of two runs = best baseline estimate

    print()
    print("=" * 64)
    print("AGGREGATE — apples-to-apples под v2 judge на same v13 questions")
    print("=" * 64)
    print(f"  v13  (chat baseline run 1):  rag={rag13:.1f}%  no={no13:.1f}%  tie={tie13:.1f}%")
    print(f"  v13b (chat baseline run 2):  rag={rag13b:.1f}%  no={no13b:.1f}%  tie={tie13b:.1f}%")
    print(f"  ── Noise band: ±{noise_band:.2f}pp")
    print(f"  ── Baseline avg: {baseline_avg:.2f}%")
    print()
    print(f"  v18  (P21.A+B finetune+subcat): rag={rag18:.1f}%  no={no18:.1f}%  tie={tie18:.1f}%")
    print(f"  Δ vs v13:  {rag18 - rag13:+.2f}pp")
    print(f"  Δ vs v13b: {rag18 - rag13b:+.2f}pp")
    print(f"  Δ vs avg:  {rag18 - baseline_avg:+.2f}pp")

    # Per-persona
    print()
    print("=" * 64)
    print("Per-persona RAG win %")
    print("=" * 64)
    for label, d in [("v13", v13), ("v13b", v13b), ("v18", v18)]:
        if not d:
            continue
        by_p = defaultdict(Counter)
        for s in d.values():
            by_p[s.get("persona", "?")][s["winner"]] += 1
        line = f"  {label:<5}"
        for p in ["client", "manager"]:
            c = by_p.get(p, Counter())
            n = sum(c.values())
            if n:
                line += f"  {p}={c.get('rag', 0) / n * 100:>5.1f}%"
        print(line)

    # Dimension deltas
    print()
    print("=" * 64)
    print("Avg dimension deltas (RAG − no_RAG)")
    print("=" * 64)
    print(f"  {'ver':<5}  {'Δacc':>6}  {'Δcomp':>6}  {'Δfmt':>6}  {'gnd_rag':>8}  {'gnd_no':>7}")
    for label, d in [("v13", v13), ("v13b", v13b), ("v18", v18)]:
        if not d:
            continue
        dacc = avg(d, "rag_accuracy", "no_rag_accuracy")
        dcomp = avg(d, "rag_completeness", "no_rag_completeness")
        dfmt = avg(d, "rag_format_quality", "no_rag_format_quality")
        gr = grounded(d, "rag_pricing_grounded")
        gn = grounded(d, "no_rag_pricing_grounded")
        print(f"  {label:<5}  {dacc:>+6.2f}  {dcomp:>+6.2f}  {dfmt:>+6.2f}  {gr:>7.1f}%  {gn:>6.1f}%")

    # Per-topic v18 vs avg(v13, v13b)
    print()
    print("=" * 64)
    print("Per-topic Δpp (v18 vs avg(v13,v13b)) — sorted")
    print("=" * 64)
    t13 = defaultdict(Counter)
    t13b = defaultdict(Counter)
    t18 = defaultdict(Counter)
    for s in v13.values():
        t13[s.get("topic", "?")][s["winner"]] += 1
    for s in v13b.values():
        t13b[s.get("topic", "?")][s["winner"]] += 1
    for s in v18.values():
        t18[s.get("topic", "?")][s["winner"]] += 1

    topics = sorted(set(t13.keys()) | set(t13b.keys()) | set(t18.keys()))
    deltas = []
    for t in topics:
        c13, c13b, c18 = t13.get(t, Counter()), t13b.get(t, Counter()), t18.get(t, Counter())
        n13t, n13bt, n18t = sum(c13.values()), sum(c13b.values()), sum(c18.values())
        if not n18t or (not n13t and not n13bt):
            continue
        p13 = c13.get("rag", 0) / max(1, n13t) * 100
        p13b = c13b.get("rag", 0) / max(1, n13bt) * 100
        p18 = c18.get("rag", 0) / max(1, n18t) * 100
        baseline = (p13 + p13b) / 2 if (n13t and n13bt) else (p13 if n13t else p13b)
        deltas.append((t, baseline, p18, p18 - baseline))
    deltas.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'topic':<24}  {'base%':>6}  {'v18%':>6}  {'Δpp':>7}")
    for t, base, p18, d in deltas:
        marker = " ✓" if d >= 3 else (" ✗" if d <= -3 else "")
        print(f"  {t:<24}  {base:>5.1f}%  {p18:>5.1f}%  {d:>+6.1f}{marker}")

    # Verdict
    print()
    print("=" * 64)
    print("FINAL VERDICT P21")
    print("=" * 64)
    delta = rag18 - baseline_avg
    confidence_threshold = noise_band * 2  # 2× noise = high confidence
    print(f"  Baseline avg: {baseline_avg:.2f}% (noise ±{noise_band:.2f}pp)")
    print(f"  v18 P21 RAG win: {rag18:.2f}%")
    print(f"  Δ: {delta:+.2f}pp")
    print(f"  Confidence threshold (2× noise): +{confidence_threshold:.2f}pp")
    print()
    if delta >= confidence_threshold:
        print(f"  🟢 SUCCESS: +{delta:.2f}pp beats {confidence_threshold:.2f}pp threshold.")
        print(f"  → P21 confirmed working. Consider P22 multi-turn / production polish.")
    elif delta >= noise_band:
        print(f"  🟡 MODEST: +{delta:.2f}pp above noise band {noise_band:.2f}pp.")
        print(f"  → Likely real improvement but not strong. Inspect topic breakdown.")
    elif delta >= -noise_band:
        print(f"  🟠 FLAT: {delta:+.2f}pp within noise band ±{noise_band:.2f}pp.")
        print(f"  → No measurable change. P21 didn't help (but also didn't hurt).")
    else:
        print(f"  🔴 REGRESSION: {delta:+.2f}pp below noise band.")
        print(f"  → Rollback recommended.")


if __name__ == "__main__":
    main()
