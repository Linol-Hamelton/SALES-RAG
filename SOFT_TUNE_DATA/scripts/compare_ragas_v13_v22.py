"""P22.C.3: compare RAGAS scores between v13 baseline and v22 (or any two runs).

Reads:
  SOFT_TUNE_DATA/scores_ragas_v13.jsonl  (baseline)
  SOFT_TUNE_DATA/scores_ragas_v22.jsonl  (new)

Writes:
  SOFT_TUNE_DATA/ragas_comparison_v13_v22.md  (full report)

Usage:
  python SOFT_TUNE_DATA/scripts/compare_ragas_v13_v22.py [--a v13] [--b v22]

Reports:
  - Per-metric distribution (mean, p25, p50, p75, p95)
  - Per-topic breakdown
  - Top 20 lowest-faithfulness queries (debug priority)
  - Hallucination examples
  - Overall verdict vs P22 targets:
      Faithfulness ≥0.95, Answer Relevancy ≥0.92,
      Context Precision ≥0.85, Context Recall ≥0.88,
      Composite ≥0.90
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "SOFT_TUNE_DATA"

P22_TARGETS = {
    "faithfulness": 0.95,
    "answer_relevancy": 0.92,
    "context_precision": 0.85,
    "context_recall": 0.88,
    "composite": 0.90,
}

METRICS = ("faithfulness", "answer_relevancy", "context_precision",
           "context_recall", "composite")


def load_scores(suffix: str) -> list[dict]:
    p = DATA_DIR / f"scores_ragas_{suffix}.jsonl"
    if not p.exists():
        print(f"!! {p} not found", file=sys.stderr)
        sys.exit(1)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [r for r in rows if r.get("error") is None]


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def distribution(rows: list[dict], metric: str) -> dict:
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    if not vals:
        return {"n": 0, "mean": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 3),
        "p25": round(percentile(vals, 0.25), 3),
        "p50": round(percentile(vals, 0.5), 3),
        "p75": round(percentile(vals, 0.75), 3),
        "p95": round(percentile(vals, 0.95), 3),
    }


def by_topic(rows: list[dict], metric: str) -> dict:
    topic_vals = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None and r.get("topic"):
            topic_vals[r["topic"]].append(v)
    return {t: round(statistics.mean(vs), 3) for t, vs in sorted(topic_vals.items())}


def fmt_diff(a: float, b: float) -> str:
    diff = b - a
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    return f"{arrow} {diff:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="v13", help="Baseline run suffix")
    ap.add_argument("--b", default="v22", help="New run suffix")
    ap.add_argument("--top-n-bad", type=int, default=20,
                    help="How many lowest-faithfulness queries to dump")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    a_rows = load_scores(args.a)
    b_rows = load_scores(args.b)
    print(f"Loaded {len(a_rows)} ({args.a}) + {len(b_rows)} ({args.b}) rows")

    lines = [
        f"# RAGAS Comparison: {args.a} (baseline) vs {args.b}\n",
        f"Samples: {len(a_rows)} ({args.a}) | {len(b_rows)} ({args.b})\n\n",
        "## Per-metric distribution\n",
        f"| Metric | {args.a} mean | {args.b} mean | Δ | p25→p95 {args.b} | Target | Met? |",
        "|---|---:|---:|---:|---|---:|:---:|",
    ]

    for metric in METRICS:
        da = distribution(a_rows, metric)
        db = distribution(b_rows, metric)
        target = P22_TARGETS.get(metric, 0)
        met = "✓" if db["mean"] >= target else "✗"
        delta = fmt_diff(da["mean"], db["mean"])
        lines.append(
            f"| {metric} | {da['mean']:.3f} | {db['mean']:.3f} | {delta} | "
            f"{db['p25']:.2f}→{db['p95']:.2f} | {target:.2f} | {met} |"
        )

    # Per-topic breakdown — composite only
    lines.append("\n## Per-topic composite score\n")
    lines.append(f"| Topic | {args.a} | {args.b} | Δ |")
    lines.append("|---|---:|---:|---:|")

    a_by_topic = by_topic(a_rows, "composite")
    b_by_topic = by_topic(b_rows, "composite")
    all_topics = sorted(set(a_by_topic) | set(b_by_topic))
    for t in all_topics:
        va = a_by_topic.get(t, 0)
        vb = b_by_topic.get(t, 0)
        lines.append(f"| {t} | {va:.3f} | {vb:.3f} | {fmt_diff(va, vb)} |")

    # Worst faithfulness queries в новой версии
    bad = sorted(b_rows, key=lambda r: r.get("faithfulness") or 0)[: args.top_n_bad]
    lines.append(f"\n## Top {args.top_n_bad} lowest-faithfulness queries in {args.b}\n")
    for r in bad:
        lines.append(
            f"- **{r.get('id')}** ({r.get('topic')}, intent={r.get('intent')}): "
            f"faith={r.get('faithfulness'):.2f} composite={r.get('composite'):.2f}"
        )
        lines.append(f"  - Q: {(r.get('query') or '')[:140]}")
        if r.get("hallucinations"):
            lines.append(f"  - Hallucinations: {r['hallucinations'][:3]}")
        if r.get("reasoning"):
            lines.append(f"  - Reasoning: {(r['reasoning'] or '')[:200]}")
        lines.append("")

    # Verdict
    lines.append("\n## Verdict vs P22 targets\n")
    all_met = True
    for m, target in P22_TARGETS.items():
        actual = distribution(b_rows, m)["mean"]
        ok = actual >= target
        all_met = all_met and ok
        lines.append(f"- **{m}**: {actual:.3f} vs target {target:.2f} — "
                     f"{'✓ MET' if ok else '✗ MISS (' + f'{target - actual:+.3f}' + ')'}")
    lines.append(
        f"\n**OVERALL: {'✓ ALL P22 TARGETS MET' if all_met else '✗ SOME TARGETS MISSED'}**"
    )

    out = DATA_DIR / f"ragas_comparison_{args.a}_{args.b}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")

    # Also dump short summary to stdout
    print("\n=== SHORT SUMMARY ===")
    for m in METRICS:
        da = distribution(a_rows, m)
        db = distribution(b_rows, m)
        target = P22_TARGETS.get(m, 0)
        met = "OK" if db["mean"] >= target else "MISS"
        print(f"  {m:22s}: {da['mean']:.3f} → {db['mean']:.3f} ({fmt_diff(da['mean'], db['mean'])}) "
              f"[target {target:.2f}: {met}]")


if __name__ == "__main__":
    main()
