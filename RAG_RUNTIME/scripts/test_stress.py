"""Test stress queries - saves results to file to avoid encoding issues."""
import json, urllib.request
from pathlib import Path

queries = [
    "Сделайте нам что-нибудь для офиса, бюджет небольшой",
    "Хотим вывеску",
    "Как продвигать кофейню в Махачкале?",
]

results = []
for q in queries:
    data = json.dumps({"query": q, "top_k": 8}).encode("utf-8")
    req = urllib.request.Request("http://localhost:8000/query_structured",
        data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        d = json.loads(resp.read())

    results.append({
        "query": q,
        "confidence": d.get("confidence"),
        "summary": d.get("summary", "")[:400],
        "flags": d.get("flags", []),
        "risks": d.get("risks", []),
        "reasoning": d.get("reasoning", "")[:300],
        "refs": [{"type": r["doc_type"], "score": r["score"], "snippet": r["snippet"][:100]}
                 for r in d.get("references", [])[:4]],
        "estimated_price": d.get("estimated_price"),
        "latency_ms": d.get("latency_ms"),
    })

out = Path("reports/stress_test_results.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Results saved to {out}")
for r in results:
    print(f"\n=== {r['query']}")
    print(f"  confidence: {r['confidence']}")
    print(f"  summary: {r['summary'][:300]}")
    print(f"  flags: {r['flags']}")
    print(f"  refs types: {[x['type'] for x in r['refs']]}")
    print(f"  latency: {r['latency_ms']} ms")
