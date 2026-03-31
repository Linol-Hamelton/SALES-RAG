"""Quick test of the parametric pipeline."""
import asyncio
import json
import urllib.request


def test(query: str) -> None:
    data = json.dumps({"query": query, "top_k": 8}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/query_structured",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read())

    print(f"\nQUERY: {query}")
    print(f"  confidence: {d.get('confidence')}")
    ep = d.get("estimated_price")
    print(f"  estimated_price: {ep}")
    pb_data = d.get("parametric_breakdown")
    if pb_data and pb_data.get("line_items"):
        print("  PARAMETRIC BREAKDOWN:")
        for li in pb_data["line_items"]:
            print(
                f"    {li['label']}: {li['quantity']} {li['unit']} "
                f"x {li['unit_price']:,.0f} = {li['total']:,.0f} ({li['confidence_tier']})"
            )
        print(f"    TOTAL: {pb_data['total_estimate']:,.0f}")
        print(f"    Range: {pb_data['total_min']:,.0f}-{pb_data['total_max']:,.0f}")
        print(f"    missing: {pb_data.get('missing_components')}")
    else:
        print("  NO PARAMETRIC BREAKDOWN (simple path)")
    print(f"  latency: {d.get('latency_ms')} ms")


# Also test the decomposer directly
from app.core.query_decomposer import decompose

queries = [
    "объемные буквы БАРДЕРШОП 80 см",
    "объемные буквы ДАГАВТОТРАНС 80 см монтаж спецтехника 8 часов",
    "баннер 3х6",
]

print("=== DECOMPOSER CHECK ===")
for q in queries:
    d = decompose(q)
    print(f"\nQuery: {q}")
    print(f"  is_complex: {d.is_complex}")
    print(f"  letter_text: '{d.letter_text}' ({d.letter_count} chars)")
    print(f"  height_cm: {d.height_cm}")
    print(f"  linear_meters: {d.linear_meters}")
    print(f"  hours: {d.hours}")
    print(f"  components: {[c.type for c in d.components]}")

print("\n=== API TEST ===")
for q in queries:
    test(q)
