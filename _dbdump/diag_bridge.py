"""Quick diagnostic: are service_pricing_bridge docs in Qdrant?
Run inside labus_api container."""
from app.config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

c = QdrantClient(url=settings.qdrant_url)
col = settings.qdrant_collection
print(f"Collection: {col}")
print(f"Total points: {c.count(col).count}")

# doc_type-indexed, should be fast
bridge_total = c.count(
    col,
    count_filter=Filter(must=[
        FieldCondition(key="doc_type", match=MatchValue(value="service_pricing_bridge")),
    ]),
).count
print(f"service_pricing_bridge docs: {bridge_total}")

# List all bridge docs with their 'service' field
res, _ = c.scroll(
    col,
    scroll_filter=Filter(must=[
        FieldCondition(key="doc_type", match=MatchValue(value="service_pricing_bridge")),
    ]),
    limit=20,
    with_payload=True,
    with_vectors=False,
)
print(f"\nBridge docs present ({len(res)}):")
for r in res:
    p = r.payload
    print(f"  id={r.id} doc_id={p.get('doc_id')!r} service={p.get('service')!r}")

# Now test the exact filter used in _force_inject_bridge
print("\nTest filter used by _force_inject_bridge:")
for svc in ["Дизайн логотипа", "Дизайн брендбука", "Дизайн фирменного стиля"]:
    res2, _ = c.scroll(
        col,
        scroll_filter=Filter(must=[
            FieldCondition(key="doc_type", match=MatchValue(value="service_pricing_bridge")),
            FieldCondition(key="service", match=MatchValue(value=svc)),
        ]),
        limit=1,
        with_payload=True,
    )
    print(f"  service={svc!r} → {len(res2)} hits")
