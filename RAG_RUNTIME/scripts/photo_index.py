"""Photo-analysis index: maps deal_id → enrichment payload.

Loaded once at ingest start and shared across all doc builders so that
deal_profile / offer_profile / bundle / service_composition docs can be
enriched with visual/physical attributes (product_type, dimensions,
application, ROI, image URLs) that only exist in photo_analysis_raw.jsonl.

This is the connective tissue that lets RAG match "световой короб 2м"
against a deal whose SKU strings never contain the phrase "короб".
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class PhotoIndex:
    def __init__(self, raw_path: Path):
        self.raw_path = raw_path
        self._by_deal: dict[str, dict[str, Any]] = {}
        self._roi_by_direction: dict[str, list[float]] = defaultdict(list)
        self._loaded = False

    def load(self) -> "PhotoIndex":
        if self._loaded:
            return self
        if not self.raw_path.exists():
            self._loaded = True
            return self
        with open(self.raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("metadata", {}) or {}
                deal_id = str(meta.get("deal_id") or "").strip()
                if not deal_id:
                    continue
                # If duplicate deal_id, prefer the one with more content
                existing = self._by_deal.get(deal_id)
                if existing is not None:
                    if len(rec.get("searchable_text", "")) <= len(existing.get("_src_text", "")):
                        continue
                roi = meta.get("roi_romi_avg")
                if isinstance(roi, (int, float)) and roi > 0:
                    direction = meta.get("direction") or ""
                    if direction:
                        self._roi_by_direction[direction].append(float(roi))
                self._by_deal[deal_id] = {
                    "deal_id": deal_id,
                    "direction": meta.get("direction") or "",
                    "product_type": meta.get("product_type") or "",
                    "visible_text": meta.get("visible_text") or "",
                    "dimensions": meta.get("dimensions") or "",
                    "application": meta.get("application") or "",
                    "practical_value": meta.get("practical_value") or "",
                    "roi_romi_avg": roi if isinstance(roi, (int, float)) else None,
                    "roi_description": meta.get("roi_description") or "",
                    "image_urls": list(meta.get("image_urls") or []),
                    "vision_summary": (meta.get("vision_analysis") or "")[:600],
                    "deal_title": meta.get("deal_title") or "",
                    "_src_text": rec.get("searchable_text", ""),
                }
        self._loaded = True
        return self

    def get(self, deal_id: str) -> dict[str, Any] | None:
        if not deal_id:
            return None
        return self._by_deal.get(str(deal_id).strip())

    def enrichment_text(self, deal_id: str) -> str:
        """Return a short block of visual/physical attributes for injection into searchable_text."""
        rec = self.get(deal_id)
        if not rec:
            return ""
        parts = []
        if rec["product_type"]:
            parts.append(f"тип изделия: {rec['product_type']}")
        if rec["visible_text"]:
            parts.append(f"надпись: {rec['visible_text']}")
        if rec["dimensions"]:
            parts.append(f"размеры: {rec['dimensions']}")
        if rec["application"]:
            parts.append(f"применение: {rec['application']}")
        if rec["practical_value"]:
            parts.append(f"ценность для клиента: {rec['practical_value'][:200]}")
        if rec["roi_romi_avg"]:
            parts.append(f"ROMI ~{int(rec['roi_romi_avg'])}%")
        return "Визуальные характеристики: " + "; ".join(parts) if parts else ""

    def image_urls(self, deal_id: str) -> list[str]:
        rec = self.get(deal_id)
        return list(rec["image_urls"]) if rec else []

    def roi_by_direction(self) -> dict[str, float]:
        """Return mean ROMI per direction."""
        return {
            direction: round(sum(values) / len(values), 1)
            for direction, values in self._roi_by_direction.items()
            if values
        }

    def __len__(self) -> int:
        return len(self._by_deal)


_INSTANCE: PhotoIndex | None = None


def get_photo_index(data_dir: Path | None = None) -> PhotoIndex:
    """Module-level singleton. Pass data_dir on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        _INSTANCE = PhotoIndex(data_dir / "photo_analysis_raw.jsonl").load()
    return _INSTANCE
