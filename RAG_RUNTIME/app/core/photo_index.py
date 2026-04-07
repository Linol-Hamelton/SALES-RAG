"""Runtime photo-analysis index: deal_id → enrichment payload.

Loaded once at app startup from photo_analysis_raw.jsonl and used by the
segmented references builder to enrich deal_profile / offer_profile cards
with visual attributes (product_type, dimensions, application, ROI, image URLs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PhotoIndex:
    def __init__(self, raw_path: Path):
        self.raw_path = raw_path
        self._by_deal: dict[str, dict[str, Any]] = {}
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
                existing = self._by_deal.get(deal_id)
                new_len = len(rec.get("searchable_text", ""))
                if existing is not None and new_len <= existing.get("_len", 0):
                    continue
                self._by_deal[deal_id] = {
                    "deal_id": deal_id,
                    "direction": meta.get("direction") or "",
                    "product_type": meta.get("product_type") or "",
                    "visible_text": meta.get("visible_text") or "",
                    "dimensions": meta.get("dimensions") or "",
                    "application": meta.get("application") or "",
                    "practical_value": meta.get("practical_value") or "",
                    "roi_romi_avg": meta.get("roi_romi_avg"),
                    "image_urls": list(meta.get("image_urls") or []),
                    "deal_title": meta.get("deal_title") or "",
                    "_len": new_len,
                }
        self._loaded = True
        return self

    def get(self, deal_id: str | None) -> dict[str, Any] | None:
        if not deal_id:
            return None
        return self._by_deal.get(str(deal_id).strip())

    def __len__(self) -> int:
        return len(self._by_deal)
