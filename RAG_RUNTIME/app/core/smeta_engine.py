"""
SmetaEngine: deterministic deal-estimate builder from pre-computed templates.

Architecture:
- Categories and canonical smetas are built offline by
  RAG_ANALYTICS/buildSmetaTemplates.mjs → smeta_templates.json
- Category embeddings are built offline by
  scripts/embed_smeta_categories.py → smeta_category_embeddings.npy
- At runtime SmetaEngine:
  1. Embeds the user query with the same BGE-M3 model used for retrieval.
  2. Finds top-N categories by cosine similarity.
  3. Returns the canonical smeta positions as DealItems with statistical
     prices (mean of 4 metrics) — no LLM hallucination.

Match quality levels:
- "maximal":   cosine > 0.70  → high confidence, use template as-is.
- "significant": 0.55–0.70   → use template, flag for manager review.
- "minimal":   0.40–0.55     → return template but flag as low-confidence.
- below 0.40:  no match, caller falls back to legacy LLM pipeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from app.schemas.pricing import DealItem
from app.utils.logging import get_logger

logger = get_logger("smeta_engine")

MatchQuality = Literal["maximal", "significant", "minimal", "none"]

# П7.7-A / П8.4: safety net — keyword override семантического ретривера. Если
# запрос содержит include-слова без exclude-слов, форсируем конкретную категорию
# минуя cosine-ранжирование. Нужно для коротких запросов («Сколько стоит
# логотип?», «Световая вывеска ресторана»), где enrichment keywords_text всё
# равно не вытягивают матч против конкурентов с большим n (например,
# «Титульные вывески» перебивают по bias log10(n)).
#
# Каждый override имеет поле `strong`:
#   - strong=True  → уверенный матч, forced_sim пола = SIM_MAXIMAL (auto-tier).
#                    Для коротких чётких запросов, где мы знаем категорию.
#   - strong=False → мягкий матч, пол = SIM_SIGNIFICANT (guided-tier).
#                    Используется для неоднозначных случаев.
#
# include: ВСЕ регексы должны матчиться.
# exclude: НИ ОДИН не должен матчиться (мерч-носители, под-ключ варианты,
#          пакеты/комплексы → ожидают bundle-флоу, а не canonical).
# target: имя категории как в smeta_templates.json.
_KEYWORD_OVERRIDES: list[dict] = [
    {
        "target": "Логотип",
        "strong": True,
        "include": [re.compile(r"логотип|\bлого\b", re.IGNORECASE)],
        "exclude": [
            re.compile(r"под\s*ключ|брифинг|концепц|фирменн.*стил|брендбук|айдентик|разработк.*с\s*нул|креатив|нейминг", re.IGNORECASE),
            re.compile(r"ручк|кружк|футболк|шоппер|худи|толстовк|магнит|шоколад|скотч|значок|брелок|бейдж", re.IGNORECASE),
        ],
    },
    # П8.4: «Объёмные буквы» — стабильная L1-категория, нужна L3.
    {
        "target": "Объемные буквы",
        "strong": True,
        "include": [re.compile(r"букв", re.IGNORECASE)],
        "exclude": [
            # bundle / комплексы / под-ключ → отдаются bundle-флоу
            re.compile(r"под\s*ключ|пакет\s+услуг|полный\s+комплект|\bкомплект\b|комплекс|набор", re.IGNORECASE),
            # печатная продукция: «буквы на листовке», «буквы на баннере»
            re.compile(r"на\s+листовк|на\s+баннер|на\s+этикет|визитк", re.IGNORECASE),
        ],
    },
    # П8.4: «Световые вывески» — стабильная L1-категория, нужна L3.
    {
        "target": "Световые вывески",
        "strong": True,
        "include": [
            re.compile(r"вывеск", re.IGNORECASE),
            re.compile(r"светов|светодиод|подсветк|объ[её]мн|контражур|led|лед|композит|акрил|неон", re.IGNORECASE),
        ],
        "exclude": [
            # Титульные вывески — отдельный LABUS-кластер
            re.compile(r"титульн", re.IGNORECASE),
            # «под ключ» / bundle-запросы — в bundle-флоу
            re.compile(r"под\s*ключ|пакет\s+услуг|полный\s+комплект|комплекс|\bвсе?\s+что\s+входит", re.IGNORECASE),
            # ремонт — отдельный manual-флоу
            re.compile(r"ремонт|починк|восстановл", re.IGNORECASE),
        ],
    },
]


def _check_keyword_override(query: str) -> Optional[tuple[str, bool]]:
    """Return (forced_category_name, is_strong) or None.

    is_strong=True → override уверенный, forced_sim пола = SIM_MAXIMAL.
    is_strong=False → мягкий override, пол = SIM_SIGNIFICANT.
    """
    if not query:
        return None
    for rule in _KEYWORD_OVERRIDES:
        if not all(rx.search(query) for rx in rule["include"]):
            continue
        if any(rx.search(query) for rx in rule["exclude"]):
            continue
        return (rule["target"], bool(rule.get("strong", False)))
    return None


@dataclass
class SmetaResult:
    """Result of building a smeta from templates."""
    category_id: str = ""
    category_name: str = ""
    match_quality: MatchQuality = "none"
    match_similarity: float = 0.0
    match_reason: str = ""
    deal_items: list[DealItem] = field(default_factory=list)
    total: float = 0.0
    price_band_min: float = 0.0
    price_band_max: float = 0.0
    confidence: Literal["high", "medium", "low"] = "low"
    flags: list[str] = field(default_factory=list)
    source_deal_id: str = ""
    source_deal_type: str = ""
    deals_in_category: int = 0

    @property
    def is_usable(self) -> bool:
        return self.match_quality in ("maximal", "significant", "minimal")


class SmetaEngine:
    """Template-based deal estimator."""

    # Calibrated for keywords_text v2 (примеры заказов first, длинные тексты).
    # Sims естественно ниже на ~0.05–0.10 на тех же запросах vs v1.
    SIM_MAXIMAL = 0.55
    SIM_SIGNIFICANT = 0.45
    SIM_MINIMAL = 0.38

    def __init__(self, templates_path: Path, embeddings_path: Path):
        self.templates_path = Path(templates_path)
        self.embeddings_path = Path(embeddings_path)
        self._templates: Optional[dict] = None
        self._categories: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, 1024)
        self._loaded = False

    def load(self) -> "SmetaEngine":
        if not self.templates_path.exists():
            logger.warning("smeta_templates.json not found", path=str(self.templates_path))
            self._loaded = False
            return self
        if not self.embeddings_path.exists():
            logger.warning("smeta_category_embeddings.npy not found",
                           path=str(self.embeddings_path))
            self._loaded = False
            return self

        with open(self.templates_path, encoding="utf-8") as f:
            self._templates = json.load(f)
        self._categories = self._templates.get("categories", [])
        self._embeddings = np.load(self.embeddings_path).astype(np.float32)

        if self._embeddings.shape[0] != len(self._categories):
            logger.error("Embedding/category count mismatch",
                         categories=len(self._categories),
                         embeddings=self._embeddings.shape[0])
            self._loaded = False
            return self

        self._loaded = True
        logger.info("SmetaEngine loaded",
                    categories=len(self._categories),
                    deals=self._templates.get("total_deals_analyzed", 0),
                    embedding_dim=self._embeddings.shape[1])
        return self

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._embeddings is not None and len(self._categories) > 0

    def find_category(
        self, query_embedding: np.ndarray | list[float], top_k: int = 3
    ) -> list[tuple[int, float]]:
        """Return list of (category_index, biased_similarity) sorted desc.

        Score = cosine_sim * (1 + 0.05 * log10(max(deals_count, 1))).
        Bias is mild (≤+0.13 for n=1000) — не перебивает семантику, но
        ломает шумовые ничьи в пользу крупных категорий с реальной историей.
        """
        if not self.is_ready:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        # Both embeddings are L2-normalized (BGE-M3 normalize_embeddings=True),
        # so dot product == cosine similarity.
        sims = (self._embeddings @ q.T).flatten()
        deals_counts = np.array(
            [c.get("deals_count", 1) or 1 for c in self._categories],
            dtype=np.float32,
        )
        # Stronger bias: log10(n)*0.15. Hard penalty 0.85 for n<10 categories —
        # they exist for fallback but should not outrank n>=10 on noise.
        bias = 1.0 + 0.15 * np.log10(np.maximum(deals_counts, 1.0))
        small_penalty = np.where(deals_counts < 10, 0.85, 1.0).astype(np.float32)
        scores = sims * bias * small_penalty
        top_idx = np.argsort(scores)[::-1][:top_k]
        # Return raw cosine sim (not biased) so quality thresholds remain meaningful.
        return [(int(i), float(sims[i])) for i in top_idx]

    def build_smeta(
        self,
        query: str,
        query_embedding: np.ndarray | list[float],
        decomp: Optional[dict] = None,
    ) -> SmetaResult:
        """Find best category and produce DealItems from its canonical smeta."""
        if not self.is_ready:
            return SmetaResult(match_reason="SmetaEngine not loaded")

        # П7.7-A / П8.4: keyword override — проверяем до семантики.
        override = _check_keyword_override(query)
        forced_idx: Optional[int] = None
        forced_strong = False
        if override is not None:
            forced_name, forced_strong = override
            for i, c in enumerate(self._categories):
                if c.get("category_name") == forced_name:
                    forced_idx = i
                    break
            if forced_idx is not None:
                logger.info("keyword override applied",
                            category=forced_name,
                            strong=forced_strong,
                            query=query[:80])

        matches = self.find_category(query_embedding, top_k=3)
        if not matches and forced_idx is None:
            return SmetaResult(match_reason="no category matches")

        if forced_idx is not None:
            # cosine между запросом и принудительной категорией — чтобы reason
            # был честным по качеству, а не поддельный 1.0.
            q = np.asarray(query_embedding, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            forced_sim = float((self._embeddings[forced_idx] @ q.T).flatten()[0])
            best_idx = forced_idx
            # П8.4: strong override → пол = SIM_MAXIMAL (auto-tier). Раньше всегда
            # использовался SIM_SIGNIFICANT, из-за чего override никогда не мог
            # дотянуть до auto-confidence даже для очевидных коротких запросов
            # типа «логотип цена».
            sim_floor = self.SIM_MAXIMAL if forced_strong else self.SIM_SIGNIFICANT
            best_sim = max(forced_sim, sim_floor)
        else:
            best_idx, best_sim = matches[0]
        cat = self._categories[best_idx]

        # Quality classification
        if best_sim >= self.SIM_MAXIMAL:
            quality: MatchQuality = "maximal"
            reason = f"точное совпадение категории «{cat['category_name']}» (сходство {best_sim:.0%})"
        elif best_sim >= self.SIM_SIGNIFICANT:
            quality = "significant"
            reason = f"значимое совпадение с «{cat['category_name']}» (сходство {best_sim:.0%}) — рекомендуется ручная проверка"
        elif best_sim >= self.SIM_MINIMAL:
            quality = "minimal"
            reason = f"слабое совпадение с «{cat['category_name']}» (сходство {best_sim:.0%}) — смета может быть нерелевантна"
        else:
            return SmetaResult(
                category_name=cat.get("category_name", ""),
                match_quality="none",
                match_similarity=best_sim,
                match_reason=f"нет подходящей категории (лучший кандидат {cat.get('category_name','?')} — {best_sim:.0%})",
            )

        canonical = cat.get("canonical_smeta", {})
        positions = canonical.get("positions", [])
        if not positions:
            return SmetaResult(
                category_id=cat.get("category_id", ""),
                category_name=cat.get("category_name", ""),
                match_quality="none",
                match_similarity=best_sim,
                match_reason="канонический шаблон пуст",
            )

        # Decomp-driven scaling (Fix D): когда пользователь сказал «7 букв 40 см»,
        # decomp={'letter_count': 7, 'height_cm': 40, ...}. Per-letter позиции
        # (сборка/покраска/монтаж буквы) скейлятся на letter_count, прочие — нет.
        letter_count = 0
        if decomp:
            for k in ("letter_count", "letters", "char_count", "n_letters"):
                v = decomp.get(k) if isinstance(decomp, dict) else None
                if v:
                    try:
                        letter_count = int(v)
                        break
                    except (TypeError, ValueError):
                        pass

        def _scale_quantity(p: dict, base_q: float) -> tuple[float, bool]:
            """Return (effective_quantity, was_scaled)."""
            if letter_count <= 0:
                return base_q, False
            unit = (p.get("unit") or "").lower()
            section = (p.get("b24_section") or "").lower()
            name = (p.get("product_name") or "").lower()
            per_letter_unit = unit in ("шт", "буква", "буквы", "symbol", "символ")
            per_letter_hint = any(
                kw in name or kw in section
                for kw in ("буква", "символ", "элемент", "сборк", "покраск",
                           "монтаж букв", "лицев", "торц", "контражур")
            )
            if per_letter_unit and per_letter_hint and base_q <= letter_count:
                return float(letter_count), True
            return base_q, False

        # Convert positions to DealItems
        deal_items: list[DealItem] = []
        flags: list[str] = []
        total_std_sq = 0.0
        n_with_std = 0
        any_low_conf = False
        any_scaled = False

        for p in positions:
            stats = p.get("price_stats", {}) or {}
            unit_price = float(stats.get("final", 0) or 0)
            base_q = float(p.get("quantity_typical", 1) or 1)
            quantity, was_scaled = _scale_quantity(p, base_q)
            if was_scaled:
                any_scaled = True
            item_total = round(unit_price * quantity, 2)
            confidence = stats.get("confidence", "low")
            freq = float(p.get("frequency", 0) or 0)
            sample = int(stats.get("sample_size", 0) or 0)

            notes_parts = [
                f"статистика по {sample} сделкам" if sample else "нет статистики",
                f"частота в категории {freq:.0%}" if freq else "",
                f"уверенность: {confidence}",
            ]
            deal_items.append(DealItem(
                product_name=p.get("product_name", ""),
                quantity=quantity,
                unit=p.get("unit", "шт") or "шт",
                unit_price=round(unit_price, 2),
                total=item_total,
                b24_section=p.get("b24_section", "") or "",
                notes=" · ".join(n for n in notes_parts if n),
            ))

            std = float(stats.get("std", 0) or 0)
            if std > 0:
                total_std_sq += (std * quantity) ** 2
                n_with_std += 1
            if confidence == "low":
                any_low_conf = True

        total = round(sum(di.total for di in deal_items), 2)

        # Overall confidence
        if quality == "maximal" and not any_low_conf:
            overall_conf: Literal["high", "medium", "low"] = "high"
        elif quality in ("maximal", "significant"):
            overall_conf = "medium"
        else:
            overall_conf = "low"

        # Price band: total ± aggregated std (at least ±10% for safety)
        agg_std = float(np.sqrt(total_std_sq)) if n_with_std > 0 else 0.0
        band_delta = max(agg_std, total * 0.10)
        band_min = round(max(0.0, total - band_delta), 2)
        band_max = round(total + band_delta, 2)

        if any_low_conf:
            flags.append("Высокая вариативность цены — требует ручной проверки")
        if cat.get("deals_count", 0) < 10:
            flags.append(f"Категория малочисленна ({cat['deals_count']} сделок) — оценка приблизительная")
        if any_scaled and letter_count > 0:
            flags.append(f"Количество позиций пересчитано на {letter_count} букв из запроса")

        return SmetaResult(
            category_id=cat.get("category_id", ""),
            category_name=cat.get("category_name", ""),
            match_quality=quality,
            match_similarity=best_sim,
            match_reason=reason,
            deal_items=deal_items,
            total=total,
            price_band_min=band_min,
            price_band_max=band_max,
            confidence=overall_conf,
            flags=flags,
            source_deal_id=canonical.get("source_deal_id", ""),
            source_deal_type=canonical.get("source_deal_type", ""),
            deals_in_category=int(cat.get("deals_count", 0)),
        )
