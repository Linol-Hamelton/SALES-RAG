"""
LLM generation via Deepseek API (OpenAI-compatible).
Handles both human-readable and structured JSON responses.
"""
import json
import re
import yaml
from pathlib import Path
from typing import Any
from app.utils.logging import get_logger
from app.utils.bitrix import format_deal_link_line

logger = get_logger(__name__)


def _load_prompts(settings) -> dict:
    """Load prompt templates from prompts.yaml."""
    prompts_path = Path(settings.project_root) / "configs" / "prompts.yaml"
    if prompts_path.exists():
        with open(prompts_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "system": "Ты — ценовой консультант компании Лабус (реклама и вывески). Отвечай только на основе контекста.",
        "human_query": "ВОПРОС:\n{query}\n\nКОНТЕКСТ:\n{context}\n\nОтветь на вопрос используя только данные контекста.",
        "structured_query": "ВОПРОС:\n{query}\n\nКОНТЕКСТ:\n{context}\n\nВерни ответ строго в JSON формате.",
    }


# Anti-pattern phrases operators flagged in prod feedback (chats #94/288, #95/294).
# LLM keeps reaching for "рыночные данные для Дагестана" despite system-prompt
# rules; this is a deterministic post-LLM scrub. Only rewrites the offending
# phrasing — does NOT touch numbers or product info.
_FORBIDDEN_PHRASE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bпо\s+рынк[уа]\s+Дагестан[аеу]?\b", re.IGNORECASE), "по внутренним ориентирам цеха"),
    (re.compile(r"\bрыночн[ыоа][ме]\s+ориентир[ыау]?\s+по\s+Дагестан[уаеы]?\b", re.IGNORECASE), "внутренние ориентиры цеха"),
    (re.compile(r"\bрыночн[ыоа][ме]\s+(?:данны[ме]|данных)\s+(?:для\s+)?(?:Дагестан[аеу]?|регион[аеу]?)\b", re.IGNORECASE), "внутренние ориентиры Лабус"),
    (re.compile(r"\bв\s+средн[еио][мй]\s+по\s+рынк[уа]\b", re.IGNORECASE), "по внутренним ориентирам цеха"),
    (re.compile(r"\bсредн[еио][мй]\s+рыночн[ыоа][емй]?\s+(?:цен[ауы]|стоимост[ьи])\b", re.IGNORECASE), "цена по внутренним ориентирам Лабус"),
]


def _scrub_forbidden_phrases(text: str) -> str:
    """Replace phrases operators flagged as critical errors. Idempotent.

    Applied to summary/reasoning/basis fields after LLM returns. Numbers and
    product references are untouched — only the offending sourcing language
    («рыночные данные для Дагестана») is rewritten to internal sourcing.
    """
    if not text or not isinstance(text, str):
        return text
    out = text
    for rx, replacement in _FORBIDDEN_PHRASE_PATTERNS:
        out = rx.sub(replacement, out)
    return out


def _scrub_structured_response(payload: dict) -> dict:
    """Apply phrase scrubbing to all string fields of a structured response."""
    if not isinstance(payload, dict):
        return payload
    for key in ("summary", "reasoning"):
        if key in payload and isinstance(payload[key], str):
            payload[key] = _scrub_forbidden_phrases(payload[key])
    ep = payload.get("estimated_price")
    if isinstance(ep, dict) and isinstance(ep.get("basis"), str):
        ep["basis"] = _scrub_forbidden_phrases(ep["basis"])
    flags = payload.get("flags")
    if isinstance(flags, list):
        payload["flags"] = [_scrub_forbidden_phrases(f) if isinstance(f, str) else f for f in flags]
    return payload


def _classify_deal_profile(payload: dict) -> str:
    """Classify a deal_profile as repair, full-cycle, or production-only."""
    summary = (payload.get("component_summary", "") or "").lower()
    materials = (payload.get("materials", "") or "").lower()
    all_text = f"{summary} {materials}"

    has_demolition = "демонтаж" in all_text
    has_fabrication = any(kw in all_text for kw in [
        "сборка объема", "сборка объёма", "акрил", "пвх", "композит", "фрезеров",
    ])
    has_mounting = any(kw in all_text for kw in [
        "монтаж вывески", "монтаж буквы", "монтаж букв", "монтаж конструкции",
    ])
    has_lighting = any(kw in all_text for kw in ["led", "светодиод", "подсветка", "неон"])

    if has_demolition and not has_fabrication:
        return "РЕМОНТ/ДЕМОНТАЖ"
    if has_fabrication and has_mounting and has_lighting:
        return "ПОЛНЫЙ ЦИКЛ"
    if has_fabrication and not has_mounting:
        return "ТОЛЬКО ПРОИЗВОДСТВО"
    return "НЕПОЛНЫЙ КОМПЛЕКТ"


_BUNDLE_SERVICE_CHECKS = {
    "монтаж": ["монтаж вывески", "монтаж буквы", "монтаж букв", "монтаж конструкции"],
    "каркас": ["сварка каркаса", "каркас", "профильная труба", "рама"],
    "дизайн": ["дизайн", "макет", "визуализация"],
    "подсветка": ["led", "светодиод", "подсветка", "неон", "лента"],
}


def _classify_bundle_completeness(payload: dict) -> tuple[str, list[str]]:
    """Classify bundle by service completeness. Returns (label, missing_services)."""
    sample_products = (payload.get("sample_products", "") or "").lower()

    missing = []
    for service_name, keywords in _BUNDLE_SERVICE_CHECKS.items():
        if not any(kw in sample_products for kw in keywords):
            missing.append(service_name)

    if not missing:
        return "ПОЛНЫЙ КОМПЛЕКТ", []
    if len(missing) >= 3:
        return "ТОЛЬКО ПРОИЗВОДСТВО", missing
    return "НЕПОЛНЫЙ КОМПЛЕКТ", missing


def _format_context_block(docs: list[dict], pricing_resolution=None) -> str:
    """Format retrieved docs into a context block for the LLM prompt."""
    blocks = []

    for i, doc in enumerate(docs[:12]):  # max 12 docs in context
        payload = doc.get("payload", {})
        doc_type = payload.get("doc_type", "")
        score = doc.get("final_score", doc.get("rrf_score", 0))

        if doc_type == "product":
            product_name = payload.get("product_name", "?")
            direction = payload.get("direction", "")
            base_price = payload.get("current_base_price")
            price_mode = payload.get("price_mode", "manual")
            confidence = payload.get("confidence_tier", "low")
            order_rows = payload.get("order_rows", 0)
            total_order_qty = payload.get("total_order_qty", 0) or 0
            p50 = payload.get("order_price_p50")
            manual_reason = payload.get("manual_review_reason", "")
            recommended = payload.get("recommended_price")
            suggested_min = payload.get("suggested_min")
            suggested_max = payload.get("suggested_max")
            price_ratio_range = payload.get("price_ratio_range")

            # Translate manual_review_reason to Russian
            REASON_RU = {
                "insufficient_history": "мало заказов",
                "high_price_variance": "высокая вариативность цены",
                "financial_modifier": "финансовый модификатор",
                "manual_or_missing_catalog": "нестандартный продукт",
                "no_price_anchor": "нет ценового ориентира",
            }

            section_name = payload.get("section_name", "")
            product_id = payload.get("product_id") or payload.get("product_key") or ""
            # P10.5: каталог (goods.csv) использует PRODUCT_ID как первичный ключ;
            # GOOD_ID встречается только в offers/orders (транзакционные line-item id).
            # Поэтому для product-документов показываем PRODUCT_ID — его LLM и должен
            # класть в deal_items[].good_id структурированного ответа.
            art_tag = f" PRODUCT_ID={product_id}" if product_id else ""
            labus_url = payload.get("labus_url") or ""
            url_tag = f" | {labus_url}" if labus_url else ""
            lines = [f"[Продукт{art_tag}{url_tag}] {product_name}" + (f" ({direction})" if direction else "") + (f" [{section_name}]" if section_name else "")]
            if base_price:
                lines.append(f"  Базовая цена: {base_price:.0f} руб")
            if recommended:
                lines.append(f"  Рекомендованная цена: {recommended:.0f} руб")
            if suggested_min and suggested_max:
                lines.append(f"  Диапазон: {suggested_min:.0f} – {suggested_max:.0f} руб")
            if p50:
                lines.append(f"  Медиана заказов: {p50:.0f} руб")
            lines.append(f"  Режим: {price_mode} | Уверенность: {confidence}")
            if order_rows:
                lines.append(f"  Строк заказов: {order_rows}")
            if total_order_qty:
                lines.append(f"  Всего заказано: {total_order_qty} шт")
            if price_ratio_range is not None:
                lines.append(f"  Разброс цен: {price_ratio_range:.2f}")
            if manual_reason:
                reason_text = REASON_RU.get(manual_reason, manual_reason)
                lines.append(f"  Причина ручной проверки: {reason_text}")
            analogs = payload.get("nearest_analogs", [])
            if analogs:
                lines.append(f"  Аналоги: {', '.join(a for a in analogs if a)}")
            blocks.append("\n".join(lines))

        elif doc_type == "bundle":
            title = payload.get("sample_title", payload.get("bundle_key", "?"))
            direction = payload.get("direction", "")
            median_value = payload.get("median_deal_value")
            product_count = payload.get("product_count", 0)
            deal_count = payload.get("deal_count", 0)
            sample_products = payload.get("sample_products", "")
            contained = payload.get("contained_matches", 0)

            completeness, missing = _classify_bundle_completeness(payload)
            bundle_ids = payload.get("sample_deal_ids") or ""
            bundle_tag = f" сделки:{bundle_ids}" if bundle_ids else ""
            lines = [f"[Набор ({completeness}){bundle_tag}] {title}" + (f" ({direction})" if direction else "")]
            if missing:
                lines.append(f"  НЕ ВКЛЮЧАЕТ: {', '.join(missing)} — цена заниженная для полного заказа")
            if product_count:
                lines.append(f"  Товаров в наборе: {product_count}")
            if median_value:
                lines.append(f"  Медиана стоимости: {median_value:.0f} руб ({deal_count} сделок)")
            if contained:
                lines.append(f"  Встречался в реальных заказах: {contained} раз")
            if sample_products:
                lines.append(f"  Состав: {sample_products[:400]}")
            if bundle_ids:
                from app.utils.bitrix import collect_deal_urls
                _ids = [x.strip() for x in str(bundle_ids).replace(";", ",").split(",") if x.strip()]
                urls = collect_deal_urls(_ids)
                if urls:
                    lines.append(f"  Bitrix-ссылки: {', '.join(urls[:5])}")
            blocks.append("\n".join(lines))

        elif doc_type == "pricing_policy":
            rule = payload.get("rule", payload.get("description", ""))
            price_mode = payload.get("price_mode", "")
            lines = [f"[Политика цен: {price_mode}]"]
            if rule:
                lines.append(f"  {rule[:300]}")
            blocks.append("\n".join(lines))

        elif doc_type == "faq":
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            category = payload.get("category", "")
            lines = [f"[FAQ: {category}]" if category else "[FAQ]"]
            if question:
                lines.append(f"  Q: {question}")
            if answer:
                lines.append(f"  A: {answer[:800]}")
            blocks.append("\n".join(lines))

        elif doc_type == "service_composition":
            direction = payload.get("direction", "?")
            deal_count = payload.get("deal_count", 0)
            median_value = payload.get("median_value")
            core_products = payload.get("core_products", "")
            optional_products = payload.get("optional_products", "")
            materials = payload.get("materials", "")
            cross_directions = payload.get("cross_directions", "")

            lines = [f"[Состав услуг: {direction}] ({deal_count} сделок)"]
            if median_value:
                lines.append(f"  Медиана стоимости: {median_value:.0f} руб")
            if core_products:
                lines.append(f"  Основные компоненты: {core_products[:500]}")
            if optional_products:
                lines.append(f"  Доп. компоненты: {optional_products[:300]}")
            if materials:
                lines.append(f"  Материалы: {materials}")
            if cross_directions:
                lines.append(f"  Сочетается с: {cross_directions}")
            blocks.append("\n".join(lines))

        elif doc_type == "deal_profile":
            title = payload.get("title", "?")
            direction = payload.get("direction", "")
            line_total = payload.get("line_total")
            duration = payload.get("deal_duration_days")
            component_summary = payload.get("component_summary", "")
            materials = payload.get("materials", "")
            description = payload.get("description", "")
            comments = payload.get("comments", "")
            image_urls = payload.get("image_urls", [])
            vision_analysis = payload.get("vision_analysis", "")

            deal_class = _classify_deal_profile(payload)
            deal_id = payload.get("deal_id") or ""
            deal_tag = f" #{deal_id}" if deal_id else ""
            lines = [f"[Кейс{deal_tag} ({deal_class}) — НЕ для расчёта цены] {title}" + (f" ({direction})" if direction else "")]
            if deal_class == "РЕМОНТ/ДЕМОНТАЖ":
                lines.append("  РЕМОНТНАЯ СДЕЛКА — цена не применима к производству новых изделий")
            price_parts = []
            if line_total:
                price_parts.append(f"Стоимость всей сделки: {line_total:.0f} руб")
            if duration is not None:
                price_parts.append(f"Срок: {duration} дней")
            if price_parts:
                lines.append(f"  {' | '.join(price_parts)}")
            if component_summary:
                lines.append(f"  Состав: {component_summary[:500]}")
            if materials:
                lines.append(f"  Материалы: {materials}")
            if comments:
                lines.append(f"  Комментарий: {comments[:200]}")
            if vision_analysis:
                lines.append(f"  Анализ по фото (Gemini): {vision_analysis}")
            if image_urls:
                lines.append(f"  Прямые ссылки на фото проекта: {', '.join(image_urls)}")
            bitrix_line = format_deal_link_line(deal_id, label="Bitrix")
            if bitrix_line:
                lines.append(bitrix_line)
            blocks.append("\n".join(lines))

        elif doc_type == "offer_profile":
            title = payload.get("title", "?")
            direction = payload.get("direction", "")
            line_total = payload.get("line_total")
            component_summary = payload.get("component_summary", "")
            sample_products = payload.get("sample_products", "")
            materials = payload.get("materials", "")

            # P10.6 D2: offer_id читается напрямую из payload (B4 добавил это поле).
            # Регекс-хак из P10.5-II.5 удалён — title-парсинг возвращал кривые ID,
            # если КП # не был в начале строки. deal_id остаётся для совместимости
            # с регексом в legacy docs и если offer_id ещё не проиндексирован.
            deal_id = payload.get("deal_id") or ""
            offer_id = payload.get("offer_id")
            offer_id_str = str(offer_id) if offer_id else ""
            ref_parts = []
            if offer_id_str:
                ref_parts.append(f"КП #{offer_id_str}")
            if deal_id and str(deal_id) != offer_id_str:
                ref_parts.append(f"сделка #{deal_id}")
            ref_tag = (" " + " / ".join(ref_parts)) if ref_parts else ""
            lines = [f"[Шаблон КП{ref_tag}] {title}" + (f" ({direction})" if direction else "")]
            if line_total:
                lines.append(f"  Сумма КП: {line_total:.0f} руб")
            if component_summary:
                lines.append(f"  Состав: {component_summary[:500]}")
            if sample_products:
                lines.append(f"  Позиции КП: {sample_products[:600]}")
            if materials:
                lines.append(f"  Материалы: {materials}")
            bitrix_link_id = offer_id_str or deal_id
            bitrix_line = format_deal_link_line(bitrix_link_id, label="Bitrix")
            if bitrix_line:
                lines.append(bitrix_line)
            blocks.append("\n".join(lines))

        elif doc_type == "timeline_fact":
            group_type = payload.get("group_type", "")
            group_key = payload.get("group_key", "")
            d_p50 = payload.get("duration_p50")
            d_p25 = payload.get("duration_p25")
            d_p75 = payload.get("duration_p75")
            deal_count = payload.get("deal_count", 0)
            sample_title = payload.get("sample_title", "")

            label = group_key if group_type == "direction" else (sample_title or group_key)
            lines = [f"[Сроки: {label}]"]
            parts = []
            if d_p50 is not None:
                parts.append(f"Медиана: {d_p50} дней")
            if d_p25 is not None and d_p75 is not None:
                parts.append(f"Диапазон: {d_p25}–{d_p75} дней")
            if deal_count:
                parts.append(f"Заказов: {deal_count}")
            if parts:
                lines.append(f"  {' | '.join(parts)}")
            blocks.append("\n".join(lines))

        elif doc_type == "knowledge":
            source_label = payload.get("source_label", "")
            section = payload.get("section", "")
            content = payload.get("content", payload.get("searchable_text", ""))
            # P12.3.C — surface manager-script macro type in the header so the
            # LLM treats these blocks as a canonical script, not trivia.
            is_macro = payload.get("is_macro") is True
            macro_type = payload.get("macro_type", "") or ""
            macro_tag = ""
            if is_macro:
                if macro_type:
                    macro_tag = f"Макро-скрипт: {macro_type} — "
                else:
                    macro_tag = "Макро-скрипт — "
            header = f"[{macro_tag}{source_label}: {section}]" if section and section != source_label else f"[{macro_tag}{source_label}]"
            lines = [header, f"  {content[:2000]}"]
            blocks.append("\n".join(lines))

        elif doc_type == "roadmap":
            # P11-R2: структурированный формат — LLM понимает что это процесс, не товар
            roadmap_title = payload.get("roadmap_title", "")
            section = payload.get("section", "")
            direction = payload.get("direction", "")
            service = payload.get("service", "")
            timelines = payload.get("timelines", [])
            prices = payload.get("prices", [])
            content = payload.get("searchable_text", "")

            header_parts = [f"[Регламент: {roadmap_title}"]
            if section:
                header_parts.append(f"| {section}")
            header_parts.append("]")
            header = " ".join(header_parts)

            meta_parts = []
            if direction:
                meta_parts.append(f"Направление: {direction}")
            if service:
                meta_parts.append(f"Услуга: {service}")
            if timelines:
                meta_parts.append(f"Сроки: {', '.join(str(t) for t in timelines[:3])}")
            if prices:
                meta_parts.append(f"Ценовые ориентиры: {', '.join(str(p) for p in prices[:3])}")

            lines = [header]
            if meta_parts:
                lines.append("  " + " | ".join(meta_parts))
            lines.append(f"  {content[:2000]}")
            blocks.append("\n".join(lines))

        elif doc_type == "photo_analysis":
            # searchable_text is already well-structured with header, composition, analysis
            content = payload.get("searchable_text", "")
            blocks.append(content[:2500])

        elif doc_type == "service_pricing_bridge":
            service = payload.get("service", "?")
            direction = payload.get("direction", "")
            packages = payload.get("packages", [])
            roi = payload.get("roi_anchor", "")
            prepayment = payload.get("prepayment", "")
            questions = payload.get("clarification_questions", [])
            extras = payload.get("extra_services", [])

            lines = [f"[Пакеты «{service}»]" + (f" ({direction})" if direction else "")]
            for pkg in packages:
                pmin = pkg.get("price_min", 0)
                pmax = pkg.get("price_max", 0)
                desc = pkg.get("description", "")
                lines.append(f"  • {pkg.get('name','?')}: {pmin:,} – {pmax:,} ₽" + (f" — {desc}" if desc else ""))
                products = pkg.get("products", [])
                if products:
                    lines.append(f"    Состав: {', '.join(products[:6])}")
                # P10.5-II.2: реальные offer_id с этим пакетом (G2)
                offer_ids = pkg.get("offer_ids") or []
                if offer_ids:
                    ids_str = ", ".join(f"#{i}" for i in offer_ids[:10])
                    lines.append(f"    Реальные КП: {ids_str}")
                    from app.utils.bitrix import collect_deal_urls
                    _urls = collect_deal_urls(offer_ids[:5])
                    if _urls:
                        lines.append(f"    Bitrix-ссылки: {', '.join(_urls)}")
                # P10.5-II.2: product_id'ы, если зарезолвлены на этапе ingest (G3).
                # goods.csv использует PRODUCT_ID как первичный ключ каталога.
                catalog_refs = pkg.get("product_catalog_refs") or []
                if catalog_refs:
                    resolved = [
                        f"{p.get('name','?')}→PRODUCT_ID={p.get('product_id')}"
                        f"{' ('+format(p.get('price',0),',')+' ₽)' if p.get('price') else ''}"
                        for p in catalog_refs if p.get("product_id")
                    ]
                    if resolved:
                        lines.append(f"    Каталог: {'; '.join(resolved[:6])}")
            if roi:
                lines.append(f"  ROI: {roi}")
            if prepayment:
                lines.append(f"  Оплата: {prepayment}")
            if questions:
                lines.append(f"  Уточняющие вопросы: {'; '.join(questions)}")
            if extras:
                lines.append(f"  Доп. услуги: {'; '.join(extras)}")
            blocks.append("\n".join(lines))

        elif doc_type == "offer_composition":
            title = payload.get("title", "?")
            direction = payload.get("direction", "")
            tier = payload.get("package_tier", "")
            total = payload.get("total_price", 0)
            products = payload.get("products", [])
            offer_id = payload.get("offer_id", "")

            tier_tag = f" ({tier})" if tier else ""
            lines = [f"[КП #{offer_id}{tier_tag}] {title}" + (f" ({direction})" if direction else "")]
            lines.append(f"  Итого: {total:,.0f} ₽")
            for p in products[:10]:
                qty = p.get("qty", 1)
                price = p.get("price", 0)
                name = p.get("name", "?")
                pid = p.get("product_id") or p.get("good_id") or ""
                id_prefix = f"[{pid}] " if pid else ""
                line = f"  • {id_prefix}{name}"
                if qty != 1:
                    line += f" × {qty:g}"
                line += f" — {price:,.0f} ₽"
                lines.append(line)
            bitrix_line = format_deal_link_line(offer_id, label="Bitrix")
            if bitrix_line:
                lines.append(bitrix_line)
            blocks.append("\n".join(lines))

        elif doc_type == "historical_deal":
            # P13.3 / T7: anonymized closed-deal anchor for pricing intents.
            # No PII (company/contact never in payload).
            deal_id = payload.get("deal_id", "")
            year = payload.get("year", "")
            total = payload.get("total_price", 0)
            bucket = payload.get("price_bucket", "")
            sections = payload.get("signature_sections", []) or []
            directions_blk = payload.get("directions", []) or []
            items = payload.get("items", []) or []

            head = f"[Закрытая сделка #{deal_id}"
            if year:
                head += f" / {year}"
            head += f" — {bucket}]"
            lines = [head]
            if sections:
                lines.append(f"  Категории: {', '.join(sections[:4])}")
            if directions_blk:
                lines.append(f"  Направления: {', '.join(directions_blk[:3])}")
            lines.append(f"  Сумма сделки: {float(total):,.0f} ₽")
            for it in items[:6]:
                name = (it.get("product_name") or "").strip()
                if not name:
                    continue
                qty = it.get("quantity") or 0
                price = it.get("price") or 0
                line = f"  • {name}"
                if qty:
                    line += f" × {qty:g}"
                if price:
                    line += f" @ {float(price):,.2f} ₽"
                lines.append(line)
            bitrix_line = format_deal_link_line(deal_id, label="Bitrix")
            if bitrix_line:
                lines.append(bitrix_line)
            blocks.append("\n".join(lines))

    return "\n---\n".join(blocks)


class DeepseekGenerator:
    """LLM text generation via Deepseek API."""

    def __init__(self, settings):
        self.settings = settings
        self._client = None
        self._prompts = None

    def load(self):
        """Initialize Deepseek API client."""
        if self._client is not None:
            return

        from openai import AsyncOpenAI
        # P9: bridge-документы делают контекст длиннее → LLM дольше генерирует.
        # Дефолт AsyncOpenAI = 60s приводил к "Request timed out" на логотипных
        # запросах. Поднято до 120s.
        self._client = AsyncOpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            timeout=120.0,
        )
        self._prompts = _load_prompts(self.settings)
        logger.info("Deepseek generator initialized",
                    model=self.settings.deepseek_model,
                    base_url=self.settings.deepseek_base_url)

    def _build_history_text(self, history: list) -> str:
        """Format chat history for prompt injection."""
        if not history:
            return "(нет предыдущих сообщений)"
        lines = []
        for msg in history[-6:]:  # last 6 turns max
            role_label = "Клиент" if msg.role == "user" else "Менеджер"
            lines.append(f"{role_label}: {msg.content[:400]}")
        return "\n".join(lines)

    def _build_messages(self, system: str, history: list, user_prompt: str) -> list[dict]:
        """Build OpenAI-compatible messages list with history."""
        messages = [{"role": "system", "content": system}]
        # Inject last 6 history turns as actual message objects
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content[:800]})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def generate(self, query: str, docs: list[dict], pricing_resolution=None, history: list = None, extra_context: str = "", intent_instruction: str = "") -> str:
        """Generate a human-readable response."""
        if self._client is None:
            self.load()

        context = _format_context_block(docs, pricing_resolution)
        history = history or []

        pricing_note = ""
        if pricing_resolution:
            pr = pricing_resolution
            if pr.is_financial_modifier:
                pricing_note = "\nПримечание: запрос касается финансового модификатора, а не товара/услуги."
            elif pr.total_under_key_min is not None:
                # Pre-calculated under-key total available — use it as the primary price signal
                pricing_note = f"\nИтого под ключ (расчёт): {pr.total_under_key_min:.0f}–{pr.total_under_key_max:.0f} руб"
                pricing_note += f" — {pr.confidence}"
            elif pr.estimated_value:
                pricing_note = f"\nОценка стоимости: {pr.estimated_value:.0f} руб"
                if pr.price_band_min and pr.price_band_max:
                    pricing_note += f" (диапазон: {pr.price_band_min:.0f}–{pr.price_band_max:.0f} руб)"
                pricing_note += f" — {pr.confidence}"

        full_context = (extra_context + "\n\n---\n" if extra_context else "") + context + pricing_note
        # Only inject history as text if no history message objects will be added
        history_text = self._build_history_text(history) if not history else "(см. историю выше)"
        user_prompt = self._prompts.get("human_query", "").format(
            query=query,
            context=full_context,
            history=history_text,
        )
        if intent_instruction:
            user_prompt = intent_instruction + "\n\n" + user_prompt

        try:
            response = await self._client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=self._build_messages(
                    self._prompts.get("system", ""), history, user_prompt
                ),
                temperature=0.15,
                max_tokens=1000,
            )
            return _scrub_forbidden_phrases(response.choices[0].message.content or "")
        except Exception as e:
            logger.error("Deepseek API error", error=str(e))
            raise

    async def generate_structured(self, query: str, docs: list[dict], pricing_resolution=None, extra_context: str = "", history: list = None, intent_instruction: str = "") -> dict:
        """Generate structured JSON response."""
        if self._client is None:
            self.load()

        history = history or []
        context = _format_context_block(docs, pricing_resolution)
        if extra_context:
            context = extra_context + "\n\n---\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ (найденные товары):\n" + context

        history_text = self._build_history_text(history) if not history else "(см. историю выше)"
        user_prompt = self._prompts.get("structured_query", "").format(
            query=query,
            context=context,
            history=history_text,
        )
        if intent_instruction:
            user_prompt = intent_instruction + "\n\n" + user_prompt

        try:
            response = await self._client.chat.completions.create(
                model=self.settings.deepseek_model,
                response_format={"type": "json_object"},
                messages=self._build_messages(
                    self._prompts.get("system", ""), history, user_prompt
                ),
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or "{}"
            return _scrub_structured_response(json.loads(content))
        except json.JSONDecodeError as e:
            logger.error("JSON decode error", error=str(e))
            # Fallback: try to extract JSON from the response
            return _scrub_structured_response(self._extract_json_fallback(content if "content" in dir() else "{}"))
        except Exception as e:
            logger.error("Deepseek structured API error", error=str(e))
            raise

    async def generate_deal_estimate(
        self,
        query: str,
        docs: list[dict],
        pricing_resolution=None,
        extra_context: str = "",
        history: list = None,
        intent_instruction: str = "",
    ) -> dict:
        """Generate a structured JSON response with deal_items for Bitrix24 estimate."""
        if self._client is None:
            self.load()

        history = history or []
        context = _format_context_block(docs, pricing_resolution)
        if extra_context:
            context = extra_context + "\n\n---\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ (найденные товары):\n" + context

        # Append deal_estimate_suffix to structured_query prompt
        base_prompt = self._prompts.get("structured_query", "")
        deal_suffix = self._prompts.get("deal_estimate_suffix", "")
        combined_prompt = base_prompt + ("\n\n" + deal_suffix if deal_suffix else "")

        history_text = self._build_history_text(history) if not history else "(см. историю выше)"
        user_prompt = combined_prompt.format(
            query=query,
            context=context,
            history=history_text,
        )
        if intent_instruction:
            user_prompt = intent_instruction + "\n\n" + user_prompt

        try:
            response = await self._client.chat.completions.create(
                model=self.settings.deepseek_model,
                response_format={"type": "json_object"},
                messages=self._build_messages(
                    self._prompts.get("system", ""), history, user_prompt
                ),
                temperature=0.1,
                max_tokens=2000,  # deal_items can be long
            )
            content = response.choices[0].message.content or "{}"
            return _scrub_structured_response(json.loads(content))
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in deal estimate", error=str(e))
            return _scrub_structured_response(self._extract_json_fallback(content if "content" in dir() else "{}"))
        except Exception as e:
            logger.error("Deepseek deal estimate API error", error=str(e))
            raise

    def _extract_json_fallback(self, text: str) -> dict:
        """Try to extract JSON from malformed response."""
        # Find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {
            "summary": "Не удалось сформировать структурированный ответ.",
            "confidence": "manual",
            "flags": ["structured_output_failed"],
            "suggested_bundle": [],
            "estimated_price": None,
            "price_band": {"min": None, "max": None, "currency": "RUB"},
            "reasoning": "",
            "risks": [],
            "references": [],
            "source_distinction": {"has_order_data": False, "has_offer_data": False, "dataset_type": "unknown"},
        }
