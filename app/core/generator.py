"""
LLM generation via Deepseek API (OpenAI-compatible).
Handles both human-readable and structured JSON responses.
"""
import json
import yaml
from pathlib import Path
from typing import Any
from app.utils.logging import get_logger

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
            lines = [f"[Продукт] {product_name}" + (f" ({direction})" if direction else "") + (f" [{section_name}]" if section_name else "")]
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
            lines = [f"[Набор ({completeness})] {title}" + (f" ({direction})" if direction else "")]
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
            lines = [f"[Кейс ({deal_class}) — НЕ для расчёта цены] {title}" + (f" ({direction})" if direction else "")]
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
            header = f"[{source_label}: {section}]" if section and section != source_label else f"[{source_label}]"
            lines = [header, f"  {content[:2000]}"]
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
        self._client = AsyncOpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
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

    async def generate(self, query: str, docs: list[dict], pricing_resolution=None, history: list = None, extra_context: str = "") -> str:
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

        try:
            response = await self._client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=self._build_messages(
                    self._prompts.get("system", ""), history, user_prompt
                ),
                temperature=0.15,
                max_tokens=1000,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Deepseek API error", error=str(e))
            raise

    async def generate_structured(self, query: str, docs: list[dict], pricing_resolution=None, extra_context: str = "", history: list = None) -> dict:
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
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error", error=str(e))
            # Fallback: try to extract JSON from the response
            return self._extract_json_fallback(content if "content" in dir() else "{}")
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
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in deal estimate", error=str(e))
            return self._extract_json_fallback(content if "content" in dir() else "{}")
        except Exception as e:
            logger.error("Deepseek deal estimate API error", error=str(e))
            raise

    def _extract_json_fallback(self, text: str) -> dict:
        """Try to extract JSON from malformed response."""
        import re
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
