"""
Intent classifier: hybrid Tier-1 regex + Tier-2 embedding prototype matching.

Replaces 15 scattered regex gate functions in query.py with a single
unified classification pipeline. Uses BGE-M3 embeddings (already loaded
in HybridRetriever) for semantic prototype matching.
"""
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

try:
    import numpy as np
except ImportError:
    np = None

logger = get_logger(__name__)

INTENT_NAMES = (
    "product_query", "smeta_request", "bundle_query", "consultation",
    "describe", "underspec", "out_of_scope", "financial_modifier",
    "visualization", "referential", "empty_context_smeta", "general",
)

PROTOTYPES_PATH = Path(__file__).parent.parent.parent / "configs" / "intent_prototypes.yaml"


@dataclass
class IntentResult:
    intent: str = "general"
    confidence: float = 0.0
    method: str = "default"          # "regex" | "embedding" | "default"
    hints: dict = field(default_factory=dict)

    @property
    def is_pricing_intent(self) -> bool:
        return self.intent in ("smeta_request", "product_query", "bundle_query")

    @property
    def needs_clarification(self) -> bool:
        return self.intent in ("underspec", "referential", "empty_context_smeta")

    @property
    def is_no_price(self) -> bool:
        return self.intent in (
            "consultation", "describe", "out_of_scope",
            "financial_modifier", "visualization", "referential",
            "empty_context_smeta",
        )


# ---------------------------------------------------------------------------
# Tier 1: high-confidence regex patterns (>95% precision)
# ---------------------------------------------------------------------------

_TIER1_RULES: list[tuple[str, Any, dict | None]] = []


def _rx(pattern: str, flags=re.IGNORECASE):
    return re.compile(pattern, flags)


def _build_tier1():
    """Build Tier-1 regex rules. Called once at module load."""
    rules = []

    # consultation: "Вы делаете X?" without price keywords
    _consultation_rx = [
        _rx(r"^\s*(а\s+)?(вы|Вы)\s+(делает|можете|умеете|работает|оказыва|выполня|принима|берёт|изготавлива|производит)\w*\s+"),
        _rx(r"^\s*(а\s+)?(у вас|У вас)\s+(есть|можно|бывает|имеется)"),
        _rx(r"^\s*(вы|а вы)\s+(делает|можете)\w*\s+\w+\s*\??\s*$"),
    ]
    _price_kw_rx = _rx(r"стоимост|стоит|цен[аеуы]|расценк|прайс|смет|бюджет|сколько")

    def _check_consultation(q: str) -> IntentResult | None:
        if any(rx.search(q) for rx in _consultation_rx):
            if not _price_kw_rx.search(q):
                return IntentResult("consultation", 0.95, "regex")
        return None

    rules.append(_check_consultation)

    # out_of_scope: work hours, address, contacts
    _oos_patterns = [
        _rx(r"режим\s+работы|график\s+работы|часы\s+работы|время\s+работы"),
        _rx(r"когда\s+(откры|закры|работае)|во\s+сколько\s+(откры|закры)"),
        _rx(r"расписани[еяю]|выходны[еыхй]\s+(дн|ли)"),
        _rx(r"как\s+(добрать|проехать|найти)|где\s+(вы\s+)?(находит|расположен)"),
        _rx(r"адрес\s+офиса|адрес\s+компани|\bваш\s+адрес"),
        _rx(r"номер\s+телефон|контактн.*телефон|\bтелефон\s+офис"),
    ]

    def _check_oos(q: str) -> IntentResult | None:
        if any(rx.search(q) for rx in _oos_patterns):
            return IntentResult("out_of_scope", 0.95, "regex")
        return None

    rules.append(_check_oos)

    # financial_modifier: безнал, скидка, НДС
    _fin_patterns = [
        _rx(r"безнал\w*"),
        _rx(r"\bналичн\w*|\bналичк\w*"),
        _rx(r"скидк\w*\s*\d+\s*%|скидк\w*.*стоимост|\d+\s*%\s*скидк"),
        _rx(r"надбавк\w*|наценк\w*"),
        _rx(r"\bндс\b|без\s*ндс|с\s*ндс"),
        _rx(r"предоплат\w*\s*\d+\s*%|рассрочк\w*"),
    ]

    def _check_financial(q: str) -> IntentResult | None:
        if any(rx.search(q) for rx in _fin_patterns):
            return IntentResult("financial_modifier", 0.95, "regex")
        return None

    rules.append(_check_financial)

    # visualization: "покажи как будет на фасаде"
    _viz_rx = _rx(
        r"как\s+(это\s+)?(будет\s+)?(выглядит|выглядеть|смотрит|смотреть)\w*\s+(на\s+фасад|на\s+здани|на\s+стен)"
        r"|как\s+(это\s+)?будет\s+на\s+фасад"
        r"|визуализаци|покаж\w*\s+как"
        r"|увидеть\s+как\s+(это|будет)|фото\s*ш?оп|3d\s*модел"
    )

    def _check_viz(q: str) -> IntentResult | None:
        if _viz_rx.search(q):
            return IntentResult("visualization", 0.90, "regex")
        return None

    rules.append(_check_viz)

    # empty_context_smeta: "дай смету на эту услугу" (no product noun)
    # MUST be checked BEFORE referential — "смету на эту услугу" triggers
    # both patterns, but empty_context_smeta is more specific.
    _empty_smeta_rx = [
        _rx(r"(дай|составь|напиши|сделай|подготов|сформируй|нужн[аы]?)\s+смет\w*\s+(на|для)\s+эт"),
        _rx(r"осмет\w*\s+эт"),
        _rx(r"смет\w*\s+(для|на)\s+(сделк|услуг|позици|клиент|заявк|этого|этой|этому)"),
        _rx(r"^\s*(дай|напиши)\s+смет\w*\s*$"),
    ]
    _product_nouns = (
        "логотип", "лого", "вывеск", "баннер", "листовк", "визитк", "наклейк",
        "буклет", "флаер", "брендбук", "брендинг", "штендер", "панель", "кронштейн",
        "табличк", "реклам", "печат", "монтаж", "дизайн", r"объ[её]мн", "букв",
        "светов", "фасад", "витрин", "стикер", "этикет", "упаковк", "стенд",
        "постер", r"афиш", "плакат", r"рол[а-я]п", "шаурм", "кофейн", "ресторан",
        "магазин", "аптек", "офис", "стиль", "айдентик", "компани",
        "концепци", "3d", "3-d", "навигаци", "меню", "календар",
    )

    def _check_empty_smeta(q: str) -> IntentResult | None:
        if not any(rx.search(q) for rx in _empty_smeta_rx):
            return None
        q_lower = q.lower()
        for noun in _product_nouns:
            if re.search(noun, q_lower):
                return None
        return IntentResult("empty_context_smeta", 0.95, "regex")

    rules.append(_check_empty_smeta)

    # referential: "вместо этой / для этого объекта"
    _ref_patterns = [
        _rx(r"вместо\s+(эт\w+|этой|этого|этих)"),
        _rx(r"\bэт[ау]\s+(услуг|позици|вывеск|букв|баннер|табличк|сделк)"),
        _rx(r"для\s+эт\w+\s+(объект|клиент|сделк|проект)"),
        _rx(r"как\s+в\s+прошл\w+\s+(раз|сделк|заказ)"),
    ]

    def _check_referential(q: str) -> IntentResult | None:
        if any(rx.search(q) for rx in _ref_patterns):
            return IntentResult("referential", 0.90, "regex")
        return None

    rules.append(_check_referential)

    # describe: manager script / text generation (not pricing)
    _describe_keywords = [
        "помоги составить описание", "составить описание", "помоги с описанием",
        "как ответить", "что сказать", "что ему ответить", "что ответить",
        "как презентовать", "презентуй", "скрипт", "аргумент",
        "подготовить текст", "подготовь текст", "подготовить ответ", "подготовь ответ",
        "сформулируй", "сформулировать",
        "первого ответа", "первый ответ", "ответа клиенту", "ответ клиенту",
        "клиент спрашивает", "клиент ответил", "клиент сомневается",
        "клиент пишет", "клиент написал", "клиент хочет",
    ]

    def _check_describe(q: str) -> IntentResult | None:
        q_lower = q.lower()
        if any(kw in q_lower for kw in _describe_keywords):
            return IntentResult("describe", 0.90, "regex")
        return None

    rules.append(_check_describe)

    # bundle: "под ключ / комплект / пакет услуг"
    _bundle_patterns = [
        _rx(r"под\s+ключ"),
        _rx(r"что\s+(входит|включ|идёт|идет)"),
        _rx(r"из\s+чего\s+состо|какой\s+состав|состав\s+и\s+стоимост"),
        _rx(r"\bкомплект\w*|\bкомплектац"),
        _rx(r"\bпакет\s+(услуг|рекламн|для|на)|\bпакет\w*\s+для"),
        _rx(r"\bвесь\s+пакет|\bполный\s+пакет"),
        _rx(r"\bнабор\s+(рекламн|материал|услуг|для|монтаж)|\bполный\s+набор"),
        _rx(r"комплексн\w*\s+(печатн|рекламн|кампани|проект)"),
        _rx(r"брендирован\w*|брендинг"),
        _rx(r"всё\s+для\s+открыт|все\s+для\s+открыт|для\s+открыт\w*\s+(кафе|магазин|аптек|ресторан|офис|точк)"),
    ]

    def _check_bundle(q: str) -> IntentResult | None:
        if any(rx.search(q) for rx in _bundle_patterns):
            return IntentResult("bundle_query", 0.85, "regex")
        return None

    rules.append(_check_bundle)

    # product_query: brand-design keywords (logo/identity/brandbook).
    # Anchors these to product_query regardless of embedding neighbourhood.
    # Must come AFTER consultation/describe/bundle so those specialised
    # intents win when applicable.
    _brand_design_rx = _rx(
        r"\b(логотип\w*|\bлого\b|фирменн\w+\s+стил\w*|брендбук\w*|айдентик\w*)\b"
    )

    def _check_brand_design(q: str) -> IntentResult | None:
        if _brand_design_rx.search(q):
            return IntentResult("product_query", 0.95, "regex")
        return None

    rules.append(_check_brand_design)

    return rules


_TIER1_RULES = _build_tier1()


# ---------------------------------------------------------------------------
# Tier 2: embedding prototype matching
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Hybrid intent classifier: Tier 1 regex → Tier 2 BGE-M3 prototypes."""

    def __init__(self):
        self._prototypes: dict[str, list[str]] = {}
        self._proto_embeddings: dict[str, np.ndarray] = {}
        self._embed_fn = None
        self._ready = False

    def load(self, embed_fn, prototypes_path: Path | None = None):
        """Load prototypes and pre-compute embeddings.

        Args:
            embed_fn: callable(str) -> list[float], e.g. retriever.embed_query
            prototypes_path: path to intent_prototypes.yaml
        """
        self._embed_fn = embed_fn
        path = prototypes_path or PROTOTYPES_PATH

        if not path.exists():
            logger.warning("Intent prototypes not found", path=str(path))
            self._ready = False
            return self

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._prototypes = {k: v for k, v in raw.items() if isinstance(v, list)}

        # Pre-compute embeddings for all prototypes
        if np is None:
            logger.warning("numpy not available — Tier 2 embedding disabled")
            self._ready = True
            return self
        for intent, queries in self._prototypes.items():
            embeddings = []
            for q in queries:
                vec = embed_fn(q)
                embeddings.append(np.array(vec, dtype=np.float32))
            if embeddings:
                self._proto_embeddings[intent] = np.stack(embeddings)

        total = sum(len(v) for v in self._prototypes.values())
        logger.info("Intent classifier loaded",
                    intents=len(self._prototypes), prototypes=total)
        self._ready = True
        return self

    @property
    def is_ready(self) -> bool:
        return self._ready and self._embed_fn is not None

    def classify(self, query: str, has_history: bool = False) -> IntentResult:
        """Classify query intent.

        Args:
            query: raw user query
            has_history: True if chat has previous messages (affects referential gate)
        """
        if not query or not query.strip():
            return IntentResult("underspec", 0.99, "regex")

        # Tier 1: regex rules
        for rule_fn in _TIER1_RULES:
            result = rule_fn(query)
            if result is not None:
                # referential gate only fires without history
                if result.intent == "referential" and has_history:
                    continue
                logger.debug("Tier-1 regex match", intent=result.intent, query=query[:60])
                return result

        # Tier 2: embedding prototype matching
        if self.is_ready:
            result = self._tier2_embedding(query)
            if result is not None:
                logger.debug("Tier-2 embedding match", intent=result.intent,
                             confidence=round(result.confidence, 3), query=query[:60])
                return result

        return IntentResult("general", 0.3, "default")

    def _tier2_embedding(self, query: str) -> IntentResult | None:
        """Match query against prototype embeddings via cosine similarity."""
        q_vec = np.array(self._embed_fn(query), dtype=np.float32)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        best_intent = None
        best_score = 0.0

        for intent, proto_matrix in self._proto_embeddings.items():
            # proto_matrix is already L2-normalized by BGE-M3
            sims = proto_matrix @ q_norm
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        if best_intent is not None and best_score >= 0.75:
            return IntentResult(best_intent, best_score, "embedding")

        return None


# Module-level singleton
_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier | None:
    return _classifier


def init_classifier(embed_fn, prototypes_path: Path | None = None) -> IntentClassifier:
    global _classifier
    _classifier = IntentClassifier()
    _classifier.load(embed_fn, prototypes_path)
    return _classifier
