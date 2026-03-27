"""
Vision API integration for sign/facade image analysis.
Uses OpenAI-compatible API (gpt-4o-mini by default) to extract:
- Sign type (объёмные буквы, световой короб, баннер, etc.)
- Estimated dimensions
- Materials and lighting
- Construction condition and complexity
"""
from app.utils.logging import get_logger

logger = get_logger(__name__)

VISION_PROMPT = """Ты — эксперт по рекламным конструкциям и вывескам.
Проанализируй изображение и определи:

1. **Тип конструкции**: (объёмные буквы / световой короб / баннер / лайтбокс / брандмауэр / таблич­ка / штендер / другое)
2. **Примерные размеры**: высота и ширина букв или конструкции в сантиметрах (если можно оценить по контексту — дверной проём, витрина, человек рядом)
3. **Материал и подсветка**: (акрил / металл / ПВХ / пластик / неон / LED / без подсветки)
4. **Текст/логотип**: что написано или изображено (если видно)
5. **Состояние**: (новое / хорошее / требует замены / повреждено)
6. **Сложность монтажа**: (простой / средний / сложный — фасад, высота, крепление)
7. **Рекомендации**: что можно предложить клиенту исходя из увиденного

Отвечай на русском языке. Будь конкретным. Если что-то невозможно определить — так и скажи.
"""

BATCH_VISION_PROMPT = """Ты — эксперт по рекламным конструкциям. Это фото выполненной работы рекламной компании.
Опиши ПОДРОБНО для внутренней базы знаний:

1. **Тип**: (объёмные буквы / световой короб / баннер / лайтбокс / табличка / штендер / крышная установка / другое)
2. **Подтип подсветки**: (лицевая LED / контражурная / открытый неон / без подсветки / комбинированная)
3. **Размеры**: оценка в см (высота букв, ширина конструкции)
4. **Материалы**: (ПВХ / акрил / композит / нержавейка / оцинковка / пластик)
5. **Текст на вывеске**: что написано
6. **Место установки**: (фасад / входная группа / крыша / витрина / внутри помещения)
7. **Сложность**: (простой / средний / сложный) и почему
8. **Визуальное качество**: оценка работы (аккуратность, ровность, светопередача)
9. **Ключевые слова для поиска**: 5-10 тегов через запятую

Отвечай на русском. Будь конкретным и детальным.
"""


class VisionAnalyzer:
    """Analyze sign/facade images via vision API."""

    def __init__(self, settings):
        self.settings = settings
        self._client = None
        self._available = False

    def load(self):
        if self._client is not None:
            return

        api_key = self.settings.vision_api_key
        if not api_key:
            logger.info("Vision API key not configured — image analysis disabled")
            self._available = False
            return

        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.settings.vision_base_url,
            )
            self._available = True
            logger.info("Vision analyzer initialized",
                        model=self.settings.vision_model,
                        base_url=self.settings.vision_base_url)
        except Exception as e:
            logger.warning("Vision API client failed to initialize", error=str(e))
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    async def analyze(self, image_base64: str, mime_type: str = "image/jpeg") -> str:
        """
        Analyze an image and return a Russian-language description of the sign/construction.

        Args:
            image_base64: Base64-encoded image data (without data URI prefix)
            mime_type: Image MIME type (image/jpeg, image/png, image/webp)

        Returns:
            Russian text description of what was found in the image.
            Empty string if analysis fails or is unavailable.
        """
        if not self._available or self._client is None:
            return ""

        # Limit base64 size — roughly 4MB decoded = ~5.3MB encoded
        if len(image_base64) > 5_500_000:
            logger.warning("Image too large for vision API, skipping", size=len(image_base64))
            return ""

        data_url = f"data:{mime_type};base64,{image_base64}"

        try:
            response = await self._client.chat.completions.create(
                model=self.settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": VISION_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            result = response.choices[0].message.content or ""
            logger.info("Vision analysis completed", chars=len(result))
            return result
        except Exception as e:
            logger.error("Vision API error", error=str(e))
            return ""


class SyncVisionAnalyzer:
    """Synchronous vision analyzer for batch processing in ingest scripts."""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def analyze(self, image_base64: str, mime_type: str = "image/jpeg",
                prompt: str | None = None) -> str:
        if prompt is None:
            prompt = BATCH_VISION_PROMPT

        if len(image_base64) > 5_500_000:
            return ""

        data_url = f"data:{mime_type};base64,{image_base64}"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"  Vision API error: {e}")
            return ""
