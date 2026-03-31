#!/usr/bin/env python3
"""
Vision Analysis Script for SALES_RAG.
Downloads image URLs from deal profiles and uses Vision API (OpenAI-compatible)
to extract technical details, pricing hints, and unique value propositions for RAG embeddings.

Usage:
    python scripts/vision_analysis.py [--limit N]
"""
import json
import base64
import requests
import click
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
DEALS_JSON_PATH = PROJECT_ROOT.parent / "RAG_DATA" / "deals.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "photo_analysis_docs.jsonl"
ENV_FILE = PROJECT_ROOT / "configs" / ".env"


def load_env() -> dict[str, str]:
    """Load .env file into a dict."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


VISION_PROMPT = """Выступай в роли главного маркетолога и технолога рекламно-производственной компании Labus. У тебя перед глазами фотографии шаблонного коммерческого предложения / типового комплекта (ID: {deal_id}, Название: {title}).

Это НЕ индивидуальный завершенный проект, а базовый коммерческий продукт или собранный комплект (Offer), который мы предлагаем клиентам как основу.

Твоя задача — сделать глубокий визуально-технический разбор этого комплекта для базы знаний менеджеров.

Шаг 1. Классифицируй продукт (опираясь на фото и название). Это может быть:
- Наружная и интерьерная реклама (вывески, буквы, таблички)
- Широкоформатная печать (баннеры, пленка, самоклейка)
- Полиграфия (визитки, листовки, буклеты, каталоги, стикеры)
- Сувениры и Мерч (ручки, кружки, футболки, шопперы, магниты, шоколад)

Шаг 2. Выполни технический разбор в зависимости от типа:
- Для наружной рекламы: типы применяемой подсветки, рекомендуемые материалы (акрил, ПВХ, металл).
- Для полиграфии и мерча: рекомендуемый метод нанесения (УФ, сублимация, шелкография, лазер, термотрансфер, тампопечать), материалы (тип бумаги, плотность, пластик, ткань).

Шаг 3. Ценность: Укажи 1-2 главных эстетических или качественных "фишки" этого типового комплекта. Чем он хорош в базовом виде? Как менеджер должен презентовать его надежность или премиальность клиенту?

Ответ дай в 1-2 емких абзаца без длинных вступлений."""


def download_image_as_base64(url: str) -> tuple[str, str] | None:
    """Download image from URL and return (base64_data, mime_type) or None."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": "https://labus.pro/",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 1000:
            mime = resp.headers.get("Content-Type", "image/jpeg")
            b64 = base64.b64encode(resp.content).decode("utf-8")
            if len(b64) > 5_500_000:
                return None
            return b64, mime
        else:
            print(f"[DEBUG] Ошибка скачивания {url}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[DEBUG] Ошибка соединения для {url}: {e}")
    return None


API_CLIENTS = []

def get_next_client():
    global API_CLIENTS
    available = [c for c in API_CLIENTS if not c["is_dead"] and c["cooldown_until"] < time.time()]
    
    if not available:
        alive = [c for c in API_CLIENTS if not c["is_dead"]]
        if not alive:
            return None # All keys are completely dead (daily limit reached)
        
        sleep_time = min(c["cooldown_until"] for c in alive) - time.time()
        if sleep_time > 0:
            active_keys = len(alive)
            print(f"\n[!] All {active_keys} active keys hit RPM limit. Sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        return get_next_client()
        
    available.sort(key=lambda c: c.get("last_used", 0))
    selected = available[0]
    selected["last_used"] = time.time()
    return selected

def analyze_deal_images(model: str, urls: list[str],
                        deal_id: str, title: str) -> str | None:
    """Analyze up to 3 images for a deal using vision API (with smart load balancing)."""
    content = []
    prompt_text = VISION_PROMPT.format(deal_id=deal_id, title=title)
    content.append({"type": "text", "text": prompt_text})

    images_loaded = 0
    for url in urls[:3]:
        result = download_image_as_base64(url)
        if result:
            b64, mime = result
            data_url = f"data:{mime};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            images_loaded += 1

    if images_loaded == 0:
        return None

    max_attempts = len(API_CLIENTS) * 4  # Try enough times assuming keys recover
    for attempt in range(max_attempts):
        client_dict = get_next_client()
        if not client_dict:
            print(f"\nFATAL: All API keys have exhausted their DAILY quotas! Stop the script.")
            return None
            
        try:
            response = client_dict["client"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content or None
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                if "PerDay" in error_str or "limit: 20" in error_str or "limit: 50" in error_str or "limit: 1500" in error_str or "RESOURCE_EXHAUSTED" in error_str and "retryDelay" not in error_str:
                    print(f"\n[!] Key {client_dict['key_id']} exhausted DAILY quota. Removing from rotation.")
                    client_dict["is_dead"] = True
                else:
                    print(f"\n[!] Key {client_dict['key_id']} hit RPM rate limit (429). Cooldown 60s.")
                    client_dict["cooldown_until"] = time.time() + 60.0
                continue
                
            if "503" in error_str or "500" in error_str or "timeout" in error_str.lower():
                print(f"\n[!] Server error on Key {client_dict['key_id']}. Cooldown 15s.")
                client_dict["cooldown_until"] = time.time() + 15.0
                continue
            
            # Фатальные ошибки (не 429) не ретраятся для той же сделки
            print(f"\nVision API error on deal {deal_id}: {e}")
            return None
    
    print(f"\nFailed to process deal {deal_id} after {max_attempts} retries. Skipping.")
    return None


@click.command()
@click.option("--limit", default=0, type=int, help="Limit number of deals to process (0 = all)")
def main(limit):
    env = load_env()
    api_key_str = env.get("VISION_API_KEY", "")
    base_url = env.get("VISION_BASE_URL", "https://api.artemox.com/v1")
    model = env.get("VISION_MODEL", "gemini-2.0-flash")

    if not api_key_str:
        print("ERROR: VISION_API_KEY not set in configs/.env")
        return

    # Parse multiple keys separated by commas
    api_keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    if not api_keys:
        print("ERROR: No valid keys found in VISION_API_KEY")
        return

    clients_init = [OpenAI(api_key=ak, base_url=base_url) for ak in api_keys]
    
    global API_CLIENTS
    API_CLIENTS = [{"client": c, "key_id": i+1, "cooldown_until": 0, "is_dead": False, "last_used": 0} for i, c in enumerate(clients_init)]
    
    print(f"Vision API: loaded {len(API_CLIENTS)} keys / model: {model}")

    if not DEALS_JSON_PATH.exists():
        print(f"Data not found at {DEALS_JSON_PATH}. Run generateRagData.mjs first.")
        return

    print(f"Loading deals from {DEALS_JSON_PATH}...")
    with open(DEALS_JSON_PATH, "r", encoding="utf-8") as f:
        deals = json.load(f)


    # Filter deals with images
    with_images = [d for d in deals if d.get("IMAGE_URLS")]

    processed_deal_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    processed_deal_ids.add(str(doc.get("metadata", {}).get("deal_id")))
                except json.JSONDecodeError:
                    pass

    without_analysis = [d for d in with_images if str(d.get("ID")) not in processed_deal_ids]
    print(f"Total deals: {len(deals)}, with images: {len(with_images)}, need analysis: {len(without_analysis)}")

    if not without_analysis:
        print("No new images to analyze.")
        return

    to_process = without_analysis
    if limit > 0:
        to_process = to_process[:limit]

    print(f"Processing {len(to_process)} deals via Gemini...")
    processed = 0
    errors = 0
    
    for deal in tqdm(to_process, desc="Analyzing images"):
        urls = [url for url in deal.get("IMAGE_URLS", []) if isinstance(url, str)]
        deal_id = str(deal.get("ID", "Unknown"))
        title = str(deal.get("TITLE", "Unknown"))

        if urls:
            analysis = analyze_deal_images(model, urls, deal_id, title)
            if analysis:
                doc_payload = {
                    "doc_id": f"photo_vision_{deal_id}",
                    "doc_type": "photo_analysis",
                    "searchable_text": f"Анализ фото для товара: {title}\n{analysis}",
                    "metadata": {
                        "deal_id": deal_id,
                        "deal_title": title,
                        "image_urls": urls,
                        "vision_analysis": analysis
                    },
                    "provenance": {
                        "sources": urls[:3],
                        "vision_model": model,
                        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
                    }
                }
                with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(doc_payload, ensure_ascii=False) + "\n")
                processed += 1
            else:
                errors += 1
        
    print(f"Done! Analyzed deals appended to {OUTPUT_PATH}. Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    main()
