"""Bitrix24 deal/offer URL helpers.

Sales manager needs clickable links to the original Bitrix deal for every
reference the RAG surfaces (offers.csv / goods.csv ID == Bitrix deal ID).
This module builds canonical URLs and post-processes LLM output to turn
`КП #68312`, `сделка #68312`, `#68312` mentions into markdown links.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

# Override via env BITRIX_DEAL_URL_TEMPLATE="https://host/crm/deal/details/{id}/".
# Default matches labus.bitrix24.ru portal.
DEFAULT_BITRIX_DEAL_URL_TEMPLATE = "https://labus.bitrix24.ru/crm/deal/details/{id}/"


def _template() -> str:
    # Prefer app.config (which also reads configs/.env via Pydantic Settings);
    # fall back to raw env var so pure-unit tests work without settings import.
    try:
        from app.config import settings  # local import to avoid circular
        tmpl = getattr(settings, "bitrix_deal_url_template", "") or ""
        if tmpl:
            return tmpl
    except Exception:
        pass
    return os.environ.get("BITRIX_DEAL_URL_TEMPLATE", DEFAULT_BITRIX_DEAL_URL_TEMPLATE)


def build_deal_url(deal_id) -> str | None:
    """Build Bitrix24 deal URL from deal/offer ID.

    Returns None for empty/zero/non-numeric input so callers can guard easily.
    """
    if deal_id is None:
        return None
    s = str(deal_id).strip()
    if not s or not s.isdigit() or int(s) <= 0:
        return None
    return _template().format(id=s)


# Patterns ordered from most specific (with label) to bare hashtag-id.
# Capture group "id" always holds the numeric id. The leading label (if any)
# is captured by group "label" so we can re-emit it inside the link text.
_LINK_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "КП #68312" / "кп #68312" / "КП№68312" / "КП № 68312"
    (re.compile(r"(?P<label>КП)\s*(?:#|№)\s*(?P<id>\d{3,10})", re.IGNORECASE), "КП"),
    # "сделка #68312" / "сделки #68312" / "Сделка №68312"
    (re.compile(r"(?P<label>сделк[аиуе])\s*(?:#|№)\s*(?P<id>\d{3,10})", re.IGNORECASE), "сделка"),
    # "Кейс #68312"
    (re.compile(r"(?P<label>Кейс)\s*(?:#|№)\s*(?P<id>\d{3,10})", re.IGNORECASE), "кейс"),
    # Generic "#68312" — only if bare (no word char before); lowest priority so
    # already-labelled mentions are captured above first.
    (re.compile(r"(?<![\w/])#(?P<id>\d{4,10})"), ""),
]

# Markdown link + bare URL detector — any ID inside one of these spans must
# be skipped so we don't double-wrap or wrap an id that's already a URL chunk.
_PROTECTED_SPAN = re.compile(
    r"\[[^\]]*\]\([^)]*\)|https?://\S+",
    re.IGNORECASE,
)


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    """Collect (start, end) ranges of text that must not be modified:
    existing markdown links and bare URLs.
    """
    return [(m.start(), m.end()) for m in _PROTECTED_SPAN.finditer(text)]


def _overlaps(pos: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    for a, b in ranges:
        if pos < b and end > a:
            return True
    return False


def enrich_text_with_deal_links(text: str) -> str:
    """Replace deal/offer ID mentions with markdown links to Bitrix24.

    Idempotent — running twice doesn't double-wrap. Keeps original label
    (`КП #68312` stays visible as link text).
    """
    if not text:
        return text
    if "#" not in text and "№" not in text:
        return text

    protected = _protected_ranges(text)

    # Collect candidate replacements from all patterns in a single pass, then
    # resolve overlaps by priority (earlier patterns win; longer match wins
    # when tied on start). This prevents the bare-`#id` pattern from firing
    # again inside a "сделка #id" that was already matched.
    candidates: list[tuple[int, int, str, int]] = []  # (start, end, replacement, priority)
    for priority, (pattern, label_default) in enumerate(_LINK_PATTERNS):
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if _overlaps(start, end, protected):
                continue
            deal_id = m.group("id")
            url = build_deal_url(deal_id)
            if not url:
                continue
            label = m.groupdict().get("label") or label_default
            visible = f"{label} #{deal_id}".strip() if label else f"#{deal_id}"
            replacement = f"[{visible}]({url})"
            candidates.append((start, end, replacement, priority))

    if not candidates:
        return text

    # Sort by (start, priority, -length) and drop later candidates whose span
    # overlaps an earlier accepted one.
    candidates.sort(key=lambda c: (c[0], c[3], -(c[1] - c[0])))
    accepted: list[tuple[int, int, str]] = []
    used: list[tuple[int, int]] = []
    for start, end, repl, _prio in candidates:
        if _overlaps(start, end, used):
            continue
        accepted.append((start, end, repl))
        used.append((start, end))

    accepted.sort(key=lambda c: c[0])
    out_parts: list[str] = []
    cursor = 0
    for start, end, repl in accepted:
        out_parts.append(text[cursor:start])
        out_parts.append(repl)
        cursor = end
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def format_deal_link_line(deal_id, label: str = "Bitrix") -> str | None:
    """Format a single context-line hint for the LLM.

    Example: `  Bitrix: https://labus.bitrix24.ru/crm/deal/details/68312/`
    Returns None when deal_id is empty so callers can skip cleanly.
    """
    url = build_deal_url(deal_id)
    if not url:
        return None
    return f"  {label}: {url}"


def collect_deal_urls(deal_ids: Iterable) -> list[str]:
    """Build a deduplicated list of URLs from a list of IDs."""
    seen: set[str] = set()
    out: list[str] = []
    for did in deal_ids:
        url = build_deal_url(did)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out
