#!/usr/bin/env python3
"""P13.3 regression suite — replays anchor cases from 2026-04-22 dump and
autograders responses against expected/forbidden sentinels.

Usage:
    # Local server:
    python eval.py --server http://localhost:8000

    # Prod canary:
    python eval.py --server https://62.217.178.117 --host-header ai.labus.pro --no-verify-ssl

Exit code: 0 if all cases pass, 1 if any fail.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).parent
CASES_PATH = ROOT / "cases.json"


def _flatten_text(resp: dict) -> str:
    """Concatenate every text-bearing field for sentinel matching."""
    parts = [
        resp.get("summary", ""),
        resp.get("reasoning", ""),
        " ".join(resp.get("flags") or []),
        " ".join(resp.get("risks") or []),
    ]
    for it in resp.get("deal_items") or []:
        parts.append((it.get("product_name") or "") + " " + (it.get("notes") or ""))
    pb = resp.get("parametric_breakdown") or {}
    for li in pb.get("line_items") or []:
        parts.append(li.get("label", "") + " " + li.get("product_name", ""))
    for ref in resp.get("references") or []:
        parts.append(ref.get("snippet", ""))
    return " \n ".join(p for p in parts if p)


def _doc_types(resp: dict) -> set[str]:
    return {r.get("doc_type", "") for r in (resp.get("references") or [])}


def _extract_url_count(resp: dict) -> int:
    """Count clickable Bitrix/site URLs across references + summary."""
    count = 0
    for ref in resp.get("references") or []:
        if ref.get("bitrix_url"):
            count += 1
    text = _flatten_text(resp)
    count += len(re.findall(r"https?://\S+", text))
    return count


def _extract_max_price(resp: dict) -> float | None:
    """Find the highest price mentioned anywhere in the response."""
    candidates: list[float] = []
    pb = resp.get("price_band") or {}
    if pb.get("max"):
        candidates.append(float(pb["max"]))
    if pb.get("min"):
        candidates.append(float(pb["min"]))
    ep = resp.get("estimated_price") or {}
    if ep.get("value"):
        candidates.append(float(ep["value"]))
    text = _flatten_text(resp)
    # Match ranges like "89 000 – 156 000 ₽" or "от 122 500 ₽" — take all numbers
    for m in re.finditer(r"(\d[\d\s]{2,8}(?:[\.,]\d+)?)\s*(?:₽|руб|RUB)", text):
        raw = m.group(1).replace(" ", "").replace(",", ".")
        try:
            candidates.append(float(raw))
        except ValueError:
            pass
    return max(candidates) if candidates else None


def grade(case: dict, resp: dict) -> dict:
    """Apply sentinel rules from a case definition. Returns {passed: bool, failures: [...]}."""
    failures: list[str] = []
    text = _flatten_text(resp)
    text_lower = text.lower()
    types = _doc_types(resp)

    for needed in case.get("expect_phrases", []):
        if needed.lower() not in text_lower:
            failures.append(f"missing required phrase: {needed!r}")

    any_set = case.get("expect_phrases_any") or []
    if any_set and not any(p.lower() in text_lower for p in any_set):
        failures.append(f"none of expect_phrases_any matched: {any_set}")

    for forbidden in case.get("forbid_phrases", []):
        if forbidden.lower() in text_lower:
            failures.append(f"forbidden phrase present: {forbidden!r}")

    for forbidden in case.get("forbid_phrases_strict", []):
        if forbidden.lower() in text_lower:
            failures.append(f"strict-forbidden phrase present: {forbidden!r}")

    dts_any = case.get("expect_doc_types_any") or []
    if dts_any and not any(t in types for t in dts_any):
        failures.append(f"none of expect_doc_types_any present in references: want={dts_any} got={sorted(types)}")

    pmin = case.get("expect_price_min_rub")
    if pmin is not None:
        max_price = _extract_max_price(resp)
        if max_price is None:
            failures.append(f"expected price ≥{pmin}₽ but no price detected")
        elif max_price < pmin:
            failures.append(f"max detected price {max_price}₽ < expected {pmin}₽")

    if case.get("expect_clickable_url"):
        if _extract_url_count(resp) == 0:
            failures.append("expected at least one clickable URL (Bitrix or site)")

    for field in case.get("expect_field_nonempty", []):
        val = resp.get(field)
        if not val:
            failures.append(f"expected non-empty field {field!r}, got {val!r}")

    return {"passed": not failures, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--host-header", default=None)
    ap.add_argument("--no-verify-ssl", action="store_true", default=False)
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--output", default=str(ROOT / "results.json"))
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter-id", default=None, help="Run only the case with this ID")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.filter_id:
        cases = [c for c in cases if c["id"] == args.filter_id]
    if args.limit > 0:
        cases = cases[: args.limit]

    url = f"{args.server.rstrip('/')}/query_structured"
    headers = {"Content-Type": "application/json"}
    if args.host_header:
        headers["Host"] = args.host_header

    print(f"Target:    {url}")
    print(f"Host:      {args.host_header or '(default)'}")
    print(f"Cases:     {len(cases)}")
    print()

    results: list[dict] = []
    passed = 0
    failed = 0

    with httpx.Client(verify=not args.no_verify_ssl, follow_redirects=True, timeout=120.0) as client:
        for i, case in enumerate(cases, 1):
            cid = case["id"]
            t0 = time.monotonic()
            try:
                r = client.post(url, json={"query": case["query"], "top_k": args.top_k}, headers=headers)
                elapsed = int((time.monotonic() - t0) * 1000)
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {r.text[:200]}"
                    results.append({"case_id": cid, "passed": False, "error": err, "elapsed_ms": elapsed})
                    failed += 1
                    print(f"[{i:2d}] FAIL {cid}  {err}")
                    continue
                resp = r.json()
            except Exception as e:
                results.append({"case_id": cid, "passed": False, "error": f"{type(e).__name__}: {e}"})
                failed += 1
                print(f"[{i:2d}] FAIL {cid}  network error: {e}")
                continue

            verdict = grade(case, resp)
            if verdict["passed"]:
                passed += 1
                print(f"[{i:2d}] PASS {cid}  ({elapsed} ms)")
            else:
                failed += 1
                print(f"[{i:2d}] FAIL {cid}  ({elapsed} ms)")
                for f in verdict["failures"]:
                    print(f"        ✗ {f}")

            results.append({
                "case_id": cid,
                "category": case.get("category", ""),
                "passed": verdict["passed"],
                "failures": verdict["failures"],
                "elapsed_ms": elapsed,
                "summary_preview": (resp.get("summary") or "")[:300],
                "doc_types": sorted(_doc_types(resp)),
                "max_price": _extract_max_price(resp),
                "url_count": _extract_url_count(resp),
                "historical_deals_count": len(resp.get("historical_deals") or []),
            })

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Result: {passed}/{passed + failed} passed.  Output: {args.output}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
