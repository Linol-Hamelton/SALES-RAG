#!/usr/bin/env python3
"""Generate data readiness report from analytics artifacts."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
ANALYTICS_ROOT = PROJECT_ROOT.parent / "RAG_ANALYTICS" / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPECTED_FILES = {
    "normalized/goods.normalized.csv": 9095,
    "normalized/offers.normalized.csv": 11674,
    "normalized/orders.normalized.csv": 50404,
    "facts/product_facts.csv": 12626,
    "facts/bundle_facts.csv": 12566,
    "facts/deal_facts.csv": 18161,
    "facts/template_match_facts.csv": 4404,
    "facts/client_facts.csv": 2699,
    "pricing/pricing_recommendations.csv": 12626,
    "kpis/catalog_utilization.csv": 9095,
    "kpis/client_concentration.csv": 2697,
    "kpis/monthly_direction_kpis.csv": 633,
    "kpis/price_stability.csv": 12626,
    "kpis/template_conversion_proxy.csv": 4404,
}

EXPECTED_JSON = ["qa_report.json", "pricing/pricing_summary.json", "kpis/kpi_summary.json"]


def check_file(rel_path: str, expected_rows: int | None = None) -> dict:
    path = ANALYTICS_ROOT / rel_path
    result = {"path": str(rel_path), "exists": path.exists(), "ok": False}

    if not path.exists():
        result["error"] = "file not found"
        return result

    result["size_bytes"] = path.stat().st_size

    if rel_path.endswith(".csv"):
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str, low_memory=False)
            result["row_count"] = len(df)
            result["columns"] = list(df.columns)

            if expected_rows:
                result["expected_rows"] = expected_rows
                result["row_count_ok"] = len(df) == expected_rows

            # Check key columns
            for col in ["NORMALIZED_DIRECTION", "PRICE_MODE", "CONFIDENCE_TIER"]:
                if col in df.columns:
                    result[f"has_{col.lower()}"] = True
                    non_null = df[col].notna().sum()
                    result[f"{col.lower()}_filled_pct"] = round(non_null / max(len(df), 1) * 100, 1)

            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)
    elif rel_path.endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            result["keys"] = list(data.keys()) if isinstance(data, dict) else "array"
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

    return result


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating data readiness report...")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analytics_root": str(ANALYTICS_ROOT),
        "csv_files": [],
        "json_files": [],
        "summary": {},
    }

    all_ok = True

    print("\nChecking CSV files:")
    for rel_path, expected_rows in EXPECTED_FILES.items():
        r = check_file(rel_path, expected_rows)
        report["csv_files"].append(r)
        status = "OK" if r.get("ok") else "FAIL"
        row_info = f"{r.get('row_count', '?')}/{expected_rows}" if expected_rows else ""
        print(f"  [{status}] {rel_path} {row_info}")
        if not r.get("ok"):
            all_ok = False

    print("\nChecking JSON files:")
    for rel_path in EXPECTED_JSON:
        r = check_file(rel_path)
        report["json_files"].append(r)
        status = "OK" if r.get("ok") else "FAIL"
        print(f"  [{status}] {rel_path}")
        if not r.get("ok"):
            all_ok = False

    report["summary"] = {
        "all_ok": all_ok,
        "csv_files_checked": len(EXPECTED_FILES),
        "json_files_checked": len(EXPECTED_JSON),
        "csv_ok": sum(1 for r in report["csv_files"] if r.get("ok")),
        "json_ok": sum(1 for r in report["json_files"] if r.get("ok")),
    }

    # Write JSON report
    json_path = REPORTS_DIR / "data_readiness_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report: {json_path}")

    # Write Markdown report
    md_path = REPORTS_DIR / "data_readiness_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Data Readiness Report\n\n")
        f.write(f"Generated: {report['generated_at']}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Status: {'All OK' if all_ok else 'Issues Found'}\n")
        f.write(f"- CSV files: {report['summary']['csv_ok']}/{report['summary']['csv_files_checked']}\n")
        f.write(f"- JSON files: {report['summary']['json_ok']}/{report['summary']['json_files_checked']}\n\n")
        f.write("## CSV Files\n\n")
        f.write("| File | Rows | Expected | OK |\n|---|---|---|---|\n")
        for r in report["csv_files"]:
            status = "yes" if r.get("ok") else "no"
            rows = r.get("row_count", "?")
            exp = r.get("expected_rows", "-")
            f.write(f"| {r['path']} | {rows} | {exp} | {status} |\n")
    print(f"Markdown report: {md_path}")

    if all_ok:
        print("\nAll data artifacts ready for RAG ingestion!")
    else:
        print("\nSome files have issues. Check the report.")
        sys.exit(1)


if __name__ == "__main__":
    main()
