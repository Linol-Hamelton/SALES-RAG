#!/usr/bin/env python3
"""
Evaluation runner for Labus Sales RAG system.

Runs test_cases.json against the running FastAPI server and produces:
  - eval/results.json           — per-case detailed results
  - reports/model_readiness_report.md
  - reports/model_readiness_report.json

Usage:
    # Server must be running first:
    #   uvicorn app.main:app --port 8000 --workers 1
    python scripts/eval.py
    python scripts/eval.py --server http://localhost:8000
    python scripts/eval.py --cases eval/test_cases.json --verbose
    python scripts/eval.py --categories auto_priced guided_priced --verbose
"""
import asyncio
import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

CATEGORIES = [
    "auto_priced",
    "guided_priced",
    "manual_priced",
    "bundle_query",
    "direction_ambiguous",
    "edge_case",
]


def load_test_cases(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_confidence(result: dict, expected: str | None) -> tuple[bool, str]:
    """Check that the returned confidence tier matches expected."""
    if expected is None:
        return True, "skip"
    actual = (result.get("confidence") or "").lower()
    if actual == expected.lower():
        return True, actual
    return False, actual


def check_direction(result: dict, expected: str | None) -> tuple[bool, str]:
    """Check direction match (lenient — checks if expected appears in result text)."""
    if expected is None:
        return True, "skip"
    text = json.dumps(result, ensure_ascii=False).lower()
    if expected.lower() in text:
        return True, expected
    # Also check top-level direction field
    actual = (result.get("direction") or result.get("detected_direction") or "").lower()
    if expected.lower() in actual:
        return True, actual
    return False, actual


def check_flags(result: dict, must_contain: list[str], must_not_contain: list[str]) -> tuple[bool, list[str]]:
    """Check that required flags appear and forbidden strings are absent."""
    text = json.dumps(result, ensure_ascii=False).lower()
    failures = []
    for flag in must_contain:
        if flag.lower() not in text:
            failures.append(f"MISSING: '{flag}'")
    for item in must_not_contain:
        if item.lower() in text:
            failures.append(f"FORBIDDEN: '{item}'")
    return len(failures) == 0, failures


def check_bundle(result: dict, expected_bundle: bool) -> tuple[bool, str]:
    """Check that a bundle is returned when expected."""
    if not expected_bundle:
        return True, "skip"
    bundle = result.get("bundle") or result.get("suggested_bundle") or []
    has_bundle = bool(bundle)
    if has_bundle:
        return True, f"{len(bundle)} items"
    return False, "no bundle in response"


async def run_query(
    client: httpx.AsyncClient,
    server: str,
    query: str,
    timeout: float = 30.0,
) -> tuple[dict | None, float, str | None]:
    """Call /query_structured and return (result, latency_ms, error)."""
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{server}/query_structured",
            json={"query": query},
            timeout=timeout,
        )
        elapsed = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            return resp.json(), elapsed, None
        return None, elapsed, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return None, elapsed, str(e)


def evaluate_case(case: dict, result: dict | None, error: str | None) -> dict:
    """Evaluate a single test case result."""
    if error or result is None:
        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "passed": False,
            "error": error or "no result",
            "checks": {},
            "latency_ms": 0,
        }

    confidence_ok, confidence_actual = check_confidence(result, case.get("expected_confidence"))
    direction_ok, direction_actual = check_direction(result, case.get("expected_direction"))
    flags_ok, flag_failures = check_flags(
        result,
        case.get("must_contain_flags", []),
        case.get("must_not_contain", []),
    )
    bundle_ok, bundle_detail = check_bundle(result, case.get("expected_bundle", False))

    all_passed = confidence_ok and direction_ok and flags_ok and bundle_ok

    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "passed": all_passed,
        "error": None,
        "checks": {
            "confidence": {"ok": confidence_ok, "expected": case.get("expected_confidence"), "actual": confidence_actual},
            "direction": {"ok": direction_ok, "expected": case.get("expected_direction"), "actual": direction_actual},
            "flags": {"ok": flags_ok, "failures": flag_failures},
            "bundle": {"ok": bundle_ok, "detail": bundle_detail},
        },
        "notes": case.get("notes", ""),
    }


async def run_eval(
    server: str,
    cases_path: Path,
    categories: list[str] | None,
    verbose: bool,
    concurrency: int,
) -> dict:
    """Run all test cases and return aggregated results."""
    cases = load_test_cases(cases_path)

    if categories:
        cases = [c for c in cases if c["category"] in categories]
        console.print(f"Filtered to {len(cases)} cases in categories: {categories}")

    results = []
    latencies = []

    async with httpx.AsyncClient() as client:
        # First check health
        try:
            health = await client.get(f"{server}/health", timeout=5)
            health_data = health.json()
            console.print(f"Server health: {health_data.get('status', '?')}")
        except Exception as e:
            console.print(f"[red]Cannot reach server at {server}: {e}[/red]")
            console.print("Start the server first: uvicorn app.main:app --port 8000 --workers 1")
            sys.exit(1)

        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(case: dict) -> dict:
            async with semaphore:
                result, latency_ms, error = await run_query(client, server, case["query"])
                eval_result = evaluate_case(case, result, error)
                eval_result["latency_ms"] = latency_ms
                latencies.append(latency_ms)
                if verbose:
                    status = "[green]PASS[/green]" if eval_result["passed"] else "[red]FAIL[/red]"
                    console.print(f"  {status} [{case['id']}] {case['query'][:60]} ({latency_ms:.0f}ms)")
                    if not eval_result["passed"] and eval_result.get("checks"):
                        for check_name, check in eval_result["checks"].items():
                            if not check.get("ok"):
                                console.print(f"       ↳ {check_name}: {check}")
                return eval_result

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running eval...", total=len(cases))
            tasks = []
            for case in cases:
                tasks.append(run_one(case))

            for coro in asyncio.as_completed(tasks):
                r = await coro
                results.append(r)
                progress.advance(task)

    return aggregate_results(results, latencies)


def aggregate_results(results: list[dict], latencies: list[float]) -> dict:
    """Aggregate per-case results into summary metrics."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    errors = sum(1 for r in results if r.get("error"))

    # By category
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "failed": 0}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed"] += 1

    # Confidence accuracy (only where expected != null)
    conf_cases = [r for r in results if r.get("checks", {}).get("confidence", {}).get("expected") not in (None, "skip")]
    conf_ok = sum(1 for r in conf_cases if r.get("checks", {}).get("confidence", {}).get("ok", False))
    conf_acc = conf_ok / len(conf_cases) if conf_cases else None

    # Direction accuracy
    dir_cases = [r for r in results if r.get("checks", {}).get("direction", {}).get("expected") not in (None, "skip")]
    dir_ok = sum(1 for r in dir_cases if r.get("checks", {}).get("direction", {}).get("ok", False))
    dir_acc = dir_ok / len(dir_cases) if dir_cases else None

    # Bundle recall
    bundle_cases = [r for r in results if r.get("checks", {}).get("bundle", {}).get("detail") != "skip"]
    bundle_ok = sum(1 for r in bundle_cases if r.get("checks", {}).get("bundle", {}).get("ok", False))
    bundle_recall = bundle_ok / len(bundle_cases) if bundle_cases else None

    # Latency stats
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # Flag checks accuracy
    flag_cases = [r for r in results if r.get("checks", {}).get("flags")]
    flag_ok = sum(1 for r in flag_cases if r.get("checks", {}).get("flags", {}).get("ok", False))
    flag_acc = flag_ok / len(flag_cases) if flag_cases else None

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / total if total else 0,
        "metrics": {
            "confidence_accuracy": conf_acc,
            "direction_accuracy": dir_acc,
            "bundle_recall": bundle_recall,
            "flag_accuracy": flag_acc,
        },
        "latency_ms": {
            "avg": round(avg_latency, 1),
            "p95": round(p95_latency, 1),
            "max": round(max_latency, 1),
        },
        "by_category": by_category,
        "cases": results,
    }
    return summary


def print_summary(summary: dict) -> None:
    """Print a rich summary table."""
    console.print()
    console.rule("[bold]Eval Summary[/bold]")

    total = summary["total_cases"]
    passed = summary["passed"]
    rate = summary["pass_rate"] * 100
    color = "green" if rate >= 70 else "yellow" if rate >= 50 else "red"

    console.print(f"  Total: {total}  Passed: [{color}]{passed}[/{color}]  Rate: [{color}]{rate:.1f}%[/{color}]")

    m = summary["metrics"]
    if m["confidence_accuracy"] is not None:
        console.print(f"  Confidence accuracy: {m['confidence_accuracy']*100:.1f}%")
    if m["direction_accuracy"] is not None:
        console.print(f"  Direction accuracy: {m['direction_accuracy']*100:.1f}%")
    if m["bundle_recall"] is not None:
        console.print(f"  Bundle recall: {m['bundle_recall']*100:.1f}%")
    if m["flag_accuracy"] is not None:
        console.print(f"  Flag check accuracy: {m['flag_accuracy']*100:.1f}%")

    lat = summary["latency_ms"]
    console.print(f"  Latency — avg: {lat['avg']}ms  p95: {lat['p95']}ms  max: {lat['max']}ms")

    # Category table
    table = Table(title="Results by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Rate", justify="right")

    for cat, stats in summary["by_category"].items():
        cat_rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        color = "green" if cat_rate >= 70 else "yellow" if cat_rate >= 50 else "red"
        table.add_row(
            cat,
            str(stats["total"]),
            str(stats["passed"]),
            str(stats["failed"]),
            f"[{color}]{cat_rate:.0f}%[/{color}]",
        )

    console.print(table)

    # Failures
    failures = [r for r in summary["cases"] if not r["passed"]]
    if failures:
        console.print(f"\n[red]Failed cases ({len(failures)}):[/red]")
        for r in failures[:20]:  # show at most 20
            console.print(f"  [{r['id']}] {r['query'][:60]}")
            if r.get("error"):
                console.print(f"    error: {r['error']}")
            elif r.get("checks"):
                for check_name, check in r["checks"].items():
                    if not check.get("ok"):
                        console.print(f"    {check_name}: {check}")


def write_results(summary: dict, output_dir: Path, reports_dir: Path) -> None:
    """Write results.json and model_readiness_report files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # eval/results.json
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"\nResults written to {results_path}")

    # reports/model_readiness_report.json
    report_json_path = reports_dir / "model_readiness_report.json"
    slim = {k: v for k, v in summary.items() if k != "cases"}
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

    # reports/model_readiness_report.md
    m = summary["metrics"]
    lat = summary["latency_ms"]

    def pct(v) -> str:
        return f"{v*100:.1f}%" if v is not None else "n/a"

    lines = [
        "# Model Readiness Report",
        "",
        f"Generated: {summary['timestamp']}",
        "",
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cases | {summary['total_cases']} |",
        f"| Passed | {summary['passed']} |",
        f"| Failed | {summary['failed']} |",
        f"| Pass rate | {pct(summary['pass_rate'])} |",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Confidence accuracy | {pct(m['confidence_accuracy'])} |",
        f"| Direction accuracy | {pct(m['direction_accuracy'])} |",
        f"| Bundle recall | {pct(m['bundle_recall'])} |",
        f"| Flag accuracy | {pct(m['flag_accuracy'])} |",
        "",
        "## Latency",
        "",
        f"| Percentile | ms |",
        f"|-----------|-----|",
        f"| avg | {lat['avg']} |",
        f"| p95 | {lat['p95']} |",
        f"| max | {lat['max']} |",
        "",
        "## By Category",
        "",
        "| Category | Total | Passed | Rate |",
        "|----------|-------|--------|------|",
    ]

    for cat, stats in summary["by_category"].items():
        cat_rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        lines.append(f"| {cat} | {stats['total']} | {stats['passed']} | {cat_rate:.0f}% |")

    lines += [
        "",
        "## Failed Cases",
        "",
    ]
    failures = [r for r in summary["cases"] if not r["passed"]]
    if failures:
        for r in failures:
            lines.append(f"### [{r['id']}] {r['query']}")
            if r.get("error"):
                lines.append(f"- **Error:** {r['error']}")
            elif r.get("checks"):
                for check_name, check in r["checks"].items():
                    if not check.get("ok"):
                        lines.append(f"- **{check_name}:** {check}")
            lines.append("")
    else:
        lines.append("_No failures._")

    report_md_path = reports_dir / "model_readiness_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    console.print(f"Report written to {report_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Eval runner for Labus Sales RAG")
    parser.add_argument("--server", default="http://localhost:8000", help="FastAPI server URL")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "test_cases.json"), help="Path to test_cases.json")
    parser.add_argument("--categories", nargs="*", choices=CATEGORIES, help="Filter by category")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval"), help="Directory for results.json")
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"), help="Directory for report files")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent requests")
    parser.add_argument("--verbose", action="store_true", help="Print per-case results")
    args = parser.parse_args()

    console.print(f"[bold]Labus Sales RAG — Eval Runner[/bold]")
    console.print(f"Server: {args.server}")
    console.print(f"Cases: {args.cases}")
    console.print()

    summary = asyncio.run(
        run_eval(
            server=args.server,
            cases_path=Path(args.cases),
            categories=args.categories,
            verbose=args.verbose,
            concurrency=args.concurrency,
        )
    )

    print_summary(summary)
    write_results(
        summary,
        output_dir=Path(args.output_dir),
        reports_dir=Path(args.reports_dir),
    )

    # Exit with code 1 if pass rate < 50%
    if summary["pass_rate"] < 0.5:
        console.print("[red]Pass rate below 50% — model is NOT ready.[/red]")
        sys.exit(1)
    else:
        console.print(f"[green]Eval complete.[/green]")


if __name__ == "__main__":
    main()
