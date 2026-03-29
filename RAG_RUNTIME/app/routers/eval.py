"""Eval summary endpoint."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Eval"])

PROJECT_ROOT = Path(__file__).parent.parent.parent


@router.get("/eval_summary", tags=["Eval"])
async def eval_summary() -> dict:
    """Return last eval results."""
    results_path = PROJECT_ROOT / "eval" / "results.json"
    if not results_path.exists():
        raise HTTPException(
            404,
            "Eval results not found. Run: python scripts/eval_run.py"
        )
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)
