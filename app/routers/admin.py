"""Admin endpoints: index rebuild, feedback debug."""
import subprocess
import sys
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Depends
from app.schemas.query import RebuildIndexRequest, RebuildIndexResponse
from app.database import get_connection
from app.auth import get_current_user
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _run_rebuild(doc_types: list[str], recreate: bool):
    """Background task to rebuild index."""
    try:
        # Step 1: Ingest
        ingest_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest.py")]
        for dt in doc_types:
            ingest_cmd.extend(["-t", dt])
        logger.info("Starting ingestion", doc_types=doc_types)
        result = subprocess.run(ingest_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Ingestion failed", stderr=result.stderr)
            return

        # Step 2: Build index
        build_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "build_index.py")]
        if recreate:
            build_cmd.append("--recreate")
        logger.info("Starting index build")
        result = subprocess.run(build_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Index build failed", stderr=result.stderr)
            return

        logger.info("Index rebuild complete")
    except Exception as e:
        logger.error("Rebuild failed", error=str(e))


@router.post("/rebuild_index", response_model=RebuildIndexResponse, tags=["Admin"])
async def rebuild_index(
    req: RebuildIndexRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> RebuildIndexResponse:
    """
    Trigger a background index rebuild.
    Runs ingestion + embedding in the background.
    The running API continues serving requests with the old index.
    """
    doc_types = req.doc_types or ["product", "bundle", "policy", "support"]
    background_tasks.add_task(_run_rebuild, doc_types, req.recreate_collection)

    logger.info("Index rebuild scheduled", doc_types=doc_types)
    return RebuildIndexResponse(
        status="started",
        message="Index rebuild started in background. Check logs for progress.",
        doc_types=doc_types,
    )


@router.get("/feedback/lessons", tags=["Admin"])
def list_feedback_lessons(
    limit: int = 50,
    active_only: bool = True,
    request: Request = None,
    user: dict = Depends(get_current_user),
):
    """List feedback lessons stored in DB (admin debug endpoint)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    where = "WHERE fl.is_active = 1" if active_only else ""
    rows = conn.execute(f"""
        SELECT fl.id, fl.feedback_id, fl.user_query, fl.direction,
               fl.lesson_text, fl.rating, fl.match_count, fl.is_active,
               fl.created_at, mf.comment as original_comment
        FROM feedback_lessons fl
        LEFT JOIN message_feedback mf ON mf.id = fl.feedback_id
        {where}
        ORDER BY fl.created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/feedback/stats", tags=["Admin"])
def feedback_stats(user: dict = Depends(get_current_user)):
    """Aggregate stats about feedback collection (admin debug endpoint)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_feedback,
            SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN comment != '' THEN 1 ELSE 0 END) as with_comment
        FROM message_feedback
    """).fetchone()
    lessons = conn.execute("""
        SELECT
            COUNT(*) as total_lessons,
            SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_lessons,
            SUM(match_count) as total_matches
        FROM feedback_lessons
    """).fetchone()
    rules = conn.execute("SELECT COUNT(*) as total_rules FROM feedback_rules WHERE is_active = 1").fetchone()
    conn.close()

    # Also get in-memory cache state
    feedback_store = getattr(request.app.state, "feedback_store", None) if request else None
    cache_size = len(feedback_store._lessons) if feedback_store else -1

    return {
        "feedback": dict(stats),
        "lessons": dict(lessons),
        "rules": dict(rules),
        "in_memory_cache_size": cache_size,
    }


@router.post("/feedback/reload", tags=["Admin"])
def reload_feedback_store(request: Request, user: dict = Depends(get_current_user)):
    """Reload feedback lessons from DB into in-memory cache (admin)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if not feedback_store:
        raise HTTPException(503, "Feedback store not initialized")
    feedback_store.load()
    return {"status": "ok", "lessons_loaded": len(feedback_store._lessons)}


@router.get("/messages", tags=["Admin"])
def list_all_messages(
    limit: int = 100,
    offset: int = 0,
    username: str | None = None,
    with_feedback_only: bool = False,
    user: dict = Depends(get_current_user),
):
    """
    View all messages across all users with their feedback comments.
    Supports pagination (limit/offset), filter by username, or only messages with feedback.
    """
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")

    filters = []
    params: list = []

    if username:
        filters.append("u.username = ?")
        params.append(username)
    if with_feedback_only:
        filters.append("mf.id IS NOT NULL")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    conn = get_connection()
    rows = conn.execute(f"""
        SELECT
            m.id            as message_id,
            m.chat_id,
            c.title         as chat_title,
            u.username,
            m.role,
            m.content,
            m.mode,
            m.latency_ms,
            m.created_at,
            mf.rating       as feedback_rating,
            mf.comment      as feedback_comment,
            mf.created_at   as feedback_at,
            -- preceding user query for assistant messages
            (SELECT content FROM messages p
             WHERE p.chat_id = m.chat_id AND p.id < m.id AND p.role = 'user'
             ORDER BY p.id DESC LIMIT 1) as user_query
        FROM messages m
        JOIN chats c ON c.id = m.chat_id
        JOIN users u ON u.id = c.user_id
        LEFT JOIN message_feedback mf ON mf.message_id = m.id
        {where}
        ORDER BY m.id DESC
        LIMIT ? OFFSET ?
    """, (*params, limit, offset)).fetchall()

    total = conn.execute(f"""
        SELECT COUNT(*) FROM messages m
        JOIN chats c ON c.id = m.chat_id
        JOIN users u ON u.id = c.user_id
        LEFT JOIN message_feedback mf ON mf.message_id = m.id
        {where}
    """, params).fetchone()[0]

    conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [dict(r) for r in rows],
    }
