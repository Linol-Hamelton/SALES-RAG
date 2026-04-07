"""Admin endpoints: index rebuild, feedback management, user/chat management."""
import subprocess
import sys
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Depends
from pydantic import BaseModel
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


@router.get("/chats_list", tags=["Admin"])
def list_admin_chats(
    limit: int = 50,
    offset: int = 0,
    user_id: int | None = None,
    with_feedback_only: bool = False,
    user: dict = Depends(get_current_user),
):
    """List all chats with summary stats, grouped by user. Supports filtering."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")

    filters = []
    params: list = []
    if user_id:
        filters.append("c.user_id = ?")
        params.append(user_id)
    if with_feedback_only:
        filters.append("""c.id IN (
            SELECT DISTINCT m2.chat_id FROM messages m2
            JOIN message_feedback mf2 ON mf2.message_id = m2.id)""")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    conn = get_connection()
    rows = conn.execute(f"""
        SELECT
            c.id            as chat_id,
            c.title,
            u.username,
            u.id            as user_id,
            c.created_at,
            c.updated_at,
            (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) as message_count,
            (SELECT COUNT(*) FROM messages m
             JOIN message_feedback mf ON mf.message_id = m.id
             WHERE m.chat_id = c.id) as feedback_count,
            (SELECT COUNT(*) FROM messages m
             JOIN message_feedback mf ON mf.message_id = m.id
             WHERE m.chat_id = c.id AND mf.comment != '') as comment_count
        FROM chats c
        JOIN users u ON u.id = c.user_id
        {where}
        ORDER BY c.updated_at DESC
        LIMIT ? OFFSET ?
    """, (*params, limit, offset)).fetchall()

    total = conn.execute(f"""
        SELECT COUNT(*) FROM chats c
        JOIN users u ON u.id = c.user_id
        {where}
    """, params).fetchone()[0]

    conn.close()
    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


@router.get("/chats_list/{chat_id}/messages", tags=["Admin"])
def get_admin_chat_messages(chat_id: int, user: dict = Depends(get_current_user)):
    """Get all messages in a specific chat with feedback (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            m.id            as message_id,
            m.chat_id,
            m.role,
            m.content,
            m.mode,
            m.latency_ms,
            m.created_at,
            mf.id           as feedback_id,
            mf.rating       as feedback_rating,
            mf.comment      as feedback_comment,
            mf.created_at   as feedback_at
        FROM messages m
        LEFT JOIN message_feedback mf ON mf.message_id = m.id
        WHERE m.chat_id = ?
        ORDER BY m.id ASC
    """, (chat_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
            mf.id           as feedback_id,
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


# ── Feedback lesson management ────────────────────────────────────────────


class EditLessonRequest(BaseModel):
    lesson_text: str | None = None
    is_active: bool | None = None


@router.patch("/feedback/lessons/{lesson_id}", tags=["Admin"])
def edit_feedback_lesson(
    lesson_id: int,
    req: EditLessonRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Edit or deactivate a feedback lesson (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    existing = conn.execute("SELECT id FROM feedback_lessons WHERE id = ?", (lesson_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Lesson not found")
    updates = []
    params = []
    if req.lesson_text is not None:
        updates.append("lesson_text = ?")
        params.append(req.lesson_text)
    if req.is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if req.is_active else 0)
    if not updates:
        conn.close()
        return {"status": "no changes"}
    params.append(lesson_id)
    conn.execute(f"UPDATE feedback_lessons SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    # Reload in-memory cache
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store:
        feedback_store.load()
    return {"status": "updated", "lesson_id": lesson_id}


@router.delete("/feedback/lessons/{lesson_id}", tags=["Admin"])
def delete_feedback_lesson(
    lesson_id: int,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Delete a feedback lesson permanently (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    existing = conn.execute("SELECT id FROM feedback_lessons WHERE id = ?", (lesson_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Lesson not found")
    conn.execute("DELETE FROM feedback_lessons WHERE id = ?", (lesson_id,))
    conn.commit()
    conn.close()
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store:
        feedback_store.load()
    return {"status": "deleted", "lesson_id": lesson_id}


# ── Feedback comment management ───────────────────────────────────────────


class EditCommentRequest(BaseModel):
    comment: str


class CreateFeedbackRequest(BaseModel):
    message_id: int
    rating: int  # -1 or 1
    comment: str = ""


@router.post("/feedback/create", tags=["Admin"])
def admin_create_feedback(
    req: CreateFeedbackRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Create feedback on any assistant message (admin only).
    If feedback already exists for (message, admin) it is updated instead.
    Also creates a feedback lesson if the message is assistant and comment is non-empty.
    """
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    if req.rating not in (-1, 1):
        raise HTTPException(400, "rating must be -1 or 1")
    conn = get_connection()
    msg = conn.execute(
        "SELECT id, chat_id, role, structured_data FROM messages WHERE id = ?",
        (req.message_id,),
    ).fetchone()
    if not msg:
        conn.close()
        raise HTTPException(404, "Message not found")
    conn.execute("""
        INSERT INTO message_feedback (message_id, user_id, rating, comment)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(message_id, user_id) DO UPDATE SET
            rating=excluded.rating, comment=excluded.comment
    """, (req.message_id, user["id"], req.rating, req.comment))
    conn.commit()
    fb_row = conn.execute(
        "SELECT id FROM message_feedback WHERE message_id = ? AND user_id = ?",
        (req.message_id, user["id"]),
    ).fetchone()
    feedback_id = fb_row["id"] if fb_row else None

    # Create lesson (best-effort), mirroring chats.submit_feedback logic
    if req.comment and req.comment.strip() and feedback_id and msg["role"] == "assistant":
        try:
            import json as _json
            feedback_store = getattr(request.app.state, "feedback_store", None)
            retriever = getattr(request.app.state, "retriever", None)
            if feedback_store and retriever and retriever.is_ready:
                user_query_row = conn.execute("""
                    SELECT content FROM messages
                    WHERE chat_id = ? AND id < ? AND role = 'user'
                    ORDER BY id DESC LIMIT 1
                """, (msg["chat_id"], req.message_id)).fetchone()
                direction = ""
                if msg["structured_data"]:
                    try:
                        sd = _json.loads(msg["structured_data"])
                        bundle = sd.get("suggested_bundle") or []
                        if bundle and bundle[0].get("direction"):
                            direction = bundle[0]["direction"]
                        if not direction:
                            src = sd.get("source_distinction") or {}
                            direction = src.get("dataset_type", "") or ""
                    except (ValueError, TypeError, AttributeError):
                        pass
                if user_query_row:
                    uq = user_query_row["content"]
                    qe = retriever.embed_query(uq)
                    feedback_store.add_lesson(
                        feedback_id=feedback_id,
                        user_query=uq,
                        query_embedding=qe,
                        direction=direction,
                        comment=req.comment,
                        rating=req.rating,
                    )
        except Exception:
            pass

    conn.close()
    return {
        "status": "ok",
        "feedback_id": feedback_id,
        "message_id": req.message_id,
        "rating": req.rating,
        "comment": req.comment,
    }


@router.patch("/feedback/comments/{feedback_id}", tags=["Admin"])
def edit_feedback_comment(
    feedback_id: int,
    req: EditCommentRequest,
    user: dict = Depends(get_current_user),
):
    """Edit a feedback comment (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    existing = conn.execute("SELECT id FROM message_feedback WHERE id = ?", (feedback_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Feedback not found")
    conn.execute("UPDATE message_feedback SET comment = ? WHERE id = ?", (req.comment, feedback_id))
    conn.commit()
    conn.close()
    return {"status": "updated", "feedback_id": feedback_id}


@router.delete("/feedback/comments/{feedback_id}", tags=["Admin"])
def delete_feedback_comment(
    feedback_id: int,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Delete a feedback entry and its lesson (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    existing = conn.execute("SELECT id FROM message_feedback WHERE id = ?", (feedback_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Feedback not found")
    # Delete associated lesson first (FK constraint)
    conn.execute("DELETE FROM feedback_lessons WHERE feedback_id = ?", (feedback_id,))
    conn.execute("DELETE FROM message_feedback WHERE id = ?", (feedback_id,))
    conn.commit()
    conn.close()
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store:
        feedback_store.load()
    return {"status": "deleted", "feedback_id": feedback_id}


# ── Chat management ───────────────────────────────────────────────────────


@router.delete("/chats/{chat_id}", tags=["Admin"])
def delete_chat(chat_id: int, user: dict = Depends(get_current_user)):
    """Delete any chat and all its messages/feedback (admin only). Cascade deletes."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    chat = conn.execute("SELECT id, user_id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    # Delete feedback lessons linked to this chat's messages
    conn.execute("""
        DELETE FROM feedback_lessons WHERE feedback_id IN (
            SELECT mf.id FROM message_feedback mf
            JOIN messages m ON m.id = mf.message_id
            WHERE m.chat_id = ?
        )
    """, (chat_id,))
    # Cascade: messages + message_feedback deleted by FK
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "chat_id": chat_id}


# ── User management (deactivate/delete active users) ─────────────────────


@router.post("/users/{user_id}/deactivate", tags=["Admin"])
def deactivate_user(user_id: int, user: dict = Depends(get_current_user)):
    """Deactivate a user account (admin only). User can no longer log in."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    if user_id == user["id"]:
        raise HTTPException(400, "Cannot deactivate yourself")
    conn = get_connection()
    target = conn.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not target:
        conn.close()
        raise HTTPException(404, "User not found")
    if target["role"] == "admin":
        conn.close()
        raise HTTPException(400, "Cannot deactivate another admin")
    conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "deactivated", "user_id": user_id, "username": target["username"]}


@router.delete("/users/{user_id}", tags=["Admin"])
def delete_user(user_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Permanently delete a user and all their data (admin only). Cascade deletes chats, messages, feedback."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    if user_id == user["id"]:
        raise HTTPException(400, "Cannot delete yourself")
    conn = get_connection()
    target = conn.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not target:
        conn.close()
        raise HTTPException(404, "User not found")
    if target["role"] == "admin":
        conn.close()
        raise HTTPException(400, "Cannot delete another admin")
    # Delete feedback lessons linked to user's messages
    conn.execute("""
        DELETE FROM feedback_lessons WHERE feedback_id IN (
            SELECT mf.id FROM message_feedback mf
            JOIN messages m ON m.id = mf.message_id
            JOIN chats c ON c.id = m.chat_id
            WHERE c.user_id = ?
        )
    """, (user_id,))
    # CASCADE handles chats → messages → message_feedback
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store:
        feedback_store.load()
    return {"status": "deleted", "user_id": user_id, "username": target["username"]}
