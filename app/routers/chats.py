"""Chat management endpoints: list chats, get messages, create chat, delete chat, feedback."""
import json
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, Request

from app.database import get_connection
from app.auth import get_current_user

router = APIRouter(prefix="/chats", tags=["Chats"])


# ── Schemas ──────────────────────────────────────────────────────────────

class ChatCreate(BaseModel):
    title: str = "Новый чат"


class ChatUpdate(BaseModel):
    title: str


class ChatSummary(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message: str = ""


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    mode: str = "structured"
    structured_data: dict | None = None
    latency_ms: int = 0
    created_at: str
    feedback: int | None = None  # -1, 0 (none), 1
    feedback_comment: str = ""


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=-1, le=1)
    comment: str = ""


# ── Chats CRUD ───────────────────────────────────────────────────────────

@router.get("", response_model=list[ChatSummary])
def list_chats(user: dict = Depends(get_current_user)):
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.id, c.title, c.created_at, c.updated_at,
               COUNT(m.id) as message_count,
               COALESCE((SELECT content FROM messages WHERE chat_id = c.id ORDER BY id DESC LIMIT 1), '') as last_message
        FROM chats c
        LEFT JOIN messages m ON m.chat_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    """, (user["id"],)).fetchall()
    conn.close()
    return [ChatSummary(
        id=r["id"], title=r["title"], created_at=r["created_at"],
        updated_at=r["updated_at"], message_count=r["message_count"],
        last_message=r["last_message"][:100],
    ) for r in rows]


@router.post("", response_model=ChatSummary)
def create_chat(req: ChatCreate, user: dict = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO chats (user_id, title) VALUES (?, ?)",
        (user["id"], req.title),
    )
    conn.commit()
    chat_id = cur.lastrowid
    row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    conn.close()
    return ChatSummary(
        id=row["id"], title=row["title"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


@router.patch("/{chat_id}", response_model=ChatSummary)
def update_chat(chat_id: int, req: ChatUpdate, user: dict = Depends(get_current_user)):
    conn = get_connection()
    row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.execute("UPDATE chats SET title = ?, updated_at = datetime('now') WHERE id = ?", (req.title, chat_id))
    conn.commit()
    updated = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    conn.close()
    return ChatSummary(id=updated["id"], title=updated["title"],
                       created_at=updated["created_at"], updated_at=updated["updated_at"])


@router.delete("/{chat_id}")
def delete_chat(chat_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    row = conn.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


# ── Messages ─────────────────────────────────────────────────────────────

@router.get("/{chat_id}/messages", response_model=list[MessageOut])
def get_messages(chat_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    chat = conn.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    rows = conn.execute("""
        SELECT m.id, m.role, m.content, m.mode, m.structured_data, m.latency_ms, m.created_at,
               mf.rating as feedback, mf.comment as feedback_comment
        FROM messages m
        LEFT JOIN message_feedback mf ON mf.message_id = m.id AND mf.user_id = ?
        WHERE m.chat_id = ?
        ORDER BY m.id ASC
    """, (user["id"], chat_id)).fetchall()
    conn.close()
    result = []
    for r in rows:
        sd = None
        if r["structured_data"]:
            try:
                sd = json.loads(r["structured_data"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(MessageOut(
            id=r["id"], role=r["role"], content=r["content"],
            mode=r["mode"] or "structured",
            structured_data=sd,
            latency_ms=r["latency_ms"] or 0,
            created_at=r["created_at"],
            feedback=r["feedback"],
            feedback_comment=r["feedback_comment"] or "",
        ))
    return result


# ── Feedback (RLHF) ─────────────────────────────────────────────────────

@router.post("/{chat_id}/messages/{message_id}/feedback")
def submit_feedback(chat_id: int, message_id: int, req: FeedbackRequest,
                    request: Request, user: dict = Depends(get_current_user)):
    conn = get_connection()
    # Verify ownership
    msg = conn.execute("""
        SELECT m.id FROM messages m
        JOIN chats c ON c.id = m.chat_id
        WHERE m.id = ? AND m.chat_id = ? AND c.user_id = ?
    """, (message_id, chat_id, user["id"])).fetchone()
    if not msg:
        conn.close()
        raise HTTPException(404, "Message not found")
    conn.execute("""
        INSERT INTO message_feedback (message_id, user_id, rating, comment)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(message_id, user_id) DO UPDATE SET rating=excluded.rating, comment=excluded.comment
    """, (message_id, user["id"], req.rating, req.comment))
    conn.commit()

    # Get feedback row ID for lesson creation
    fb_row = conn.execute(
        "SELECT id FROM message_feedback WHERE message_id = ? AND user_id = ?",
        (message_id, user["id"])
    ).fetchone()
    feedback_id = fb_row["id"] if fb_row else None

    # Create lesson from feedback (if comment is non-empty)
    if req.comment and req.comment.strip() and feedback_id:
        try:
            feedback_store = getattr(request.app.state, "feedback_store", None)
            retriever = getattr(request.app.state, "retriever", None)
            if feedback_store and retriever and retriever.is_ready:
                # Find the user query that preceded this assistant message
                user_query_row = conn.execute("""
                    SELECT content FROM messages
                    WHERE chat_id = ? AND id < ? AND role = 'user'
                    ORDER BY id DESC LIMIT 1
                """, (chat_id, message_id)).fetchone()
                # Extract direction from assistant message's structured_data
                asst_msg_row = conn.execute(
                    "SELECT structured_data FROM messages WHERE id = ?",
                    (message_id,)
                ).fetchone()
                direction = ""
                if asst_msg_row and asst_msg_row["structured_data"]:
                    try:
                        sd = json.loads(asst_msg_row["structured_data"])
                        # Try suggested_bundle first
                        bundle = sd.get("suggested_bundle") or []
                        if bundle and bundle[0].get("direction"):
                            direction = bundle[0]["direction"]
                        # Fallback: source_distinction dataset_type
                        if not direction:
                            src = sd.get("source_distinction") or {}
                            direction = src.get("dataset_type", "") or ""
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                if user_query_row:
                    user_query = user_query_row["content"]
                    query_embedding = retriever.embed_query(user_query)
                    feedback_store.add_lesson(
                        feedback_id=feedback_id,
                        user_query=user_query,
                        query_embedding=query_embedding,
                        direction=direction,
                        comment=req.comment,
                        rating=req.rating,
                    )
        except Exception:
            pass  # Don't fail the feedback endpoint over lesson creation

    conn.close()
    return {"status": "ok", "rating": req.rating}


# ── Feedback export (for RLHF training) ─────────────────────────────────

@router.get("/feedback/export")
def export_feedback(user: dict = Depends(get_current_user)):
    """Export all feedback data for RLHF training. Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    rows = conn.execute("""
        SELECT mf.rating, mf.comment, m.role, m.content, m.structured_data,
               prev.content as user_query, c.title as chat_title
        FROM message_feedback mf
        JOIN messages m ON m.id = mf.message_id
        JOIN chats c ON c.id = m.chat_id
        LEFT JOIN messages prev ON prev.chat_id = m.chat_id AND prev.id = (
            SELECT MAX(p.id) FROM messages p WHERE p.chat_id = m.chat_id AND p.id < m.id AND p.role = 'user'
        )
        ORDER BY mf.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Helper: save message to chat ─────────────────────────────────────────

def save_message(chat_id: int, user_id: int, role: str, content: str,
                 mode: str = "structured", structured_data: dict | None = None,
                 latency_ms: int = 0) -> int:
    """Save a message and return its ID. Also updates chat.updated_at."""
    conn = get_connection()
    # Verify ownership
    chat = conn.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    sd_json = json.dumps(structured_data, ensure_ascii=False) if structured_data else None
    cur = conn.execute(
        "INSERT INTO messages (chat_id, role, content, mode, structured_data, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
        (chat_id, role, content, mode, sd_json, latency_ms),
    )
    conn.execute("UPDATE chats SET updated_at = datetime('now') WHERE id = ?", (chat_id,))
    conn.commit()
    msg_id = cur.lastrowid
    conn.close()
    return msg_id


def get_chat_history(chat_id: int, user_id: int, limit: int = 12) -> list[dict]:
    """Get last N messages from a chat for context injection."""
    conn = get_connection()
    chat = conn.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
    if not chat:
        conn.close()
        return []
    rows = conn.execute("""
        SELECT role, content FROM messages WHERE chat_id = ?
        ORDER BY id DESC LIMIT ?
    """, (chat_id, limit)).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
