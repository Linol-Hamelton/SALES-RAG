"""Feedback-aware learning: stores lessons from user feedback and injects them into generation context.

Architecture:
- feedback_lessons table: processed feedback entries with pre-computed query embeddings
- feedback_rules table: admin-curated permanent rules distilled from feedback patterns
- In-memory numpy matrix for fast cosine similarity search (<1ms for 10k entries)
- Lessons are injected into LLM context as "RLHF instructions" before generation
"""
import struct
import numpy as np
from app.database import get_connection
from app.utils.logging import get_logger

logger = get_logger("feedback_store")

EMBEDDING_DIM = 1024  # BGE-M3 dimension
SIMILARITY_THRESHOLD = 0.45
MAX_LESSONS_PER_QUERY = 3
MAX_RULES_PER_QUERY = 5


def _serialize_embedding(vec: list[float]) -> bytes:
    """Serialize float list to compact binary for SQLite BLOB."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize BLOB back to numpy array."""
    n = len(blob) // 4  # float32 = 4 bytes
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


class FeedbackStore:
    """Manages feedback lessons and rules with in-memory embedding cache."""

    def __init__(self):
        self._embeddings: np.ndarray | None = None  # (N, dim)
        self._lessons: list[dict] = []  # parallel list
        self._loaded = False

    def load(self):
        """Load all active lessons from SQLite into memory."""
        conn = get_connection()
        rows = conn.execute("""
            SELECT id, feedback_id, user_query, query_embedding, direction,
                   lesson_text, rating, match_count
            FROM feedback_lessons WHERE is_active = 1
        """).fetchall()
        conn.close()

        self._lessons = []
        embeddings = []
        for r in rows:
            self._lessons.append({
                "id": r["id"],
                "feedback_id": r["feedback_id"],
                "user_query": r["user_query"],
                "direction": r["direction"],
                "lesson_text": r["lesson_text"],
                "rating": r["rating"],
                "match_count": r["match_count"],
            })
            embeddings.append(_deserialize_embedding(r["query_embedding"]))

        if embeddings:
            self._embeddings = np.vstack(embeddings)  # (N, dim)
        else:
            self._embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        self._loaded = True
        logger.info("FeedbackStore loaded", lessons=len(self._lessons))

    def add_lesson(self, feedback_id: int, user_query: str, query_embedding: list[float],
                   direction: str, comment: str, rating: int) -> int | None:
        """Create a lesson from feedback comment. Returns lesson ID or None if skipped."""
        if not comment or not comment.strip():
            return None

        # Distill lesson text
        if rating == -1:
            lesson_text = f"НЕ ДЕЛАЙ ТАК: {comment.strip()}"
        else:
            lesson_text = f"ДЕЛАЙ ТАК: {comment.strip()}"

        emb_blob = _serialize_embedding(query_embedding)

        conn = get_connection()
        try:
            cur = conn.execute("""
                INSERT INTO feedback_lessons (feedback_id, user_query, query_embedding,
                    direction, lesson_text, rating)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (feedback_id, user_query, emb_blob, direction, lesson_text, rating))
            conn.commit()
            lesson_id = cur.lastrowid
        except Exception as e:
            # UNIQUE constraint on feedback_id — update existing
            conn.execute("""
                UPDATE feedback_lessons SET lesson_text = ?, rating = ?,
                    query_embedding = ?, direction = ?
                WHERE feedback_id = ?
            """, (lesson_text, rating, emb_blob, direction, feedback_id))
            conn.commit()
            row = conn.execute("SELECT id FROM feedback_lessons WHERE feedback_id = ?",
                               (feedback_id,)).fetchone()
            lesson_id = row["id"] if row else None
        conn.close()

        # Update in-memory cache
        emb_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        lesson_meta = {
            "id": lesson_id,
            "feedback_id": feedback_id,
            "user_query": user_query,
            "direction": direction,
            "lesson_text": lesson_text,
            "rating": rating,
            "match_count": 0,
        }

        # Check if already in cache (update case)
        existing_idx = next(
            (i for i, l in enumerate(self._lessons) if l["feedback_id"] == feedback_id),
            None
        )
        if existing_idx is not None:
            self._lessons[existing_idx] = lesson_meta
            self._embeddings[existing_idx] = emb_np[0]
        else:
            self._lessons.append(lesson_meta)
            if self._embeddings is not None and self._embeddings.shape[0] > 0:
                self._embeddings = np.vstack([self._embeddings, emb_np])
            else:
                self._embeddings = emb_np

        logger.info("Lesson added", lesson_id=lesson_id, rating=rating,
                     query=user_query[:50])
        return lesson_id

    def find_relevant(self, query_embedding: list[float], direction: str = "",
                      top_k: int = MAX_LESSONS_PER_QUERY) -> list[dict]:
        """Find top-K relevant lessons by cosine similarity."""
        if not self._loaded or self._embeddings is None or self._embeddings.shape[0] == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        # Cosine similarity (embeddings are already normalized by BGE-M3)
        scores = self._embeddings @ q_vec

        # Direction boost: +0.1 for same direction
        if direction:
            for i, lesson in enumerate(self._lessons):
                if lesson["direction"] and direction.lower() in lesson["direction"].lower():
                    scores[i] += 0.1

        # Filter by threshold and get top-K
        mask = scores >= SIMILARITY_THRESHOLD
        if not mask.any():
            return []

        indices = np.where(mask)[0]
        top_indices = indices[np.argsort(scores[indices])[::-1]][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self._lessons[idx],
                "similarity": float(scores[idx]),
            })

        # Increment match_count in DB (async-safe since workers=1)
        if results:
            self._increment_match_counts([r["id"] for r in results])

        return results

    def get_active_rules(self, direction: str = "") -> list[dict]:
        """Get curated rules applicable to this query context."""
        conn = get_connection()
        if direction:
            rows = conn.execute("""
                SELECT id, rule_text, direction, priority FROM feedback_rules
                WHERE is_active = 1 AND (direction = '' OR direction = ?)
                ORDER BY priority DESC LIMIT ?
            """, (direction, MAX_RULES_PER_QUERY)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, rule_text, direction, priority FROM feedback_rules
                WHERE is_active = 1
                ORDER BY priority DESC LIMIT ?
            """, (MAX_RULES_PER_QUERY,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _increment_match_counts(self, lesson_ids: list[int]):
        """Increment match_count for matched lessons."""
        conn = get_connection()
        for lid in lesson_ids:
            conn.execute("UPDATE feedback_lessons SET match_count = match_count + 1 WHERE id = ?",
                         (lid,))
            # Also update in-memory
            for lesson in self._lessons:
                if lesson["id"] == lid:
                    lesson["match_count"] = lesson.get("match_count", 0) + 1
        conn.commit()
        conn.close()


def build_feedback_context(feedback_store: "FeedbackStore",
                           query_embedding: list[float] | None = None,
                           direction: str = "") -> str:
    """Build feedback context string for injection into LLM prompt.

    P10/A6: rules (Tier 2) и lessons (Tier 1) расцеплены. `query_embedding` может
    быть None — тогда собираются только rules из БД (они не требуют эмбеддинга).
    Подсчёт правил/уроков всегда логируется, чтобы silent-fail был наблюдаемым.
    """
    blocks = []
    rules_count = 0
    lessons_count = 0

    # Tier 2: Curated rules (always first, most authoritative) — не требуют embedding.
    try:
        rules = feedback_store.get_active_rules(direction)
    except Exception as e:
        logger.warning("feedback_rules fetch failed", error=str(e))
        rules = []
    if rules:
        rule_lines = [f"- {r['rule_text']}" for r in rules]
        blocks.append("ПРАВИЛА ОТВЕТА (обязательные):\n" + "\n".join(rule_lines))
        rules_count = len(rules)

    # Tier 1: Auto-matched lessons — только когда есть query_embedding.
    if query_embedding is not None:
        try:
            lessons = feedback_store.find_relevant(query_embedding, direction)
        except Exception as e:
            logger.warning("feedback_lessons lookup failed", error=str(e))
            lessons = []
        if lessons:
            lesson_lines = []
            for l in lessons:
                prefix = "ОШИБКА В ПРОШЛОМ" if l["rating"] == -1 else "ХОРОШИЙ ПРИМЕР"
                lesson_lines.append(
                    f"[{prefix}, сходство {l['similarity']:.0%}] "
                    f"Запрос «{l['user_query'][:80]}»: {l['lesson_text'][:300]}"
                )
            blocks.append(
                "ОПЫТ ПРОШЛЫХ ОТВЕТОВ (учти при формировании ответа):\n"
                + "\n".join(lesson_lines)
            )
            lessons_count = len(lessons)

    logger.info("feedback_context_built",
                rules_count=rules_count,
                lessons_count=lessons_count,
                direction=direction or "",
                has_embedding=query_embedding is not None)

    return "\n\n".join(blocks)
