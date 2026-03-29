"""SQLite database setup with aiosqlite for async operations."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "labus_rag.db"


def get_db_path() -> str:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return str(DB_PATH)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            email       TEXT    DEFAULT '',
            hashed_pw   TEXT    NOT NULL,
            full_name   TEXT    DEFAULT '',
            role        TEXT    DEFAULT 'user',
            is_active   INTEGER DEFAULT 1,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title       TEXT    DEFAULT 'Новый чат',
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role        TEXT    NOT NULL CHECK(role IN ('user','assistant')),
            content     TEXT    NOT NULL,
            mode        TEXT    DEFAULT 'structured',
            structured_data TEXT DEFAULT NULL,
            latency_ms  INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS message_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            rating      INTEGER NOT NULL CHECK(rating IN (-1, 1)),
            comment     TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT (datetime('now')),
            UNIQUE(message_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS feedback_lessons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id     INTEGER NOT NULL REFERENCES message_feedback(id),
            user_query      TEXT    NOT NULL,
            query_embedding BLOB    NOT NULL,
            direction       TEXT    DEFAULT '',
            lesson_text     TEXT    NOT NULL,
            rating          INTEGER NOT NULL,
            is_active       INTEGER DEFAULT 1,
            match_count     INTEGER DEFAULT 0,
            created_at      TEXT    DEFAULT (datetime('now')),
            UNIQUE(feedback_id)
        );

        CREATE TABLE IF NOT EXISTS feedback_rules (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_text       TEXT    NOT NULL,
            direction       TEXT    DEFAULT '',
            priority        INTEGER DEFAULT 0,
            source_ids      TEXT    DEFAULT '[]',
            is_active       INTEGER DEFAULT 1,
            created_at      TEXT    DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_message ON message_feedback(message_id);
        CREATE INDEX IF NOT EXISTS idx_fl_active ON feedback_lessons(is_active);
    """)
    conn.commit()
    conn.close()
