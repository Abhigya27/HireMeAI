import sqlite3
from contextlib import contextmanager

import config
from client import complete

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_turn(session_id: str, role: str, content: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def get_history(session_id: str, limit: int = config.MAX_HISTORY_TURNS) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def clear_history(session_id: str):
    with _connect() as conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.commit()


def condense_query(session_id: str, new_query: str) -> str:
    """Rewrite a follow-up question into a standalone question using chat history.
    This is what makes retrieval history-aware without needing LangChain: we simply
    ask the LLM to resolve pronouns/references before we embed and search.
    Skips the extra Groq call entirely if there's no history yet.
    """
    history = get_history(session_id)
    if not history:
        return new_query

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)

    prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up
as a standalone question that can be understood without the history. If the follow-up
question is already standalone, return it unchanged. Only output the rewritten question,
nothing else, no quotes, no preamble.

Conversation history:
{history_text}

Follow-up question: {new_query}

Standalone question:"""

    try:
        rewritten = complete(prompt, temperature=0.0)
        return rewritten.strip().strip('"')
    except Exception as e:
        print(f"[memory] condense_query failed, falling back to raw query: {e}")
        return new_query