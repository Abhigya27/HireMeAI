import re
import sqlite3
from contextlib import contextmanager

from backend import config
from backend.client import complete

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    active_project TEXT
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
        conn.executescript(SCHEMA)
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
        conn.execute(
            "DELETE FROM session_state WHERE session_id = ?", (session_id,))
        conn.commit()


def get_active_project(session_id: str) -> str | None:
    """The GitHub project (if any) that route_project_query last flagged as
    the subject of an "overview", so a later confirmation/follow-up message
    ("yes", "tell me about the backend") can be recognized as a deep-dive on
    that same project rather than a fresh, unrelated mention.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT active_project FROM session_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row[0] if row else None


def set_active_project(session_id: str, project_key: str | None):
    """Upsert this session's active project. Called after an "overview"
    reply so the immediately following turn is eligible for "deep_dive".
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO session_state (session_id, active_project)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET active_project = excluded.active_project
            """,
            (session_id, project_key),
        )
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

    prompt = f"""Rewrite a follow-up question into a standalone question using ONLY the
conversation history below.

Conversation history:
{history_text}

Follow-up question: {new_query}

Rules:
- If the follow-up is already standalone, return it unchanged.
- Resolve pronouns, ordinals, and references such as "the first one", "the third project",
  "the last one", or "the second-last thing" using ONLY what is explicitly written in the
  history.
- Count list items carefully before resolving ordinal references.
- Never invent a project, feature, technology, or fact.
- Do not provide an explanation or reasoning trace.

Return ONLY the rewritten standalone question, prefixed exactly with:
STANDALONE:
"""

    try:
        raw = complete(prompt, temperature=0.0,
                        max_tokens=config.CONDENSE_MAX_TOKENS)
        match = re.search(r"STANDALONE:\s*(.+)", raw)
        if match:
            rewritten = match.group(1).strip()
        else:
            lines = [line.strip()
                     for line in raw.strip().splitlines() if line.strip()]
            rewritten = lines[-1] if lines else new_query
        return rewritten.strip('"')
    except Exception as e:
        print(
            f"[memory] condense_query failed, falling back to raw query: {e}")
        return new_query