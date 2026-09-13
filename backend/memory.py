import re
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

    prompt = f"""You resolve follow-up questions in a conversation into standalone questions.

Conversation history:
{history_text}

Follow-up question: {new_query}

Instructions:
- Rewrite the follow-up so it can be understood with no access to the history above.
- If it's already standalone, keep it unchanged.
- Pay close attention to ordinal or positional references such as "the first one", "the
  third project", "the second last thing", "the last one mentioned". Before answering,
  explicitly count the items in the relevant list from the history: count forward from the
  start for "first/second/third...", and count backward from the end for
  "last/second-last/second-to-last/third-last...". Double-check your count is correct
  before naming the item — a common mistake is picking the last item when asked for the
  second-last, so verify by counting on your fingers, not by guessing.
- Only resolve references using what is actually written in the history — never guess.

First, on 2-3 short scratch lines, show your counting/reasoning.
Then, on the FINAL line of your response, write ONLY the rewritten standalone question,
prefixed exactly with "STANDALONE:" and nothing else on that line.
"""

    try:
        raw = complete(prompt, temperature=0.0)
        match = re.search(r"STANDALONE:\s*(.+)", raw)
        if match:
            rewritten = match.group(1).strip()
        else:
            lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
            rewritten = lines[-1] if lines else new_query
        return rewritten.strip('"')
    except Exception as e:
        print(f"[memory] condense_query failed, falling back to raw query: {e}")
        return new_query