import sqlite3

from backend import config, memory


def test_new_session_starts_empty():
    assert memory.get_history("brand-new") == []
    assert memory.get_active_project("brand-new") is None


def test_turns_are_persisted_to_sqlite_in_order():
    memory.save_turn("s1", "user", "hi")
    memory.save_turn("s1", "assistant", "hello")
    memory.save_turn("s1", "user", "what do you build?")

    # read straight from the DB file, bypassing the module under test
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM turns WHERE session_id='s1' ORDER BY id").fetchall()
    conn.close()
    assert rows == [("user", "hi"), ("assistant", "hello"), ("user", "what do you build?")]

    assert [t["content"] for t in memory.get_history("s1")] == [
        "hi", "hello", "what do you build?"]


def test_get_history_limit_returns_most_recent_turns_in_chronological_order():
    for i in range(5):
        memory.save_turn("s1", "user", f"m{i}")
    assert [t["content"] for t in memory.get_history("s1", limit=2)] == ["m3", "m4"]


def test_sessions_are_isolated():
    memory.save_turn("a", "user", "from a")
    memory.save_turn("b", "user", "from b")
    assert [t["content"] for t in memory.get_history("a")] == ["from a"]
    assert [t["content"] for t in memory.get_history("b")] == ["from b"]


def test_active_project_roundtrip_and_overwrite():
    memory.set_active_project("s1", "MediaShare")
    assert memory.get_active_project("s1") == "MediaShare"

    memory.set_active_project("s1", "ScrubGPT")      # upsert, not duplicate row
    assert memory.get_active_project("s1") == "ScrubGPT"

    memory.set_active_project("s1", None)
    assert memory.get_active_project("s1") is None


def test_active_project_is_per_session():
    memory.set_active_project("a", "MediaShare")
    assert memory.get_active_project("b") is None


def test_clear_history_removes_turns_and_active_project_for_that_session_only():
    memory.save_turn("a", "user", "x")
    memory.save_turn("b", "user", "y")
    memory.set_active_project("a", "MediaShare")

    memory.clear_history("a")

    assert memory.get_history("a") == []
    assert memory.get_active_project("a") is None
    assert len(memory.get_history("b")) == 1


# --------------------------------------------------------------------------
# query condensation
# --------------------------------------------------------------------------
def test_condense_skips_llm_when_there_is_no_history(mock_llm):
    assert memory.condense_query("fresh", "What is MediaShare?") == "What is MediaShare?"
    assert mock_llm.calls == []


def test_condense_sends_conversation_history_to_llm_and_uses_rewrite(mock_llm):
    memory.save_turn("s1", "user", "Tell me about MediaShare")
    memory.save_turn("s1", "assistant", "MediaShare is a media sharing app.")
    mock_llm.when("Rewrite a follow-up question",
                  "STANDALONE: What technologies does MediaShare use?")

    result = memory.condense_query("s1", "what tech does it use?")

    assert result == "What technologies does MediaShare use?"
    assert len(mock_llm.calls) == 1
    prompt = mock_llm.calls[0]["prompt"]
    assert "user: Tell me about MediaShare" in prompt          # history was included
    assert "assistant: MediaShare is a media sharing app." in prompt
    assert "what tech does it use?" in prompt                  # the follow-up itself
    assert mock_llm.calls[0]["stream"] is False


def test_condense_uses_last_line_when_marker_missing(mock_llm):
    memory.save_turn("s1", "user", "hi")
    mock_llm.when("Rewrite a follow-up question", "Some preamble\nWhat is ScrubGPT?")
    assert memory.condense_query("s1", "and that one?") == "What is ScrubGPT?"


def test_condense_falls_back_to_raw_query_when_llm_fails(mock_llm):
    memory.save_turn("s1", "user", "hi")
    mock_llm.error = RuntimeError("groq down")
    assert memory.condense_query("s1", "and that one?") == "and that one?"
