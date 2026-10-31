from backend import config, memory
from backend.chat_pipeline import DECLINE_MESSAGE

# A plain profile question: the project router ignores it, so it reaches the
# general RAG path (no GitHub calls).
PROFILE_Q = "What programming languages do you know?"


def _chat(client, query, session="s1"):
    return client.post("/chat", json={"session_id": session, "query": query})


# --------------------------------------------------------------------------
# basic endpoints / validation
# --------------------------------------------------------------------------
def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_rejects_empty_query(client):
    assert _chat(client, "").status_code == 400


def test_chat_rejects_whitespace_only_query(client):
    assert _chat(client, "   \n\t ").status_code == 400


def test_chat_rejects_query_over_max_length(client):
    resp = _chat(client, "x" * (config.MAX_QUERY_LENGTH + 1))
    assert resp.status_code == 422


def test_chat_requires_session_id_and_query(client):
    assert client.post("/chat", json={"query": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "s"}).status_code == 422


# --------------------------------------------------------------------------
# /chat behaviour (LLM mocked)
# --------------------------------------------------------------------------
def test_chat_streams_answer_and_persists_both_turns(client, mock_llm):
    mock_llm.when("You ARE Abhigya Narain", "I mostly work in Python and SQL.")

    resp = _chat(client, PROFILE_Q)

    assert resp.status_code == 200
    assert resp.text == "I mostly work in Python and SQL."
    assert [t["role"] for t in memory.get_history("s1")] == ["user", "assistant"]
    assert memory.get_history("s1")[1]["content"] == "I mostly work in Python and SQL."


def test_chat_answer_prompt_is_grounded_in_retrieved_chunks(client, mock_llm):
    _chat(client, "Which database stores the transcript data?")

    answer_prompt = mock_llm.prompts_containing("You ARE Abhigya Narain")[-1]
    assert "Qdrant" in answer_prompt          # the ScrubGPT fixture paragraph was retrieved


def test_out_of_scope_message_is_declined_without_calling_the_answer_model(client, mock_llm):
    mock_llm.when("scope gate", "NO")

    resp = _chat(client, "Write me a poem about clouds")

    assert resp.status_code == 200
    assert resp.text == DECLINE_MESSAGE
    assert mock_llm.prompts_containing("You ARE Abhigya Narain") == []
    history = memory.get_history("s1")
    assert [t["role"] for t in history] == ["user", "assistant"]
    assert history[1]["content"] == DECLINE_MESSAGE


def test_follow_up_goes_through_query_condensation_with_history(client, mock_llm):
    mock_llm.when("You ARE Abhigya Narain", "First answer.")
    _chat(client, PROFILE_Q)

    mock_llm.when("Rewrite a follow-up question",
                  "STANDALONE: Which SQL databases have you used?")
    resp = _chat(client, "tell me more")

    assert resp.status_code == 200
    condense_prompts = mock_llm.prompts_containing("Rewrite a follow-up question")
    assert len(condense_prompts) == 1
    assert PROFILE_Q in condense_prompts[0]                       # history was passed in
    # ...and the rewritten query is what reaches the answer prompt
    assert "Which SQL databases have you used?" in mock_llm.prompts_containing(
        "You ARE Abhigya Narain")[-1]


def test_list_projects_route_is_deterministic_and_links_every_repo(client, mock_llm):
    resp = _chat(client, "tell me about your projects")

    assert resp.status_code == 200
    for project in config.GITHUB_PROJECTS.values():
        assert f"https://github.com/{project['owner']}/{project['repo']}" in resp.text
    assert mock_llm.calls == []                                   # no LLM involved
    assert memory.get_active_project("s1") == "__all_projects__"


def test_clear_chat_wipes_the_session(client):
    _chat(client, PROFILE_Q)
    assert memory.get_history("s1")

    resp = client.post("/chat/clear", params={"session_id": "s1"})

    assert resp.status_code == 200 and resp.json() == {"status": "cleared"}
    assert memory.get_history("s1") == []


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
def _assert_limited_within(codes, limit, ok_codes):
    """The first rejection must be a 429 that arrives no later than request
    limit+1, and everything before it must have been a normal response. (The
    exact boundary isn't asserted: how slowapi counts hits for stacked
    @limiter.limit decorators is its own business; what matters here is that
    the cap is wired up and enforced.)"""
    assert 429 in codes[:limit + 1], f"no 429 within {limit + 1} requests: {codes}"
    first_429 = codes.index(429)
    assert first_429 >= 1
    assert all(c in ok_codes for c in codes[:first_429])


def test_chat_per_minute_limit_returns_429(client):
    """Whitespace queries are rejected with 400 *after* the limiter has counted
    them, so this exercises the real limit without touching the LLM."""
    limit = int(config.CHAT_RATE_LIMIT_SHORT.split("/")[0])
    codes = [_chat(client, "   ", session="rl").status_code for _ in range(limit + 1)]
    _assert_limited_within(codes, limit, ok_codes={400})


def test_match_per_minute_limit_returns_429(client):
    limit = int(config.MATCH_RATE_LIMIT_SHORT.split("/")[0])
    files = {"file": ("empty.txt", b"", "text/plain")}
    codes = [client.post("/match", files=files).status_code for _ in range(limit + 1)]
    _assert_limited_within(codes, limit, ok_codes={400, 422})


def test_rate_limit_is_tracked_per_route(client):
    """Burning the /chat budget must not block /health."""
    limit = int(config.CHAT_RATE_LIMIT_SHORT.split("/")[0])
    for _ in range(limit + 1):
        _chat(client, "   ", session="rl")
    assert client.get("/health").status_code == 200
