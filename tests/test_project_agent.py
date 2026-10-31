import json

import pytest

from backend import config, project_agent
from backend.project_agent import AgentTrace, ALL_PROJECTS_KEY, answer_deep_dive, route_project_query
from conftest import make_plain_message, make_tool_call_message


# --------------------------------------------------------------------------
# deterministic router
# --------------------------------------------------------------------------
def test_router_project_list_query():
    assert route_project_query("tell me about your projects", None)[
        "action"] == "list_projects"


def test_router_named_project_without_technical_terms_is_overview():
    r = route_project_query("Tell me about MediaShare", None)
    assert r == {"action": "overview", "project_key": "MediaShare"}


def test_router_named_project_with_technical_terms_is_deep_dive():
    r = route_project_query(
        "How does ScrubGPT store embeddings in its vector database?", None)
    assert r == {"action": "deep_dive", "project_key": "ScrubGPT"}


def test_router_resolves_configured_alias():
    r = route_project_query("show me the code of the media sharing app", None)
    assert r["project_key"] == "MediaShare"


def test_router_pronoun_technical_question_means_hiremeai():
    r = route_project_query("what retriever do you use?", None)
    assert r == {"action": "deep_dive", "project_key": "HireMeAI"}


def test_router_plain_profile_question_is_not_a_project_route():
    assert route_project_query("what are your skills?", None)[
        "action"] == "none"
    assert route_project_query("   ", None)["action"] == "none"


def test_router_short_confirmation_continues_the_active_project():
    r = route_project_query("yes", "MediaShare")
    assert r == {"action": "deep_dive", "project_key": "MediaShare"}


def test_router_short_continuation_after_project_list_stays_a_list():
    assert route_project_query("tell me more", ALL_PROJECTS_KEY)[
        "action"] == "list_projects"


def test_router_does_not_invent_a_project_from_a_feature_name():
    assert route_project_query("tell me about your resume parser", None)[
        "action"] == "unknown_project"


def test_all_projects_answer_links_every_configured_repo():
    gen, links = project_agent.answer_all_projects()
    text = "".join(gen)
    assert len(links) == len(config.PROJECT_KEYS)
    for project in config.GITHUB_PROJECTS.values():
        assert project["repo"] in " ".join(links)
    assert "MediaShare" in text


# --------------------------------------------------------------------------
# deep-dive instrumentation (AgentTrace)
# --------------------------------------------------------------------------
@pytest.fixture
def fake_github(monkeypatch):
    """Fake repo: three files; 'missing.py' is listed by nobody and unreadable."""
    files = {"backend/rag.py": "RAG CODE",
             "backend/main.py": "MAIN CODE", "README.md": "# readme"}

    monkeypatch.setattr(project_agent.github_client, "list_files",
                        lambda owner, repo, branch="main": list(files))

    def get_file(owner, repo, path, branch="main"):
        if path not in files:
            return None
        return files[path], f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"

    monkeypatch.setattr(project_agent.github_client, "get_file", get_file)
    return files


def test_deep_dive_trace_records_tool_calls_files_rounds_and_timing(fake_github, mock_llm):
    mock_llm.tool_script = [
        make_tool_call_message("backend/rag.py"),
        make_tool_call_message("backend/main.py", "missing.py"),
        make_plain_message("enough research"),
    ]
    mock_llm.when("Based only on what you actually read", "Final answer.")

    trace = AgentTrace()
    gen, links = answer_deep_dive(
        "HireMeAI", "How does retrieval work?", trace=trace)
    # not finished until the stream is consumed
    assert trace.wall_clock_seconds is None
    text = "".join(gen)

    assert text == "Final answer."
    assert trace.project_key == "HireMeAI" and trace.outcome == "answered"
    assert trace.num_tool_calls == 3
    assert trace.files_read == ["backend/rag.py", "backend/main.py"]
    assert trace.files_failed == ["missing.py"]
    assert trace.research_rounds_used == 3
    assert trace.exhausted_rounds is False
    assert 0 <= trace.research_seconds <= trace.wall_clock_seconds
    # answer links: repo + one per successfully-read file
    assert len(links) == 3 and links[0].endswith("/HireMeAI")
    assert any(l.endswith("backend/rag.py") for l in links)


def test_deep_dive_trace_flags_exhausted_rounds(fake_github, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "DEEP_DIVE_MAX_ROUNDS", 2)
    mock_llm.tool_script = [make_tool_call_message(
        "backend/rag.py") for _ in range(5)]

    trace = AgentTrace()
    gen, _links = answer_deep_dive("HireMeAI", "q", trace=trace)
    "".join(gen)

    assert trace.exhausted_rounds is True
    assert trace.research_rounds_used == 2
    assert trace.num_tool_calls == 2


def test_deep_dive_with_no_tool_calls_still_produces_trace(fake_github, mock_llm):
    trace = AgentTrace()
    gen, _ = answer_deep_dive("HireMeAI", "q", trace=trace)
    "".join(gen)
    assert trace.num_tool_calls == 0 and trace.files_read == []
    assert trace.research_rounds_used == 1


def test_deep_dive_trace_when_repo_listing_fails(monkeypatch):
    monkeypatch.setattr(project_agent.github_client,
                        "list_files", lambda *a, **k: [])
    trace = AgentTrace()
    gen, links = answer_deep_dive("HireMeAI", "q", trace=trace)
    "".join(gen)
    assert trace.outcome == "no_file_list" and trace.wall_clock_seconds is not None
    assert len(links) == 1


def test_deep_dive_trace_for_unconfigured_project():
    trace = AgentTrace()
    gen, links = answer_deep_dive("NotAProject", "q", trace=trace)
    assert "don't have GitHub access" in "".join(gen)
    assert trace.outcome == "no_project" and links == []


def test_trace_is_json_serialisable_and_trace_arg_is_optional(fake_github):
    # old call signature still works
    gen, _ = answer_deep_dive("HireMeAI", "q")
    "".join(gen)
    json.dumps(AgentTrace(project_key="x", files_read=["a"]).to_dict())
