import json
import logging
import re

from backend import config
import backend.github_client as github_client
from backend.client import client, complete, stream_chat, stream_messages

logger = logging.getLogger(__name__)

# Sentinel stored as the session's "active project" after a list_projects
# reply, so a short follow-up ("tell me more", "go deeper") with no specific
# project named is recognized as continuing that same general rundown
# instead of a fresh, unrelated mention. Never a real key in
# config.GITHUB_PROJECTS, so it can't collide with an actual project.
ALL_PROJECTS_KEY = "__all_projects__"


def complete_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = config.CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = config.TOOLCALL_MAX_TOKENS,
):
    """Single non-streaming tool-calling round trip.

    Used by the GitHub deep-dive loop (answer_deep_dive) to let the model
    choose which repository file(s) to read via `tools`. Returns the raw
    response message object -- not just its text -- since callers here need
    `.tool_calls` and `.model_dump()` to keep the agent loop going. This step
    can't stream: we need the complete tool_calls payload (if any) before we
    can act on it.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the full text content of one file from the project's GitHub "
            "repository, by its exact path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Exact file path within the repository, e.g. 'backend/main.py'",
                }
            },
            "required": ["path"],
        },
    },
}


def _fake_stream(text: str, chunk_size: int = 40):
    """Turn an already-complete string into a generator of chunks, so every
    project_agent function can return the same (generator, links) shape
    regardless of whether the answer came from a real LLM stream or a static
    message (missing repo config, missing README, etc.).
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _normalize(text: str) -> str:
    """Normalize user text for deterministic project/intent matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _project_profiles() -> dict[str, dict]:
    """Build the current project profiles from config + info.txt.

    config.py defines project identity/GitHub access; Data/info.txt supplies
    the current human-facing facts. This is intentionally loaded on demand so
    editing info.txt changes answers without duplicating descriptions in code.
    """
    try:
        with open(config.INFO_TXT_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def norm(value: str) -> str:
        return _normalize(value)

    profiles: dict[str, dict] = {}
    for key in config.PROJECT_KEYS:
        cfg = config.GITHUB_PROJECTS[key]
        display_name = cfg.get("display_name", key)
        aliases = [display_name, key, *cfg.get("aliases", [])]
        aliases = [a for a in aliases if a]

        matched = None
        for paragraph in paragraphs:
            normalized_paragraph = norm(paragraph)
            if any(f" {norm(alias)} " in f" {normalized_paragraph} " for alias in aliases):
                matched = paragraph
                break

        profiles[key] = {
            "display_name": display_name,
            "aliases": cfg.get("aliases", []),
            "summary": matched,
        }

    return profiles


def _project_matches() -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for key in config.PROJECT_KEYS:
        cfg = config.GITHUB_PROJECTS[key]
        aliases = [cfg.get("display_name", key), key, *cfg.get("aliases", [])]
        for alias in aliases:
            normalized = _normalize(alias)
            if normalized:
                matches.append((normalized, key))
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches


def _match_known_project(query: str) -> str | None:
    """Return a project only for an explicit configured name/alias.

    There is deliberately no fuzzy matching here. Fuzzy matching is what
    allows phrases like "resume parser" to drift into an unrelated project.
    """
    normalized = _normalize(query)
    padded = f" {normalized} "
    for alias, key in _project_matches():
        if alias and f" {alias} " in padded:
            return key
    return None


def _looks_like_project_list_query(query: str) -> bool:
    q = _normalize(query)
    patterns = (
        "tell me about his projects",
        "tell me about her projects",
        "tell me about your projects",
        "tell me about my projects",
        "what projects have you built",
        "what projects have you made",
        "what projects did you build",
        "what projects did you make",
        "what have you built",
        "what have you worked on",
        "show me your projects",
        "show me his projects",
        "list your projects",
        "list his projects",
        "list the projects",
        "all your projects",
        "all his projects",
        "your projects",
        "his projects",
        "project list",
        "portfolio projects",
        "project repos",
        "project repositories",
        "project links",
        "repository links",
        "repo links",
    )
    if any(p in q for p in patterns):
        return True

    # Catch natural variants such as "can you walk me through all the projects?"
    has_project_plural = bool(re.search(r"\bprojects\b", q))
    has_broad_word = bool(re.search(
        r"\b(all|every|each|what|which|list|show|tell|overview|portfolio)\b", q))
    has_repo_request = bool(
        re.search(r"\b(link|links|repo|repos|repository|repositories|url|urls)\b", q))
    has_project_term = bool(re.search(r"\b(project|projects|portfolio)\b", q))
    return (
        has_project_plural and has_broad_word and _match_known_project(
            query) is None
    ) or (
        has_repo_request and has_project_term and has_broad_word
        and _match_known_project(query) is None
    )


def _looks_like_deep_dive(query: str) -> bool:
    q = _normalize(query)
    return bool(re.search(
        r"\b("
        r"code|implementation|implement|architecture|endpoint|api|backend|frontend|"
        r"database|schema|authentication|oauth|jwt|logic|internals|internal|"
        r"function|class|file|repository|repo|deployment|docker|container|"
        r"how does|how did you build|how did you implement|walk me through|"
        r"in depth|in detail|under the hood|technical deep dive|deep dive"
        r")\b", q))


def _looks_like_short_continuation(query: str) -> bool:
    q = _normalize(query)
    return (
        len(q.split()) <= 8
        and bool(re.fullmatch(
            r"(yes|yeah|yep|sure|okay|ok|go ahead|continue|go on|"
            r"tell me more|more|go deeper|dive deeper|continue please|"
            r"explain more|what else|details|more details|tell me in detail)",
            q,
        ))
    )


def _looks_like_unknown_project_query(query: str) -> bool:
    q = _normalize(query)
    if _match_known_project(query):
        return False

    has_project_noun = bool(re.search(
        r"\b(project|app|application|model|chatbot|tool|converter|scraper|parser)\b",
        q,
    ))
    has_project_intent = bool(re.search(
        r"\b(tell me|what is|what's|how does|how did|explain|describe|"
        r"details about|information about|talk about|built|made|developed|worked on)\b",
        q,
    ))
    return has_project_noun and has_project_intent


def route_project_query(query: str, active_project: str | None) -> dict:
    """Deterministic project router.

    The set of real projects is small and known. Using explicit matching here
    is more reliable than asking an LLM to invent a route/key and then trying
    to recover from fuzzy matches.
    """
    if not query.strip():
        return {"action": "none", "project_key": None}

    # Broad portfolio/project questions always use the authoritative registry.
    if _looks_like_project_list_query(query):
        return {"action": "list_projects", "project_key": None}

    # A generic continuation after the project list remains a list operation.
    if active_project == ALL_PROJECTS_KEY and _looks_like_short_continuation(query):
        return {"action": "list_projects", "project_key": None}

    matched_project = _match_known_project(query)

    # Short confirmations after discussing a single project stay attached to it.
    if matched_project is None and active_project in config.GITHUB_PROJECTS and _looks_like_short_continuation(query):
        return {
            "action": "deep_dive",
            "project_key": active_project,
        }

    if matched_project is not None:
        if _looks_like_deep_dive(query):
            return {"action": "deep_dive", "project_key": matched_project}
        return {"action": "overview", "project_key": matched_project}

    # Refuse to let the normal RAG model turn an unknown project-like phrase
    # into a made-up project. main.py gives a grounded correction instead.
    if _looks_like_unknown_project_query(query):
        return {
            "action": "unknown_project",
            "project_key": None,
        }

    return {"action": "none", "project_key": None}


def answer_all_projects():
    """Return the current portfolio directly from config + info.txt.

    The LLM is intentionally not used to decide which projects exist. This is
    a deterministic enumeration of the configured projects, with each
    description read from the current info.txt content.
    """
    profiles = _project_profiles()
    lines = ["I currently have these projects in my profile:", ""]

    for i, key in enumerate(config.PROJECT_KEYS, start=1):
        profile = profiles[key]
        summary = profile["summary"]
        if summary:
            lines.append(f"{i}. **{profile['display_name']}** — {summary}")
        else:
            lines.append(
                f"{i}. **{profile['display_name']}** — I have this project configured, "
                "but its description is not currently present in info.txt."
            )
        lines.append("")

    lines.append(
        "Only projects configured in config.py are treated as portfolio projects. "
        "Features, skills, frameworks, and capabilities mentioned in info.txt "
        "are not automatically promoted into standalone projects."
    )

    links = []
    for key in config.PROJECT_KEYS:
        project = config.GITHUB_PROJECTS.get(key)
        if project and project.get("owner") and project.get("repo"):
            links.append(github_client.repo_url(
                project["owner"], project["repo"]))

    return _fake_stream("\n".join(lines)), links


def answer_unknown_project():
    """Grounded response when a user names something that is not configured."""
    profiles = _project_profiles()
    names = ", ".join(
        f"**{profiles[key]['display_name']}**" for key in config.PROJECT_KEYS
    )
    text = (
        "I don't have a separate project by that name in my profile. "
        f"The projects currently configured are {names}. "
        "Project features are not treated as separate projects unless they are "
        "explicitly added to the project configuration."
    )
    return _fake_stream(text), []


def answer_overview(project_key: str, user_question: str):
    """Stage 1: read the project's README and answer from it alone -- a
    crisp, first-person introduction (as Abhigya, since this chatbot speaks
    on his behalf) ending with a short invitation to go deeper.

    Returns (generator[str], links). The generator yields real streamed
    tokens as the LLM generates them, so the reply appears live instead of
    all at once.
    """
    project = config.GITHUB_PROJECTS.get(project_key)
    if not project:
        return _fake_stream(f'I don\'t have GitHub access wired up for "{project_key}" yet.'), []

    owner = project["owner"]
    repo = project["repo"]
    branch = project.get("branch", "main")
    repo_link = github_client.repo_url(owner, repo)

    profile_summary = _project_profiles().get(project_key, {}).get("summary")
    if not profile_summary:
        profile_summary = "No matching project description was found in info.txt."

    result = github_client.get_readme(owner, repo, branch)
    if result is None:
        text = (
            f"**{config.GITHUB_PROJECTS[project_key].get('display_name', project_key)}** — "
            f"{profile_summary} "
            "I couldn't read the GitHub README right now, so I won't add any "
            "unverified implementation details."
        )
        return _fake_stream(text), [repo_link]

    readme_text, readme_link = result

    prompt = f"""You are Abhigya Narain, briefly introducing one of your own projects.
Use the AUTHORITATIVE PROFILE SUMMARY below as the source of truth for what the project
is. Use the README only as supporting detail. Do not introduce a new project name,
feature, framework, or capability that is absent from either source.

AUTHORITATIVE PROFILE SUMMARY:
{profile_summary}

Project key:
{project_key}

README content:
{readme_text}

Question: {user_question}

Give a solid overview -- more than a one-liner: what the project does, the concrete tech
stack/frameworks it actually uses, the high-level approach, and one or two notable
technical details or design choices that make it worth mentioning, only what the README
actually supports. Structure it as a short paragraph (2-3 sentences) followed by 2-3
short bullet points of specifics -- detailed and easy to scan, not one long dense block
of text. Paraphrase in your own words throughout; never quote the README verbatim or
mirror its structure (no copied headers, no copied sentence order, no bullet-for-bullet
dump of the README itself). If the README doesn't cover something relevant to the
question, say so plainly rather than guessing.

End with one short, natural sentence inviting a deeper technical dive if they want it.
"""

    def _gen():
        yield from stream_chat(
            prompt, model=config.CHAT_MODEL, temperature=0.2,
            max_tokens=config.PROJECT_MAX_TOKENS,
        )

    return _gen(), [repo_link, readme_link]


def answer_deep_dive(project_key: str, user_question: str):
    """Stage 2: a small bounded tool-calling loop. The model sees the repo's
    file list, chooses which file(s) to read via the read_file tool
    (up to config.DEEP_DIVE_MAX_ROUNDS rounds, reading more if the first
    file(s) don't fully answer the question), then a final dedicated
    streaming call turns whatever it found into a crisp, first-person answer
    (as Abhigya) so the reply appears live.

    Returns (generator[str], links).
    """
    project = config.GITHUB_PROJECTS.get(project_key)
    if not project:
        return _fake_stream(f'I don\'t have GitHub access wired up for "{project_key}" yet.'), []

    owner = project["owner"]
    repo = project["repo"]
    branch = project.get("branch", "main")
    repo_link = github_client.repo_url(owner, repo)

    file_list = github_client.list_files(owner, repo, branch)
    if not file_list:
        text = (
            f"I couldn't list files in **{project_key}**'s repository right now. "
            f"You can browse it directly here."
        )
        return _fake_stream(text), [repo_link]

    # cap in case of an unusually large repo
    file_list_text = "\n".join(file_list[:300])

    profile_summary = _project_profiles().get(project_key, {}).get(
        "summary") or "No matching project description was found in info.txt."

    system_prompt = f"""You are Abhigya Narain, digging into your own project "{project_key}"
to answer a detailed, implementation-level question about it.

AUTHORITATIVE PROFILE SUMMARY:
{profile_summary}

You have a read_file tool
that fetches the exact text content of any file in this repository, by path.

Here is the full list of files in the repository:
{file_list_text}

Decide which file(s) are actually relevant to the user's question (prefer specific
source files over config/lock files unless the question is literally about those), and
call read_file on each. Read enough to actually answer -- if the first file you check
doesn't fully answer it, read another relevant one rather than guessing or stopping
early. Do not invent implementation details you haven't actually read. If nothing in the
file list looks relevant, say so honestly instead of picking randomly.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    tools = [READ_FILE_TOOL]
    read_links: list[str] = []

    for _ in range(config.DEEP_DIVE_MAX_ROUNDS):
        message = complete_with_tools(messages, tools)

        if not message.tool_calls:
            # model didn't ask for any more files -- stop researching here.
            # We deliberately don't keep this draft; the dedicated final call
            # below regenerates the answer as a real, live stream instead.
            break

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            path = ""
            try:
                args = json.loads(tool_call.function.arguments)
                path = args.get("path", "")
            except (json.JSONDecodeError, AttributeError):
                pass

            result = github_client.get_file(
                owner, repo, path, branch) if path else None
            if result is None:
                tool_output = f"File not found or unreadable: {path}"
            else:
                file_text, file_link = result
                read_links.append(file_link)
                # keep any single file bounded so it can't blow the context budget
                tool_output = file_text[:config.DEEP_DIVE_FILE_CHAR_CAP]

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )
    else:
        # exhausted every round still asking for files
        logger.info(
            f"[project_agent] deep dive on {project_key} exhausted "
            f"{config.DEEP_DIVE_MAX_ROUNDS} rounds without finishing research")

    # Final, dedicated streaming call: force a plain-text answer now (no more
    # tool calls), grounded only in whatever was actually read above, kept
    # detailed but tight and in Abhigya's own voice -- explicitly not a code
    # dump, since the model just read raw file content and left unchecked
    # tends to quote or paste chunks of it back verbatim.
    nudge = (
        "Based only on what you actually read above, give your final answer now -- no "
        "more tool calls. Speak as Abhigya, in first person, and actually answer the "
        "specific question using real detail from what you read (the actual function/"
        "endpoint/class names, the real flow, the specific design choice) -- don't stay "
        "generic or vague. Do NOT paste raw code blocks or reproduce file contents "
        "verbatim -- describe what the code does in plain English instead; a single "
        "short inline reference like a function name is fine, a multi-line snippet is "
        "not. Structure the answer as a short paragraph or a tight, well-organized "
        "bullet list -- detailed, but not a long wall of text or an essay. If you "
        "didn't find anything relevant, say so honestly instead of guessing."
    )
    if messages[-1]["role"] == "user":
        messages[-1] = {
            "role": "user",
            "content": messages[-1]["content"] + "\n\n" + nudge,
        }
    else:
        messages.append({"role": "user", "content": nudge})

    def _gen():
        yield from stream_messages(
            messages, model=config.CHAT_MODEL, temperature=0.2,
            max_tokens=config.DEEP_DIVE_ANSWER_MAX_TOKENS,
        )

    return _gen(), [repo_link] + read_links
