import json
import logging
import re

import config
import github_client
from backend.client import complete, complete_with_tools

logger = logging.getLogger(__name__)

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


def route_project_query(query: str, active_project: str | None) -> dict:
    """Decide whether this message should trigger a GitHub overview, a deep
    dive on the currently active project, or neither.
    Returns {"action": "overview" | "deep_dive" | "none", "project_key": str | None}.
    Defaults to "none" (falls through to the normal RAG path) on any failure
    or uncertainty -- a wrong GitHub lookup is worse than just answering normally.
    """
    project_names = list(config.GITHUB_PROJECTS.keys())
    if not project_names:
        return {"action": "none", "project_key": None}

    projects_text = "\n".join(f"- {name}" for name in project_names)

    prompt = f"""You are a router deciding whether a chat message should trigger a
GitHub repository lookup for a portfolio chatbot.

Known projects (with GitHub access configured):
{projects_text}

Currently active project in this conversation (the one last discussed in detail, or
"none" if no project is currently active): {active_project or "none"}

User's message: "{query}"

Decide exactly ONE action:
- "overview": the user is asking for more detail about one of the known projects
  (architecture, frameworks, how it works, tech stack, etc.) beyond a basic resume-level
  mention -- whether or not a project is currently active, as long as this message names
  or clearly refers to a *known* project.
- "deep_dive": there IS a currently active project, and this message is a confirmation
  ("yes", "sure", "go ahead") or a specific implementation-level follow-up (backend,
  endpoints, logic, code, how something was built) about THAT SAME active project.
- "none": anything else -- general resume questions, unrelated topics, or a mention too
  vague to justify a GitHub lookup, or a project that isn't in the known list above.

If the action is "overview" or "deep_dive", also give the exact project key, copied
character-for-character from the Known projects list above. If "none", project_key must
be null.

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{"action": "overview" | "deep_dive" | "none", "project_key": "<exact key from the list, or null>"}}
"""

    try:
        raw = complete(prompt, temperature=0.0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0) if match else raw)

        action = parsed.get("action")
        project_key = parsed.get("project_key")

        if action not in ("overview", "deep_dive", "none"):
            return {"action": "none", "project_key": None}
        if action != "none" and project_key not in config.GITHUB_PROJECTS:
            return {"action": "none", "project_key": None}

        return {"action": action, "project_key": project_key}
    except Exception as e:
        logger.warning(f"[project_agent] route_project_query failed, defaulting to none: {e}")
        return {"action": "none", "project_key": None}


def answer_overview(project_key: str, user_question: str) -> tuple[str, list[str]]:
    """Stage 1: read the project's README and answer from it alone, ending
    with an invitation to go deeper.
    """
    project = config.GITHUB_PROJECTS.get(project_key)
    if not project:
        return f'I don\'t have GitHub access configured for "{project_key}" yet.', []

    owner = project["owner"]
    repo = project["repo"]
    branch = project.get("branch", "main")
    repo_link = github_client.repo_url(owner, repo)

    result = github_client.get_readme(owner, repo, branch)
    if result is None:
        return (
            f"I found the repository for **{project_key}** but couldn't read its README. "
            f"You can browse the code directly here."
        ), [repo_link]

    readme_text, readme_link = result

    prompt = f"""You are answering a question about a specific software project, using
ONLY the README content below -- do not use outside knowledge, do not invent
frameworks, architecture details, or functionality the README doesn't actually state.

Project: {project_key}

README content:
{readme_text}

Question: {user_question}

Give a clear overview covering (only what the README actually supports): the tech
stack/frameworks used, the high-level architecture, and what the project does / how it
works. If the README doesn't cover something, say so rather than guessing.

End your answer by inviting the user to go deeper -- e.g. into backend implementation,
specific endpoints, or core logic -- if they're interested. Keep that invitation short
and natural, one sentence.
"""

    answer = complete(prompt, temperature=0.2)
    return answer, [repo_link, readme_link]


def answer_deep_dive(project_key: str, user_question: str) -> tuple[str, list[str]]:
    """Stage 2: a small bounded tool-calling loop. The model sees the repo's
    file list, chooses which file(s) to read via the read_file tool, and
    answers grounded only in what it actually read.
    """
    project = config.GITHUB_PROJECTS.get(project_key)
    if not project:
        return f'I don\'t have GitHub access configured for "{project_key}" yet.', []

    owner = project["owner"]
    repo = project["repo"]
    branch = project.get("branch", "main")
    repo_link = github_client.repo_url(owner, repo)

    file_list = github_client.list_files(owner, repo, branch)
    if not file_list:
        return (
            f"I couldn't list files in **{project_key}**'s repository right now. "
            f"You can browse it directly here."
        ), [repo_link]

    file_list_text = "\n".join(file_list[:300])  # cap in case of an unusually large repo

    system_prompt = f"""You are helping answer a detailed, implementation-level question
about the project "{project_key}". You have a read_file tool that fetches the exact
text content of any file in this repository, by path.

Here is the full list of files in the repository:
{file_list_text}

Decide which 1-3 files are actually relevant to the user's question (prefer specific
source files over config/lock files unless the question is literally about those), call
read_file on each, then answer the question grounded ONLY in what those files actually
contain. Do not guess at implementation details you haven't actually read. If nothing in
the file list looks relevant, say so honestly instead of picking randomly.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    tools = [READ_FILE_TOOL]
    read_links: list[str] = []

    for _ in range(config.DEEP_DIVE_MAX_ROUNDS):
        message = complete_with_tools(messages, tools)
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            final_answer = message.content or (
                "I wasn't able to put together an answer from the repository."
            )
            return final_answer, [repo_link] + read_links

        for tool_call in message.tool_calls:
            path = ""
            try:
                args = json.loads(tool_call.function.arguments)
                path = args.get("path", "")
            except (json.JSONDecodeError, AttributeError):
                pass

            result = github_client.get_file(owner, repo, path, branch) if path else None
            if result is None:
                tool_output = f"File not found or unreadable: {path}"
            else:
                file_text, file_link = result
                read_links.append(file_link)
                # keep any single file bounded so it can't blow the context budget
                tool_output = file_text[:6000]

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

    # exhausted rounds without a final plain-text answer
    return (
        f"I read through part of **{project_key}**'s code but couldn't finish putting "
        f"together a full answer. You can look directly here."
    ), [repo_link] + read_links
