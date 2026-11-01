"""The general-profile RAG path of /chat, factored out of main.py.

main.py and the eval scripts (eval/faithfulness_eval.py, eval/retrieval_eval.py)
call the same function, so what gets evaluated is exactly what ships -- no
second copy of the pipeline that can drift out of sync.

Order of operations is unchanged from the original inline code in main.py:
scope gate -> condense follow-up -> retrieve -> build prompt (with history).
Persisting turns stays in main.py (it has to happen around the streaming).
"""
from dataclasses import dataclass, field

from backend import config, memory
from backend.client import stream_chat
from backend.rag import (
    answer_direct_profile_fact,
    build_answer_prompt,
    get_index,
    is_relevant_query,
    retrieve,
)

DECLINE_MESSAGE = (
    "I'm just here to talk about myself — my background, skills, projects, and "
    "experience. Happy to help with any of that!"
)


@dataclass
class RagTurn:
    declined: bool                      # True -> the scope gate rejected the message
    standalone_query: str               # follow-up resolved into a standalone question
    # retrieved context (empty if declined)
    chunks: list[str] = field(default_factory=list)
    prompt: str | None = None           # final LLM prompt (None if declined)

    def stream(self):
        """Yield the reply text in chunks: the canned decline, or the live LLM stream."""
        if self.declined:
            for i in range(0, len(DECLINE_MESSAGE), 40):
                yield DECLINE_MESSAGE[i:i + 40]
            return
        yield from stream_chat(self.prompt, max_tokens=config.CHAT_MAX_TOKENS)


def prepare_rag_turn(session_id: str, query: str, k: int | None = None) -> RagTurn:
    """Run everything up to (not including) the answer generation.

    `k` overrides the retrieval depth (the /chat default is retrieve()'s own
    default); the eval passes it explicitly, /chat does not.
    """
    has_history = bool(memory.get_history(session_id, limit=1))
    if not is_relevant_query(query, has_history):
        return RagTurn(declined=True, standalone_query=query)

    standalone_query = memory.condense_query(session_id, query)

    profile_text = ""
    try:
        with open(config.INFO_TXT_PATH, "r", encoding="utf-8") as f:
            profile_text = f.read()
    except OSError:
        profile_text = ""

    direct_fact_answer = answer_direct_profile_fact(
        standalone_query, profile_text)

    if direct_fact_answer is not None:
        history = memory.get_history(session_id)
        prompt = build_answer_prompt(
            [direct_fact_answer], standalone_query, history)
        return RagTurn(declined=False, standalone_query=standalone_query,
                       chunks=[direct_fact_answer], prompt=prompt)

    index, chunk_mapping = get_index()
    chunks = (
        retrieve(standalone_query, index, chunk_mapping)
        if k is None
        else retrieve(standalone_query, index, chunk_mapping, k=k)
    )

    history = memory.get_history(session_id)
    prompt = build_answer_prompt(chunks, standalone_query, history)
    return RagTurn(declined=False, standalone_query=standalone_query,
                   chunks=chunks, prompt=prompt)
