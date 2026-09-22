import os
from dotenv import load_dotenv
from groq import Groq

from backend import config

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def stream_messages(
    messages: list[dict],
    model: str = config.CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
):
    """Yield content chunks from a streaming Groq chat completion, given a
    full messages list (system/user/assistant/tool turns).

    This is the lower-level primitive -- stream_chat() below is a thin
    single-prompt wrapper around it. project_agent uses this directly to
    stream the deep-dive agent's final answer for real, after its
    tool-calling rounds (which can't be streamed) have finished gathering
    context.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def stream_chat(
    prompt: str,
    model: str = config.CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
):
    """Yield content chunks from a streaming Groq chat completion for a
    single-turn prompt. Used for the conversational RAG tab, the project
    README overview, and the JD-match analysis, where the answer streams to
    the user instead of arriving all at once.
    """
    yield from stream_messages(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def complete(
    prompt: str,
    model: str = config.CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """Return a full (non-streaming) completion.
    Used internally for routing and query condensation, where we need the
    whole (short) response at once rather than a stream.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content