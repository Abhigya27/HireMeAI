import os
from dotenv import load_dotenv
from groq import Groq

import config

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def stream_chat(prompt: str, model: str = config.CHAT_MODEL, temperature: float = 0.2):
    """Yield content chunks from a streaming Groq chat completion.
    Used for the conversational RAG tab, where the answer streams to the user.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def complete(prompt: str, model: str = config.CHAT_MODEL, temperature: float = 0.2) -> str:
    """Return a full (non-streaming) completion.
    Used internally for query condensation and JD-match verdicts, where we need
    the whole response at once rather than a stream.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content