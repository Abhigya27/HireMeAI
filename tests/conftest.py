"""Shared pytest fixtures.

Guarantees for every test in this suite (all fixtures below are autouse):
  * no network: the Groq client is replaced by MockLLM, and the embedding model
    is replaced by a deterministic hashing embedder (no model download);
  * no shared state: config paths (info.txt, resume.txt, FAISS store, SQLite DB)
    point at a per-test tmp dir, and the in-memory RAG caches are reset.
"""
import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

# backend.client builds a Groq client at import time and Groq() raises without
# a key. A dummy value is enough because every call is mocked.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Deterministic offline embedder
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "the", "is", "are", "of", "to", "in", "on", "for", "with",
    "what", "which", "does", "do", "did", "how", "use", "uses", "it", "its",
    "that", "this", "as", "from", "by", "about", "me", "you", "your", "tell",
}


def hash_embed(text: str, dim: int = 384) -> np.ndarray:
    """Bag-of-words hashing embedding: texts sharing rare words land close
    together. Good enough to make 'known query -> expected chunk' testable."""
    vec = np.zeros(dim, dtype="float32")
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeEmbedder:
    """Stands in for fastembed.TextEmbedding (only .embed() is used)."""

    def embed(self, texts):
        for t in texts:
            yield hash_embed(t)


# ---------------------------------------------------------------------------
# Mock LLM (replaces the Groq client)
# ---------------------------------------------------------------------------
DEFAULT_ANALYSIS = {
    "topic_verdicts": [
        {"topic": "Skills", "verdict": "Strong overlap on Python and FastAPI."},
        {"topic": "Experience", "verdict": "Relevant project experience."},
    ],
    "transferable_skills": [
        {"required": "Django", "have_instead": "FastAPI",
         "why_similar": "Both are Python web frameworks."}
    ],
    "genuine_gaps": [{"gap": "No Kubernetes experience", "severity": "nice-to-have"}],
    "final_verdict": "Strong fit: core must-haves are covered.",
    "score": 82,
}


def make_tool_call_message(*paths: str):
    """A fake assistant message that asks to read_file each of `paths`."""
    calls = [
        SimpleNamespace(
            id=f"call_{i}",
            function=SimpleNamespace(
                name="read_file", arguments=json.dumps({"path": p})),
        )
        for i, p in enumerate(paths)
    ]
    dump = {
        "role": "assistant",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": "read_file", "arguments": c.function.arguments}}
            for c in calls
        ],
    }
    return SimpleNamespace(content=None, tool_calls=calls,
                           model_dump=lambda exclude_none=True: dict(dump))


def make_plain_message(text: str = "done"):
    return SimpleNamespace(
        content=text, tool_calls=None,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": text})


class MockLLM:
    """Replacement for the Groq client's chat.completions.create.

    * `when(substring, response)` -- reply `response` to any request whose
      messages contain `substring` (most recently added rule wins).
    * `tool_script` -- messages returned, in order, to requests that pass
      `tools=` (the GitHub agent's research rounds); when empty, a plain
      "no more tool calls" message is returned.
    * `error` -- if set, create() raises it.
    * `calls` -- every request seen: {"prompt", "messages", "model", "stream", "tools"}.
    """

    def __init__(self):
        self.calls = []
        self.rules = []
        self.tool_script = []
        self.error = None
        self._defaults = [
            ("scope gate", "YES"),
            ("Rewrite a follow-up question", "STANDALONE: What projects has Abhigya built?"),
            ("supportive, realistic technical reviewer", json.dumps(DEFAULT_ANALYSIS)),
        ]

    def when(self, substring: str, response: str):
        self.rules.insert(0, (substring, response))

    def prompts_containing(self, substring: str) -> list[str]:
        return [c["prompt"] for c in self.calls if substring in c["prompt"]]

    def _reply_text(self, prompt: str) -> str:
        for substring, response in [*self.rules, *self._defaults]:
            if substring in prompt:
                return response
        return "Canned answer."

    def create(self, *, model=None, messages=None, temperature=None,
               max_tokens=None, stream=False, tools=None, **_ignored):
        prompt = "\n".join((m.get("content") or "") for m in messages)
        self.calls.append({"prompt": prompt, "messages": messages, "model": model,
                           "stream": stream, "tools": tools})
        if self.error is not None:
            raise self.error

        if tools is not None:
            msg = self.tool_script.pop(0) if self.tool_script else make_plain_message()
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        text = self._reply_text(prompt)
        if stream:
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text[i:i + 12]))])
                for i in range(0, len(text), 12)
            ])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None))])


# ---------------------------------------------------------------------------
# Autouse isolation fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Per-test config paths, fake embedder, fresh DB, cleared RAG caches."""
    from backend import config, memory, rag

    info = tmp_path / "info.txt"
    resume = tmp_path / "resume.txt"
    shutil.copy(FIXTURES_DIR / "info.txt", info)
    shutil.copy(FIXTURES_DIR / "resume.txt", resume)
    store = tmp_path / "faiss_store"

    monkeypatch.setattr(config, "INFO_TXT_PATH", str(info))
    monkeypatch.setattr(config, "RESUME_TXT_PATH", str(resume))
    monkeypatch.setattr(config, "FAISS_STORE_DIR", str(store))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(store / "index.faiss"))
    monkeypatch.setattr(config, "CHUNK_MAPPING_PATH", str(store / "chunk_mapping.pkl"))
    monkeypatch.setattr(config, "INFO_FINGERPRINT_PATH", str(store / "info_source.sha256"))
    monkeypatch.setattr(config, "SQLITE_DB_PATH", str(tmp_path / "memory.db"))

    monkeypatch.setattr(rag, "_embedder", FakeEmbedder())
    monkeypatch.setattr(rag, "_index_cache", None)
    monkeypatch.setattr(rag, "_chunks_cache", None)
    monkeypatch.setattr(rag, "_index_source_fingerprint", None)

    memory.init_db()
    return SimpleNamespace(info_path=info, resume_path=resume, store_dir=store,
                           db_path=tmp_path / "memory.db")


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Swap the Groq client for MockLLM everywhere it is referenced."""
    from backend import client as client_module
    from backend import project_agent

    mock = MockLLM()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock.create)))
    monkeypatch.setattr(client_module, "client", fake_client)
    monkeypatch.setattr(project_agent, "client", fake_client)
    return mock


@pytest.fixture(autouse=True)
def block_github_network(monkeypatch):
    """github_client uses httpx.get; a test that reaches it unmocked should
    fail loudly instead of silently calling api.github.com."""
    import httpx

    def _boom(*args, **kwargs):
        raise RuntimeError("test tried to make a real HTTP request via httpx.get")

    monkeypatch.setattr(httpx, "get", _boom)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
def _reset_limiter(app):
    limiter = app.state.limiter
    try:
        limiter.reset()
    except AttributeError:  # very old slowapi
        limiter._storage.reset()


@pytest.fixture
def client(isolated_env, mock_llm):
    """FastAPI TestClient with the lifespan running (init_db + index build,
    both against the tmp fixtures) and a clean rate-limiter."""
    from fastapi.testclient import TestClient
    from backend.main import app

    _reset_limiter(app)
    with TestClient(app) as c:
        yield c
    _reset_limiter(app)


# ---------------------------------------------------------------------------
# Sample-file fixtures
# ---------------------------------------------------------------------------
JD_TEXT = ("Senior Python Backend Engineer. Required: FastAPI, PostgreSQL, "
           "REST API design. Nice to have: Kubernetes.")


def make_pdf_bytes(text: str) -> bytes:
    """A minimal, valid one-page PDF containing `text` (no PDF library needed)."""
    esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, xref_pos)
    return out


def make_docx_bytes(text: str) -> bytes:
    from docx import Document
    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def sample_jds():
    """{'txt'|'pdf'|'docx': (filename, bytes)} -- the same tiny JD in each format."""
    return {
        "txt": ("jd.txt", JD_TEXT.encode()),
        "pdf": ("jd.pdf", make_pdf_bytes(JD_TEXT)),
        "docx": ("jd.docx", make_docx_bytes(JD_TEXT)),
    }
