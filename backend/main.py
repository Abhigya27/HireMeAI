from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend import config, memory, project_agent
from backend.rag import get_index, retrieve, build_answer_prompt, build_projects_overview_prompt, is_relevant_query
from backend.client import stream_chat
from backend.ats import extract_text, score_match_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init_db()
    get_index()  # warm the FAISS index + embedder once on boot, not per-request
    yield


app = FastAPI(title="Ask Abhigya API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your Streamlit URL before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# per-client-IP rate limiting, in-memory (no Redis needed for a single-instance
# deployment). Limits themselves live in config.py.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class ChatRequest(BaseModel):
    session_id: str
    query: str = Field(..., max_length=config.MAX_QUERY_LENGTH)


def _links_block(links: list[str]) -> str:
    if not links:
        return ""
    unique_links = list(dict.fromkeys(links))  # de-dupe, keep order
    links_text = "\n".join(f"- {link}" for link in unique_links)
    return f"\n\n**Source:**\n{links_text}"


def _stream_project_reply(session_id: str, user_query: str, gen, links: list[str], set_active: str | None):
    """Shared plumbing for the overview/deep_dive routes: stream the
    generator's chunks straight through to the client as they're generated,
    then append the source links, persist the turn, and (for overview) mark
    the project as active -- once the stream is actually exhausted.
    """
    def event_stream():
        full_text = ""
        for token in gen:
            full_text += token
            yield token

        block = _links_block(links)
        if block:
            yield block

        memory.save_turn(session_id, "user", user_query)
        memory.save_turn(session_id, "assistant", full_text + block)
        if set_active is not None:
            memory.set_active_project(session_id, set_active)

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/chat")
@limiter.limit(config.CHAT_RATE_LIMIT_SHORT)
@limiter.limit(config.CHAT_RATE_LIMIT_LONG)
def chat(request: Request, req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 0a. does this look like a request for GitHub-level project detail?
    # checked first so a bare "yes" confirming an active deep-dive is handled
    # here, before the general topic gate below ever sees it in isolation.
    active_project = memory.get_active_project(req.session_id)
    route = project_agent.route_project_query(req.query, active_project)

    if route["action"] == "overview":
        gen, links = project_agent.answer_overview(
            route["project_key"], req.query)
        return _stream_project_reply(
            req.session_id, req.query, gen, links, set_active=route["project_key"])

    if route["action"] == "deep_dive":
        gen, links = project_agent.answer_deep_dive(
            route["project_key"], req.query)
        return _stream_project_reply(
            req.session_id, req.query, gen, links, set_active=None)

    if route["action"] == "list_projects":
        # Project identity and repository links come from the deterministic
        # registry, so the LLM cannot omit or invent a project URL.
        gen, links = project_agent.answer_all_projects()

        memory.save_turn(req.session_id, "user", req.query)

        def event_stream():
            full_reply = ""
            for token in gen:
                full_reply += token
                yield token
            block = _links_block(links)
            if block:
                full_reply += block
                yield block
            memory.save_turn(req.session_id, "assistant", full_reply)
            # lightweight marker so a short follow-up ("tell me more") is
            # recognized as continuing this same general rundown rather than
            # a fresh, unrelated mention -- see project_agent.ALL_PROJECTS_KEY
            memory.set_active_project(
                req.session_id, project_agent.ALL_PROJECTS_KEY)

        return StreamingResponse(event_stream(), media_type="text/plain")

    # 0b. general topic gate -- only reached for messages the project router
    # didn't already claim. Keeps the chatbot from answering as a
    # general-purpose assistant.
    has_history = bool(memory.get_history(req.session_id, limit=1))
    if not is_relevant_query(req.query, has_history):
        decline = (
            "I'm just here to talk about myself — my background, skills, projects, and "
            "experience. Happy to help with any of that!"
        )
        memory.save_turn(req.session_id, "user", req.query)
        memory.save_turn(req.session_id, "assistant", decline)

        def decline_stream():
            for i in range(0, len(decline), 40):
                yield decline[i:i + 40]

        return StreamingResponse(decline_stream(), media_type="text/plain")

    # --- otherwise: existing RAG path, unchanged ---

    # 1. resolve follow-up references ("what about him?") into a standalone query
    standalone_query = memory.condense_query(req.session_id, req.query)

    # 2. retrieve context using the standalone query
    index, chunk_mapping = get_index()
    chunks = retrieve(standalone_query, index, chunk_mapping)

    # 3. build the final prompt with retrieved context + recent history
    history = memory.get_history(req.session_id)
    prompt = build_answer_prompt(chunks, standalone_query, history)

    # 4. persist the user's turn now, assistant's turn once streaming finishes
    memory.save_turn(req.session_id, "user", req.query)

    def event_stream():
        full_reply = ""
        for token in stream_chat(prompt, max_tokens=config.CHAT_MAX_TOKENS):
            full_reply += token
            yield token
        memory.save_turn(req.session_id, "assistant", full_reply)

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/chat/clear")
def clear_chat(session_id: str):
    memory.clear_history(session_id)
    return {"status": "cleared"}


@app.post("/match")
@limiter.limit(config.MATCH_RATE_LIMIT_SHORT)
@limiter.limit(config.MATCH_RATE_LIMIT_LONG)
async def match(request: Request, file: UploadFile = File(...)):
    file_bytes = await file.read()

    if len(file_bytes) > config.MAX_UPLOAD_BYTES:
        max_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"File too large (max {max_mb}MB).")

    try:
        jd_text = extract_text(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not jd_text.strip():
        raise HTTPException(
            status_code=400, detail="Could not extract any text from the file.")

    # Streams the raw analysis text as it's generated, then a final chunk
    # (prefixed with ats.RESULT_MARKER) carrying the parsed structured
    # result -- see ats.score_match_stream for the wire format. The frontend
    # buffers chunks and splits on that marker once it appears.
    return StreamingResponse(score_match_stream(jd_text), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok"}
