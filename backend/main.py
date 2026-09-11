from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import memory
from rag import get_index, retrieve, build_answer_prompt
from client import stream_chat
from ats import extract_text, score_match


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


class ChatRequest(BaseModel):
    session_id: str
    query: str


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

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
        for token in stream_chat(prompt):
            full_reply += token
            yield token
        memory.save_turn(req.session_id, "assistant", full_reply)

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/chat/clear")
def clear_chat(session_id: str):
    memory.clear_history(session_id)
    return {"status": "cleared"}


@app.post("/match")
async def match(file: UploadFile = File(...)):
    file_bytes = await file.read()

    try:
        jd_text = extract_text(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

    return score_match(jd_text)


@app.get("/health")
def health():
    return {"status": "ok"}