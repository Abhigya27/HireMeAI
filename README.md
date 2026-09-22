# Ask Abhigya 🤖

A personal AI chatbot + job-fit matcher, built as an interactive resume/portfolio site.

Two tools in one app:

1. **💬 Conversational RAG Chatbot** — ask questions about Abhigya's background, skills, and projects. History-aware: follow-ups like *"tell me about the second last one"* are correctly resolved against the prior conversation before retrieval runs.
2. **📄 Job Description Matcher** — upload a job description (`.txt` / `.pdf` / `.docx`) and get a fit score, verdict, and skill-gap breakdown against Abhigya's resume.

Built with **plain Python** — no LangChain, no heavy ML frameworks. Chunking, retrieval, and conversation memory are all hand-rolled in a few hundred lines.

---

## Architecture

```
   STREAMLIT                          FASTAPI
 frontend/app.py    ──── HTTP ────▶  backend/main.py
                                          │
                       ┌──────────────────┼──────────────────┐
                       ▼                  ▼                  ▼
                    rag.py            memory.py         job_matcher.py
                 (FAISS +            (SQLite +          (pypdf / docx +
                fastembed)         query condensing)    hybrid scoring)
                       │                  │                  │
                       └──────────────────┼──────────────────┘
                                          ▼
                                       Groq API
                                (openai/gpt-oss-120b)
```

- Frontend and backend are **two separate processes** talking over plain HTTP — no shared imports or state.
- **No vector DB service** — FAISS runs in-process, index persisted to disk as a flat file.
- **No memory framework** — a plain SQLite table plus one extra LLM call ("query condensation") gives history-awareness without something like LangChain's `ConversationBufferMemory`.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit, httpx |
| Backend | FastAPI, Uvicorn |
| Embeddings | fastembed (`BAAI/bge-small-en-v1.5` — ONNX runtime, no PyTorch/`sentence-transformers`) |
| Vector search | FAISS (`IndexFlatL2`) |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| History storage | SQLite |
| File parsing | pypdf, python-docx |

## Project structure

```
ask-abhigya/
├── backend/
│   ├── main.py           # FastAPI app: /chat, /chat/clear, /match, /health
│   ├── rag.py             # embedding, chunking, FAISS index, retrieval, answer prompt
│   ├── memory.py          # SQLite session history + history-aware query condensation
│   ├── job_matcher.py     # file parsing (txt/pdf/docx) + resume-vs-JD scoring
│   ├── groq_client.py     # shared Groq client (streaming + non-streaming)
│   ├── config.py          # paths & constants
│   ├── requirements.txt
│   ├── .env.example
│   └── data/
│       ├── info.txt       # RAG source document (chunked + embedded)
│       └── resume.txt     # full resume/skills summary (used whole, for JD matching)
├── frontend/
│   ├── app.py             # sidebar + Chat tab + JD Matcher tab
│   ├── requirements.txt
│   └── .env.example
├── DEPLOY.md              # Render + Streamlit Community Cloud deployment guide
└── .gitignore
```

## How the RAG pipeline works

1. **Chunking** — `info.txt` is split into fixed-size word chunks (`chunk_text()`).
2. **Embedding** — each chunk is embedded with `fastembed` (384-dim vectors) and stored in a FAISS `IndexFlatL2`, persisted under `faiss_store/`.
3. **On every chat message:**
   - `condense_query()` rewrites follow-ups ("what about him?", "the second last one") into standalone questions using recent turns from SQLite — this is what makes retrieval history-aware. It explicitly asks the model to count list items before resolving ordinal references, since that's the failure mode most likely to trip up a plain LLM call.
   - The standalone query is embedded and used to search FAISS for the top-k relevant chunks.
   - Retrieved chunks + recent conversation history are assembled into one prompt (`build_answer_prompt()`) with strict grounding rules: no speculation, no hedged guesses, no attributing general skills to a specific project unless the context says so.
   - The prompt streams to Groq; tokens stream straight through to the frontend.
4. Both the user's question and the assistant's full reply are saved back to SQLite for the next turn.

## Job Description Matcher

1. The uploaded file is parsed to plain text (`extract_text()` — routes by extension to `pypdf` or `python-docx`).
2. **Hybrid scoring:**
   - *Embedding similarity* — cosine similarity between resume and JD embeddings, rescaled to 0–100.
   - *LLM verdict* — a structured JSON response (score, verdict, matching skills, missing skills, reasoning) from Groq.
   - The final score averages the two, so it isn't relying on a single, un-sanity-checked signal.

---

## Getting started (local)

```bash
git clone <your-repo-url>
cd ask-abhigya
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

cp backend/.env.example backend/.env
# edit backend/.env and add your GROQ_API_KEY
```

**Terminal 1 — backend:**
```bash
uvicorn backend.main:app --reload
```

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run app.py
```

Frontend opens at `http://localhost:8501`, talking to the backend at `http://127.0.0.1:8000`.

> Before running, drop your real content into `backend/data/info.txt` and `backend/data/resume.txt` — both ship as placeholders.

## Environment variables

| Variable | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | `backend/.env` | Your [Groq API key](https://console.groq.com) |
| `BACKEND_URL` | `frontend/.env` (local) or Streamlit Cloud **Secrets** (deployed) | URL of the FastAPI backend |

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | `{session_id, query}` → streamed plain-text answer |
| `/chat/clear` | POST | `?session_id=...` → clears that session's history |
| `/match` | POST | multipart file upload → JSON fit score + verdict |
| `/health` | GET | `{"status": "ok"}` liveness check |

Interactive docs available at `/docs` once the backend is running.

## Deployment

See [`DEPLOY.md`](./DEPLOY.md) for the full walkthrough — backend on Render, frontend on Streamlit Community Cloud, both on free tiers.

## Known limitations

- Fixed-size word chunking (no paragraph/semantic-aware splitting) can occasionally split one project's description across two chunks.
- SQLite chat history is per-process — on Render's free tier it resets whenever the service spins down from inactivity (by design, not a bug — matches the "reset on restart" behavior chosen for this project).
- PDF extraction (`pypdf`) doesn't OCR — scanned/image-only PDFs won't yield extractable text.

## Connect

- ✉️ [Email](mailto:your_email@example.com)
- 💻 [GitHub](https://github.com/your-username)
- 🔗 [LinkedIn](https://linkedin.com/in/your-profile)
- 📸 [Instagram](https://instagram.com/your-handle)

## License

The code in this repo is free to fork and adapt for your own portfolio. The contents of `backend/data/` are personal information — please don't reuse that verbatim.
