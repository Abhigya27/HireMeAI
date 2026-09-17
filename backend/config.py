import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
INFO_TXT_PATH = os.path.join(DATA_DIR, "info.txt")
RESUME_TXT_PATH = os.path.join(DATA_DIR, "resume.txt")

FAISS_STORE_DIR = os.path.join(BASE_DIR, "faiss_store")
FAISS_INDEX_PATH = os.path.join(FAISS_STORE_DIR, "index.faiss")
CHUNK_MAPPING_PATH = os.path.join(FAISS_STORE_DIR, "chunk_mapping.pkl")

SQLITE_DB_PATH = os.path.join(BASE_DIR, "memory.db")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

CHAT_MODEL = "openai/gpt-oss-120b"

# how many past turns (user+assistant messages) to pull for history-aware chat
MAX_HISTORY_TURNS = 6

# chunking: max words per chunk (whole paragraphs under this stay intact),
# and how many trailing words carry over into the next chunk when a section
# has to be split, so a fact near the boundary isn't lost from every chunk
CHUNK_MAX_WORDS = 80
CHUNK_OVERLAP_WORDS = 15

# --- GitHub project-reading agent ---

# Optional: raises GitHub's API rate limit from 60/hour (unauthenticated) to
# 5000/hour. A read-only "public_repo" scope personal access token is enough.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Explicit mapping of project name (as it might come up in chat) -> GitHub repo.
# Fill this in with your real repos. Example:
# GITHUB_PROJECTS = {
#     "personal RAG chatbot": {"owner": "your-username", "repo": "ask-abhigya", "branch": "main"},
#     "house price prediction": {"owner": "your-username", "repo": "house-price-ml", "branch": "main"},
# }
GITHUB_PROJECTS: dict = {}

# safety cap on tool-call rounds during a deep-dive, so the agent can't loop forever
DEEP_DIVE_MAX_ROUNDS = 4

# --- safety / abuse protection ---

# per-client-IP rate limits. Two windows per route: a short burst cap and a
# sustained hourly cap, so a quick flurry of genuine use isn't blocked but
# sustained hammering is. /match is limited tighter since one request costs
# two LLM calls plus embeddings, not one.
CHAT_RATE_LIMIT_SHORT = "15/minute"
CHAT_RATE_LIMIT_LONG = "100/hour"
MATCH_RATE_LIMIT_SHORT = "5/minute"
MATCH_RATE_LIMIT_LONG = "20/hour"

MAX_QUERY_LENGTH = 2000  # characters, per chat message
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB, per JD file upload