import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data")
INFO_TXT_PATH = os.path.join(DATA_DIR, "info.txt")
RESUME_TXT_PATH = os.path.join(DATA_DIR, "resume.txt")

FAISS_STORE_DIR = os.path.join(BASE_DIR, "faiss_store")
FAISS_INDEX_PATH = os.path.join(FAISS_STORE_DIR, "index.faiss")
CHUNK_MAPPING_PATH = os.path.join(FAISS_STORE_DIR, "chunk_mapping.pkl")
INFO_FINGERPRINT_PATH = os.path.join(FAISS_STORE_DIR, "info_source.sha256")

SQLITE_DB_PATH = os.path.join(BASE_DIR, "memory.db")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# fast + tool-calling capable: RAG chat, routing, project agent
CHAT_MODEL = "qwen/qwen3.8-27b"
# stronger reasoning for the one-shot JD comparison; not latency-sensitive
MATCH_MODEL = "openai/gpt-oss-120b"

# response length caps -- keeps answers crisp instead of turning into long
# essays / README dumps. MATCH_MAX_TOKENS is generous since that response is
# a structured JSON blob (topic verdicts + gaps + transferable skills), not
# free-form prose.
CHAT_MAX_TOKENS = 600
# README-based single-project overview -- bumped a bit so it can go beyond a
# bare one-liner (tech stack + a real detail or two), while still staying
# well short of a README dump.
PROJECT_MAX_TOKENS = 550
# separate, slightly larger budget for a deep-dive's final answer, since
# it's synthesizing real file content rather than just a README summary
DEEP_DIVE_ANSWER_MAX_TOKENS = 650
# "tell me about your projects" (general, no single project named) is
# answered from the same RAG profile (info.txt) as everything else in
# /chat -- NOT from GitHub -- since info.txt can include projects that
# aren't AI-related / aren't wired up in GITHUB_PROJECTS at all. Retrieval
# pulls more chunks than a normal chat turn so it has a real shot at
# surfacing every project paragraph, not just the nearest couple of matches.
ROUTER_MAX_TOKENS = 150
GATE_MAX_TOKENS = 10
CONDENSE_MAX_TOKENS = 250
TOOLCALL_MAX_TOKENS = 500
PROJECTS_OVERVIEW_RETRIEVE_K = 12
PROJECTS_OVERVIEW_MAX_TOKENS = 900
# Bumped 2000 -> 3200: the structured JD-match JSON (topic verdicts +
# transferable skills + gaps + final verdict) was intermittently getting cut
# off mid-object at 2000 tokens, which broke JSON parsing and silently fell
# back to the raw-text error path (see ats._parse_comparison_response).
MATCH_MAX_TOKENS = 3200

# how many past turns (user+assistant messages) to pull for history-aware chat
MAX_HISTORY_TURNS = 6

# chunking: max words per chunk (whole paragraphs under this stay intact),
# and how many trailing words carry over into the next chunk when a section
# has to be split, so a fact near the boundary isn't lost from every chunk
CHUNK_MAX_WORDS = 90
CHUNK_OVERLAP_WORDS = 20

# ---------------------------------------------------------------------------
# PROJECT REGISTRY / GITHUB ACCESS
# ---------------------------------------------------------------------------
# info.txt is the source of truth for project descriptions/facts. This config
# only declares which projects the GitHub agent knows how to inspect. To add a
# project: add its description to Data/info.txt and add its GitHub entry here.
# Do NOT copy the project description into this file; that would create two
# sources of truth.
#
# display_name is the canonical name shown to users. aliases are optional
# informal names users may use when referring to the same project.
GITHUB_PROJECTS = {
    "MediaShare": {
        "display_name": "MediaShare",
        "aliases": [
            "media share",
            "media sharing app",
            "media sharing application",
            "instagram-style media sharing app",
            "instagram style media sharing app",
        ],
        "owner": "Abhigya27",
        "repo": "MediaShare",
        "branch": "main",
    },
    "HireMeAI": {
        "display_name": "HireMeAI",
        "aliases": [
            "hire me ai",
            "personal ai chatbot",
            "portfolio chatbot",
            "job description matcher",
            "job description matching",
            "resume matcher",
        ],
        "owner": "Abhigya27",
        "repo": "HireMeAI",
        "branch": "main",
    },
    "ScrubGPT": {
        "display_name": "ScrubGPT",
        "aliases": [
            "scrub gpt",
            "youtube rag converter",
            "youtube rag",
            "youtube playlist rag",
            "youtube playlist converter",
            "youtube video rag",
        ],
        "owner": "Abhigya27",
        "repo": "ScrubGPT",
        "branch": "main",
    },
    "house-price-prediction": {
        "display_name": "House Price Prediction",
        "aliases": [
            "house price prediction",
            "house price model",
            "house price",
        ],
        "owner": "Abhigya27",
        "repo": "house-price-prediction",
        "branch": "main",
    },
}

PROJECT_KEYS = tuple(GITHUB_PROJECTS.keys())

# These are profile capabilities/features, not standalone portfolio projects.
# This list is deliberately tiny and only exists to make the intent explicit;
# project existence itself still comes from GITHUB_PROJECTS + info.txt.
PROJECT_FEATURE_NOT_PROJECTS = (
    "resume parser",
    "backend & frontend integration",
    "backend and frontend integration",
    "youtube playlist scraper",
    "score predictor",
)

# --- GitHub project-reading agent ---

# Optional: raises GitHub's API rate limit from 60/hour (unauthenticated) to
# 5000/hour. A read-only "public_repo" scope personal access token is enough.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# safety cap on tool-call rounds during a deep-dive, so the agent can't loop forever.
# Bumped 4 -> 6: with only 4 rounds the agent sometimes ran out of budget
# mid-investigation on questions that needed more than one or two files, and
# gave up (or picked the wrong project) too early.
DEEP_DIVE_MAX_ROUNDS = 6

# max characters kept from any single file the deep-dive agent reads, so one
# huge file can't blow the context budget
DEEP_DIVE_FILE_CHAR_CAP = 8000

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
