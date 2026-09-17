import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DATA_DIR = os.path.join(BASE_DIR, "data")
PROJECT_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data")
DATA_DIR = (
    BACKEND_DATA_DIR
    if os.path.isdir(BACKEND_DATA_DIR)
    else PROJECT_DATA_DIR
)
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
