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

MAX_HISTORY_TURNS = 5