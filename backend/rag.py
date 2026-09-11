import os
import pickle

import faiss
import numpy as np
from fastembed import TextEmbedding

import config

_embedder = None
_index_cache = None
_chunks_cache = None


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _embedder


def embed_text(text: str) -> np.ndarray:
    return np.array(list(_get_embedder().embed([text]))[0], dtype="float32")


def chunk_text(text: str, max_words: int = 50) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _build_index_from_source() -> tuple[faiss.Index, list[str]]:
    with open(config.INFO_TXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = np.array([embed_text(c) for c in chunks], dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(config.FAISS_STORE_DIR, exist_ok=True)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.CHUNK_MAPPING_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


def load_or_build_index() -> tuple[faiss.Index, list[str]]:
    valid = (
        os.path.exists(config.FAISS_INDEX_PATH)
        and os.path.getsize(config.FAISS_INDEX_PATH) > 0
        and os.path.exists(config.CHUNK_MAPPING_PATH)
        and os.path.getsize(config.CHUNK_MAPPING_PATH) > 0
    )

    if valid:
        try:
            index = faiss.read_index(config.FAISS_INDEX_PATH)
            with open(config.CHUNK_MAPPING_PATH, "rb") as f:
                chunk_mapping = pickle.load(f)
            return index, chunk_mapping
        except Exception as e:
            print(f"[rag] corrupted index, rebuilding: {e}")

    print("[rag] generating new FAISS index from info.txt")
    return _build_index_from_source()


def get_index() -> tuple[faiss.Index, list[str]]:
    """Cached accessor so the index is loaded/built once per process, not per request."""
    global _index_cache, _chunks_cache
    if _index_cache is None:
        _index_cache, _chunks_cache = load_or_build_index()
    return _index_cache, _chunks_cache


def retrieve(query: str, index: faiss.Index, chunk_mapping: list[str], k: int = 3) -> list[str]:
    query_vec = embed_text(query)
    _distances, indices = index.search(np.array([query_vec]), k)
    return [chunk_mapping[i] for i in indices[0] if i != -1]


def build_answer_prompt(context_chunks: list[str], query: str, history: list[dict] | None = None) -> str:
    context = "\n\n".join(context_chunks)

    history_block = ""
    if history:
        turns = "\n".join(f"{h['role']}: {h['content']}" for h in history)
        history_block = f"\nConversation so far:\n{turns}\n"

    return f"""You are answering questions about Abhigya Narain, using only the context provided below.
If the answer isn't in the context, say you don't know rather than making something up.
{history_block}
Context:
{context}

Question: {query}
Answer:"""