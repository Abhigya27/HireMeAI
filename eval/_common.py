"""Helpers shared by the eval scripts (results files, tables, temp backend state)."""
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"


# --------------------------------------------------------------------------
# results files
# --------------------------------------------------------------------------
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_results(prefix: str, payload: dict) -> Path:
    """Write eval/results/<prefix>_<UTC timestamp>.json, stamping generated_at and commit."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{prefix}_{stamp}.json"
    body = {"generated_at": now_iso(), "commit": git_commit(), **payload}
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def print_table(headers: list[str], rows: list[list]) -> None:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))  # noqa: E731
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print(line(r))


def ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def parse_json_object(text: str) -> dict | None:
    """Pull the first {...} object out of an LLM reply (tolerates fences/prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def expected_list(question: dict) -> list[str]:
    """`expected_substring` may be a string, a list of acceptable strings, or null."""
    exp = question.get("expected_substring")
    if exp is None:
        return []
    return [exp] if isinstance(exp, str) else [e for e in exp if e]


def substring_hit(expected: list[str], texts: list[str]) -> bool:
    """Case-insensitive: does any expected string occur in any of the texts?"""
    lowered = [t.lower() for t in texts]
    return any(e.lower() in t for e in expected for t in lowered)


# --------------------------------------------------------------------------
# running the real backend code against throwaway state
# --------------------------------------------------------------------------
@contextmanager
def temp_backend_state(info_path: str | None = None):
    """Point the backend at a throwaway SQLite DB so eval runs never touch the
    real memory.db. If `info_path` is given, also index that file instead of
    Data/info.txt, using a temporary FAISS store (e.g. the CI smoke fixture).
    """
    from backend import config, memory, rag

    tmp = Path(tempfile.mkdtemp(prefix="hiremeai_eval_"))
    keys = ["SQLITE_DB_PATH", "INFO_TXT_PATH", "FAISS_STORE_DIR", "FAISS_INDEX_PATH",
            "CHUNK_MAPPING_PATH", "INFO_FINGERPRINT_PATH"]
    saved = {k: getattr(config, k) for k in keys}
    saved_cache = (rag._index_cache, rag._chunks_cache, rag._index_source_fingerprint)
    try:
        config.SQLITE_DB_PATH = str(tmp / "memory.db")
        if info_path:
            store = tmp / "faiss_store"
            config.INFO_TXT_PATH = str(Path(info_path).resolve())
            config.FAISS_STORE_DIR = str(store)
            config.FAISS_INDEX_PATH = str(store / "index.faiss")
            config.CHUNK_MAPPING_PATH = str(store / "chunk_mapping.pkl")
            config.INFO_FINGERPRINT_PATH = str(store / "info_source.sha256")
            rag._index_cache = rag._chunks_cache = rag._index_source_fingerprint = None
        memory.init_db()
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        rag._index_cache, rag._chunks_cache, rag._index_source_fingerprint = saved_cache
        shutil.rmtree(tmp, ignore_errors=True)


def seed_history(session_id: str, history: list[dict] | None) -> None:
    """Pre-load prior turns (question.history) so follow-ups can be resolved."""
    from backend import memory
    for turn in history or []:
        memory.save_turn(session_id, turn["role"], turn["content"])


# --------------------------------------------------------------------------
# JD fit verdict bands -- mirrors score_band() in frontend/app.py
# --------------------------------------------------------------------------
VERDICTS = ("strong_fit", "moderate_fit", "weak_fit")


def verdict_from_score(score: float) -> str:
    if score >= 75:
        return "strong_fit"
    if score >= 50:
        return "moderate_fit"
    return "weak_fit"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
