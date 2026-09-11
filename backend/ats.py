import io
import json
import re

import numpy as np
from pypdf import PdfReader
from docx import Document

import config
from rag import embed_text
from groq_client import complete


def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: .{ext or 'unknown'}. Use txt, pdf, or docx.")


def _load_resume_text() -> str:
    with open(config.RESUME_TXT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _embedding_similarity(resume_text: str, jd_text: str) -> float:
    resume_vec = embed_text(resume_text)
    jd_vec = embed_text(jd_text)
    cosine = float(
        np.dot(resume_vec, jd_vec)
        / (np.linalg.norm(resume_vec) * np.linalg.norm(jd_vec) + 1e-8)
    )
    # cosine is roughly in [-1, 1]; rescale to a 0-100 baseline score
    return float(np.clip((cosine + 1) / 2 * 100, 0, 100))


def _llm_verdict(resume_text: str, jd_text: str) -> dict:
    prompt = f"""You are assessing how well a candidate's resume matches a job description.

Resume:
{resume_text}

Job Description:
{jd_text}

Respond with ONLY a JSON object (no markdown fences, no extra text) with these exact keys:
{{
  "score": <integer 0-100>,
  "verdict": "<one short sentence verdict>",
  "matching_skills": ["...", "..."],
  "missing_skills": ["...", "..."],
  "reasoning": "<2-3 sentence explanation>"
}}"""

    raw = complete(prompt, temperature=0.2)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "score": None,
            "verdict": "Could not parse model output",
            "matching_skills": [],
            "missing_skills": [],
            "reasoning": raw,
        }


def score_match(jd_text: str) -> dict:
    resume_text = _load_resume_text()

    embedding_score = _embedding_similarity(resume_text, jd_text)
    llm_result = _llm_verdict(resume_text, jd_text)

    llm_score = llm_result.get("score")
    if isinstance(llm_score, (int, float)):
        final_score = round((embedding_score + llm_score) / 2, 1)
    else:
        final_score = round(embedding_score, 1)

    return {
        "final_score": final_score,
        "embedding_score": round(embedding_score, 1),
        "llm_score": llm_score,
        "verdict": llm_result.get("verdict"),
        "matching_skills": llm_result.get("matching_skills", []),
        "missing_skills": llm_result.get("missing_skills", []),
        "reasoning": llm_result.get("reasoning"),
    }