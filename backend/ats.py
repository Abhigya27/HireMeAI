import io
import json
import re

import numpy as np
from pypdf import PdfReader
from docx import Document

import config
from rag import embed_text
from client import complete


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


def _llm_field_comparison(resume_text: str, jd_text: str) -> dict:
    prompt = f"""You are comparing a candidate's resume against a job description, field by field,
to help the candidate honestly understand their fit for this specific role.

Resume:
{resume_text}

Job Description:
{jd_text}

Do the following:

1. Identify the topics/fields worth comparing. Always include "Skills", "Projects", and
   "Experience" if the resume has relevant content for them. Add any other distinct
   requirement categories the job description raises (e.g. Education, Certifications,
   Domain Knowledge, Tools). Don't invent a topic with nothing real to compare.

2. For each topic, write ONE short, specific sentence verdict comparing what the job
   description asks for against what the resume actually shows. Be concrete, not vague
   filler like "seems like a decent match."

3. Separately, identify skills/tools the job description explicitly asks for that the
   resume doesn't have under that exact name, but where the resume shows a genuinely
   comparable, transferable skill (for example: the job wants Django, the resume shows
   FastAPI -- both are Python web frameworks). For each, name the required skill, the
   resume's closest equivalent, and a short, honest reason they're comparable. Only
   include equivalences that are genuinely reasonable -- do not stretch to force a match.

4. List genuine gaps: things the job description clearly wants where the resume has
   neither the exact skill nor a reasonable equivalent. Be honest here -- do not hide or
   soften a real gap.

5. Write a final overall verdict, 2-4 sentences: state plainly whether this looks like a
   good fit, a stretch, or not a good fit, and where the candidate is lacking if so. Keep
   the tone encouraging and constructive, but never invent strengths or downplay a real
   gap just to sound nicer -- an honest "this is a stretch because X" is more useful than
   false positivity.

6. Give an overall fit score from 0-100 reflecting all of the above.

Respond with ONLY a JSON object (no markdown fences, no extra text) with exactly this shape:
{{
  "topic_verdicts": [{{"topic": "...", "verdict": "..."}}],
  "transferable_skills": [{{"required": "...", "have_instead": "...", "why_similar": "..."}}],
  "genuine_gaps": ["...", "..."],
  "final_verdict": "...",
  "score": <integer 0-100>
}}"""

    raw = complete(prompt, temperature=0.2)

    # be defensive: strip any markdown fences / stray text around the JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "topic_verdicts": [],
            "transferable_skills": [],
            "genuine_gaps": [],
            "final_verdict": "Could not parse the model's output. Raw response: " + raw[:500],
            "score": None,
        }


def score_match(jd_text: str) -> dict:
    resume_text = _load_resume_text()

    embedding_score = _embedding_similarity(resume_text, jd_text)
    analysis = _llm_field_comparison(resume_text, jd_text)

    llm_score = analysis.get("score")
    if isinstance(llm_score, (int, float)):
        final_score = round((embedding_score + llm_score) / 2, 1)
    else:
        final_score = round(embedding_score, 1)

    return {
        "final_score": final_score,
        "embedding_score": round(embedding_score, 1),
        "llm_score": llm_score,
        "topic_verdicts": analysis.get("topic_verdicts", []),
        "transferable_skills": analysis.get("transferable_skills", []),
        "genuine_gaps": analysis.get("genuine_gaps", []),
        "final_verdict": analysis.get("final_verdict"),
    }