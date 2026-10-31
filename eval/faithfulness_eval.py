"""1.2 Faithfulness eval: are /chat answers grounded in the retrieved context,
and are out-of-scope questions declined instead of answered?

    python -m eval.faithfulness_eval --questions eval/questions.json

How it runs: it imports backend.chat_pipeline.prepare_rag_turn -- the same
function /chat uses for the general profile path (scope gate -> condense ->
retrieve -> prompt) -- so it sees the exact retrieved chunks and generates the
answer with the same prompt and model settings. It deliberately does NOT go
through HTTP, because (a) the chunks aren't exposed over HTTP and (b) project
questions such as "what vector DB does ScrubGPT use?" are claimed by the
project router and answered from GitHub, which is a different pipeline (that
one is measured by agent_eval). The judge is a second LLM call at temperature 0
(default model: config.MATCH_MODEL, i.e. not the model that wrote the answer).

Needs GROQ_API_KEY. Questions with expected_substring=null are fine here
(it isn't used), so this runs on the skeleton as-is.
"""
import argparse
import sys
import time
import uuid
from collections import Counter

from eval import _common as C

GROUNDED_PROMPT = """You are grading whether an AI assistant's answer is grounded in the given context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Classify the answer as exactly one of: GROUNDED, PARTIALLY_GROUNDED, HALLUCINATED.
- GROUNDED: every claim in the answer is supported by the context.
- PARTIALLY_GROUNDED: some claims are supported, at least one is not.
- HALLUCINATED: the answer makes claims the context does not support.

Respond as JSON: {"verdict": "...", "reason": "one sentence"}"""

REFUSAL_PROMPT = """You are checking whether an AI assistant declined or attempted to answer a question.

The assistant is a personal portfolio chatbot that should ONLY discuss its owner's
background, skills, projects and experience. The question below is unrelated to that.

Question:
{question}

Answer:
{answer}

Classify the answer as exactly one of: DECLINED, ATTEMPTED_ANSWER.
- DECLINED: it says it can't help with this / redirects to its own topics, and does NOT provide the requested content.
- ATTEMPTED_ANSWER: it provides the requested content (even partially) or otherwise answers the unrelated question.

Respond as JSON: {"verdict": "...", "reason": "one sentence"}"""

GROUND_VERDICTS = {"GROUNDED", "PARTIALLY_GROUNDED", "HALLUCINATED"}
REFUSAL_VERDICTS = {"DECLINED", "ATTEMPTED_ANSWER"}


def judge(template: str, allowed: set[str], model: str, **fields) -> tuple[str, str]:
    from backend.client import complete
    prompt = template
    for key, value in fields.items():             # str.replace: templates contain literal JSON braces
        prompt = prompt.replace("{" + key + "}", value)
    raw = complete(prompt, model=model, temperature=0.0, max_tokens=2000)
    obj = C.parse_json_object(raw) or {}
    verdict = str(obj.get("verdict", "")).strip().upper()
    return (verdict if verdict in allowed else "JUDGE_ERROR"), str(obj.get("reason", raw[:200]))


def evaluate_question(q, judge_model, tag):
    from backend.chat_pipeline import prepare_rag_turn

    session_id = f"eval-{tag}-{uuid.uuid4()}"
    C.seed_history(session_id, q.get("history"))
    turn = prepare_rag_turn(session_id, q["query"])
    answer = "".join(turn.stream())
    is_oos = q["type"] == "out_of_scope"

    record = {"id": q["id"], "type": q["type"], "query": q["query"], "answer": answer,
              "declined_by_gate": turn.declined, "retrieved_chunks": turn.chunks}

    if is_oos:
        if turn.declined:                                   # canned decline: no judge needed
            record.update(verdict="DECLINED", reason="blocked by the scope gate")
        else:
            record["verdict"], record["reason"] = judge(
                REFUSAL_PROMPT, REFUSAL_VERDICTS, judge_model, question=q["query"], answer=answer)
    elif turn.declined:
        record.update(verdict="WRONGLY_DECLINED", reason="scope gate rejected an in-scope question")
    else:
        record["verdict"], record["reason"] = judge(
            GROUNDED_PROMPT, GROUND_VERDICTS, judge_model,
            context="\n\n".join(turn.chunks), question=turn.standalone_query, answer=answer)
    return record


def main(argv=None):
    from backend import config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--judge-model", default=config.MATCH_MODEL)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (Groq rate limits)")
    ap.add_argument("--limit", type=int, help="only run the first N questions (quick trial)")
    args = ap.parse_args(argv)

    questions = C.load_json(args.questions)
    if args.limit:
        questions = questions[:args.limit]

    records, errors = [], []
    with C.temp_backend_state():
        for n, q in enumerate(questions, start=1):
            try:
                rec = evaluate_question(q, args.judge_model, tag=q["id"])
                records.append(rec)
                print(f"[{n}/{len(questions)}] {q['id']:<5} {q['type']:<13} -> {rec['verdict']}")
            except Exception as e:                           # keep going; report at the end
                errors.append({"id": q["id"], "error": repr(e)})
                print(f"[{n}/{len(questions)}] {q['id']:<5} ERROR {e!r}")
            if args.sleep:
                time.sleep(args.sleep)

    in_scope = [r for r in records if r["type"] != "out_of_scope"]
    oos = [r for r in records if r["type"] == "out_of_scope"]
    g = Counter(r["verdict"] for r in in_scope)
    judged = g["GROUNDED"] + g["PARTIALLY_GROUNDED"] + g["HALLUCINATED"]
    pct_grounded = C.ratio(g["GROUNDED"], judged)
    refusals = sum(r["verdict"] == "DECLINED" for r in oos)
    pct_refusal = C.ratio(refusals, len(oos))

    print("\nFaithfulness (in-scope answers)")
    C.print_table(["verdict", "count"], [[v, g[v]] for v in
                  ["GROUNDED", "PARTIALLY_GROUNDED", "HALLUCINATED", "WRONGLY_DECLINED", "JUDGE_ERROR"]])
    print(f"\n% grounded         = {C.fmt_pct(pct_grounded)}  ({g['GROUNDED']}/{judged})")
    print(f"% correct refusals = {C.fmt_pct(pct_refusal)}  ({refusals}/{len(oos)})")
    if errors:
        print(f"{len(errors)} question(s) errored and are excluded from the metrics.")

    path = C.write_results("faithfulness", {
        "questions_file": args.questions, "judge_model": args.judge_model,
        "answer_model": config.CHAT_MODEL,
        "scope_note": "profile-RAG path only (chat_pipeline); GitHub-routed answers are measured by agent_eval",
        "n_in_scope": len(in_scope), "n_out_of_scope": len(oos),
        "pct_grounded": pct_grounded, "pct_correct_refusals": pct_refusal,
        "in_scope_verdict_counts": dict(g), "wrongly_declined": g["WRONGLY_DECLINED"],
        "errors": errors, "per_question": records,
    })
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
