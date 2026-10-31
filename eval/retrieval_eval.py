"""1.1 Retrieval eval: does the real RAG retriever surface the right chunk?

    python -m eval.retrieval_eval --questions eval/questions.json --k 3

Questions with `expected_substring: null` (not filled in yet) and out_of_scope
questions are skipped and counted separately. Follow-up questions carry a
`history` list; by default their query is rewritten by the real
memory.condense_query (a live LLM call) before retrieval -- use
`--followups raw` to retrieve with the unrewritten text, or `skip` to leave
them out (the offline CI smoke set has none).
"""
import argparse
import sys
import uuid
from collections import defaultdict

from eval import _common as C


def run(questions, k, followups, info_path):
    from backend import memory
    from backend.rag import get_index, retrieve_with_ids

    per_question, skipped_unfilled, skipped_other = [], 0, 0

    with C.temp_backend_state(info_path):
        index, chunk_mapping = get_index()

        for q in questions:
            expected = C.expected_list(q)
            if q["type"] == "out_of_scope":
                skipped_other += 1
                continue
            if not expected:
                skipped_unfilled += 1
                continue
            if q["type"] == "followup" and followups == "skip":
                skipped_other += 1
                continue

            retrieval_query = q["query"]
            if q["type"] == "followup" and followups == "condense" and q.get("history"):
                sid = f"eval-{uuid.uuid4()}"
                C.seed_history(sid, q["history"])
                retrieval_query = memory.condense_query(sid, q["query"])

            hits = retrieve_with_ids(retrieval_query, index, chunk_mapping, k=k)
            per_question.append({
                "id": q["id"],
                "type": q["type"],
                "query": q["query"],
                "retrieval_query": retrieval_query,
                "expected": expected,
                "hit": C.substring_hit(expected, [text for _i, text in hits]),
                "retrieved_ids": [i for i, _t in hits],
            })

    return per_question, skipped_unfilled, skipped_other


def summarize(per_question):
    by_type_counts = defaultdict(lambda: [0, 0])      # type -> [hits, total]
    for r in per_question:
        by_type_counts[r["type"]][1] += 1
        by_type_counts[r["type"]][0] += int(r["hit"])
    total_hits = sum(int(r["hit"]) for r in per_question)
    overall = C.ratio(total_hits, len(per_question))
    by_type = {t: C.ratio(h, n) for t, (h, n) in sorted(by_type_counts.items())}
    return overall, by_type, by_type_counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--followups", choices=["condense", "raw", "skip"], default="condense")
    ap.add_argument("--info-path", help="index this file instead of Data/info.txt (temp FAISS store)")
    ap.add_argument("--min-hit-rate", type=float, help="exit 1 if overall hit-rate is below this (for CI)")
    args = ap.parse_args(argv)

    questions = C.load_json(args.questions)
    per_question, unfilled, other = run(questions, args.k, args.followups, args.info_path)

    if not per_question:
        print(f"No evaluable questions ({unfilled} have expected_substring=null; fill them in first).")
        return 2

    overall, by_type, counts = summarize(per_question)
    rows = [[t, counts[t][1], counts[t][0], C.fmt_pct(by_type[t])] for t in by_type]
    rows.append(["OVERALL", len(per_question), sum(c[0] for c in counts.values()), C.fmt_pct(overall)])
    print(f"\nRetrieval hit-rate@{args.k}")
    C.print_table(["type", "n", "hits", f"hit@{args.k}"], rows)
    print(f"\nskipped: {unfilled} unfilled (expected_substring=null), {other} out_of_scope/followup-skipped")
    misses = [r for r in per_question if not r["hit"]]
    if misses:
        print("\nMisses:")
        for r in misses:
            print(f"  {r['id']}: {r['retrieval_query']!r} -> expected one of {r['expected']}")

    path = C.write_results("retrieval", {
        "k": args.k, "followups_mode": args.followups, "info_path": args.info_path,
        "questions_file": args.questions,
        "n_evaluated": len(per_question), "n_skipped_unfilled": unfilled, "n_skipped_other": other,
        "overall_hit_rate": overall, "by_type": by_type, "per_question": per_question,
    })
    print(f"\nwrote {path}")

    if args.min_hit_rate is not None and overall < args.min_hit_rate:
        print(f"FAIL: hit-rate {overall:.3f} < required {args.min_hit_rate:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
