"""1.5 GitHub deep-dive agent: tool calls, files read, and whether it read the right file.

    python -m eval.agent_eval --questions eval/agent_questions.json

Calls project_agent.answer_deep_dive directly with an AgentTrace, so it reports
exactly what the agent did (num_tool_calls, files_read, research_rounds_used,
wall_clock_seconds). It also checks whether the deterministic router would have
sent the question to the deep-dive in the first place.

Hits the live GitHub API and Groq. Unauthenticated GitHub allows 60 requests/hour
and each question costs 1 listing + ~1-6 file reads, so set GITHUB_TOKEN (a
read-only public_repo token is enough) before running the full set.
"""
import argparse
import statistics
import sys
import time

from eval import _common as C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/agent_questions.json")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions")
    args = ap.parse_args(argv)

    from backend import config
    from backend.project_agent import AgentTrace, answer_deep_dive, route_project_query

    if not config.GITHUB_TOKEN:
        print("WARNING: GITHUB_TOKEN is not set -- GitHub's 60 requests/hour limit will likely stop a full run.\n")

    questions = C.load_json(args.questions)
    results = []
    with C.temp_backend_state():
        for n, q in enumerate(questions, start=1):
            project_key = q.get("project_key", "HireMeAI")
            expected = C.expected_list({"expected_substring": q.get("expected_file_substring")})
            route = route_project_query(q["question"], None)
            routed_ok = route["action"] == "deep_dive" and route["project_key"] == project_key

            trace = AgentTrace()
            try:
                gen, links = answer_deep_dive(project_key, q["question"], trace=trace)
                answer = "".join(gen)
            except Exception as e:
                print(f"[{n}/{len(questions)}] {q['id']} ERROR {e!r}")
                results.append({"id": q["id"], "error": repr(e)})
                continue

            file_hit = C.substring_hit(expected, trace.files_read) if expected else None
            results.append({
                "id": q["id"], "question": q["question"], "project_key": project_key,
                "expected_file_substring": expected, "expected_file_read": file_hit,
                "routed_to_deep_dive": routed_ok, "trace": trace.to_dict(),
                "answer": answer, "links": links,
            })
            print(f"[{n}/{len(questions)}] {q['id']:<5} tool_calls={trace.num_tool_calls} "
                  f"files={len(set(trace.files_read))} expected_hit={file_hit} "
                  f"{trace.wall_clock_seconds}s routed={routed_ok}")
            if args.sleep:
                time.sleep(args.sleep)

    ok = [r for r in results if "trace" in r and r["trace"]["outcome"] == "answered"]
    if not ok:
        print("No successful agent runs.")
        return 2
    judged = [r for r in ok if r["expected_file_read"] is not None]
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else None  # noqa: E731
    summary = {
        "questions": len(questions), "answered": len(ok),
        "errors_or_failed_outcomes": len(questions) - len(ok),
        "avg_tool_calls": mean([r["trace"]["num_tool_calls"] for r in ok]),
        "avg_unique_files_read": mean([len(set(r["trace"]["files_read"])) for r in ok]),
        "avg_research_rounds": mean([r["trace"]["research_rounds_used"] for r in ok]),
        "avg_wall_clock_seconds": mean([r["trace"]["wall_clock_seconds"] for r in ok]),
        "pct_expected_file_read": C.ratio(sum(r["expected_file_read"] for r in judged), len(judged)),
        "pct_hit_round_limit": C.ratio(sum(r["trace"]["exhausted_rounds"] for r in ok), len(ok)),
        "pct_routed_to_deep_dive": C.ratio(sum(r["routed_to_deep_dive"] for r in ok), len(ok)),
        "github_token_used": bool(config.GITHUB_TOKEN),
    }
    print("\nGitHub agent")
    C.print_table(["metric", "value"], [
        ["avg tool calls / answer", summary["avg_tool_calls"]],
        ["avg unique files read", summary["avg_unique_files_read"]],
        ["avg research rounds", summary["avg_research_rounds"]],
        ["avg wall clock (s)", summary["avg_wall_clock_seconds"]],
        ["% expected file read", C.fmt_pct(summary["pct_expected_file_read"])],
        ["% hit round limit", C.fmt_pct(summary["pct_hit_round_limit"])],
        ["% routed to deep-dive", C.fmt_pct(summary["pct_routed_to_deep_dive"])],
    ])
    print(f"\nwrote {C.write_results('agent', {'questions_file': args.questions, **summary, 'per_question': results})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
