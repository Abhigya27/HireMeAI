"""1.3 Latency benchmark: time-to-first-token and total time for POST /chat.

    python -m eval.latency_bench --base-url http://127.0.0.1:8000 --n 50

Rate limits: production allows 15 requests/minute and 100/hour per IP, so the
default --sleep of 4.5 s keeps a sequential run under the per-minute cap
(~13/min). To benchmark faster, or to use --concurrency, start a LOCAL backend
with the limits raised first, e.g.

    CHAT_RATE_LIMIT_SHORT=1000/minute CHAT_RATE_LIMIT_LONG=10000/hour \\
        uvicorn backend.main:app

and pass --limit-note so the output records what was in force.

Requests are made the way frontend/app.py makes them (a fresh httpx.stream per
request, new session id each time), so TTFT includes connection setup.
Default question types are profile_fact/skill_fact: those take the general
RAG path. project_fact questions are usually routed to GitHub (different
latency profile) -- add them via --types if that's what you want to measure.
"""
import argparse
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from eval import _common as C


def one_request(base_url: str, query: str, timeout: float) -> dict:
    t0 = time.perf_counter()
    ttft, status, error = None, None, None
    try:
        with httpx.stream("POST", f"{base_url}/chat", timeout=timeout,
                          json={"session_id": f"bench-{uuid.uuid4()}", "query": query}) as resp:
            status = resp.status_code
            if status != 200:
                resp.read()
            else:
                for chunk in resp.iter_text():
                    if chunk and ttft is None:
                        ttft = time.perf_counter() - t0
    except httpx.HTTPError as e:
        error = repr(e)
    total = time.perf_counter() - t0
    ok = status == 200 and ttft is not None
    return {"query": query, "status": status, "ok": ok, "error": error,
            "ttft": round(ttft, 4) if ttft is not None else None, "total": round(total, 4)}


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"p50": round(float(p50), 4), "p95": round(float(p95), 4), "p99": round(float(p99), 4)}


def summarize(samples: list[dict]) -> dict:
    good = [s for s in samples if s["ok"]]
    return {
        "requests": len(samples), "ok": len(good),
        "rate_limited_429": sum(s["status"] == 429 for s in samples),
        "other_failures": sum((not s["ok"]) and s["status"] != 429 for s in samples),
        "ttft": percentiles([s["ttft"] for s in good]),
        "total": percentiles([s["total"] for s in good]),
    }


def print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}: {summary['ok']}/{summary['requests']} ok "
          f"({summary['rate_limited_429']} rate-limited, {summary['other_failures']} failed)")
    if not summary["ttft"]:
        print("  no successful requests")
        return
    C.print_table(["", "p50", "p95", "p99"], [
        ["TTFT", *(f"{summary['ttft'][p]:.2f}s" for p in ("p50", "p95", "p99"))],
        ["Total", *(f"{summary['total'][p]:.2f}s" for p in ("p50", "p95", "p99"))],
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--types", default="profile_fact,skill_fact", help="comma-separated question types to draw from")
    ap.add_argument("--n", type=int, default=50, help="requests per pass")
    ap.add_argument("--sleep", type=float, default=4.5, help="seconds between sequential requests")
    ap.add_argument("--warmup", type=int, default=1, help="initial requests to discard (cold caches)")
    ap.add_argument("--concurrency", type=int, default=1, help=">1 adds a second, concurrent pass")
    ap.add_argument("--limit-note", default="default (15/min, 100/hr)", help="recorded in the output")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    types = {t.strip() for t in args.types.split(",")}
    pool = [q["query"] for q in C.load_json(args.questions)
            if q["type"] in types and not q.get("history")]
    if not pool:
        print(f"No questions of type {sorted(types)} in {args.questions}.")
        return 2
    rng = random.Random(args.seed)
    pick = lambda: rng.choice(pool)  # noqa: E731

    for _ in range(args.warmup):
        one_request(args.base_url, pick(), args.timeout)
        time.sleep(args.sleep)

    passes = {}
    seq = []
    for i in range(args.n):
        seq.append(one_request(args.base_url, pick(), args.timeout))
        if i < args.n - 1:
            time.sleep(args.sleep)
    passes["sequential"] = {"summary": summarize(seq), "samples": seq}
    print_summary(f"Sequential (n={args.n}, sleep={args.sleep}s)", passes["sequential"]["summary"])

    if args.concurrency > 1:
        queries = [pick() for _ in range(args.n)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
            conc = list(pool_exec.map(lambda q: one_request(args.base_url, q, args.timeout), queries))
        passes["concurrent"] = {"summary": summarize(conc), "samples": conc, "concurrency": args.concurrency}
        print_summary(f"Concurrent (n={args.n}, concurrency={args.concurrency})", passes["concurrent"]["summary"])

    path = C.write_results("latency", {
        "base_url": args.base_url, "n": args.n, "sleep_seconds": args.sleep, "warmup": args.warmup,
        "question_types": sorted(types), "rate_limit_note": args.limit_note, "passes": passes,
    })
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
