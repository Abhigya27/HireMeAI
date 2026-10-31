"""1.4 JD matcher: run-to-run score variance, and agreement with your own ratings.

    python -m eval.jd_matcher_eval --mode variance
    python -m eval.jd_matcher_eval --mode agreement
    python -m eval.jd_matcher_eval --mode both            # default

Part A (variance): the first --variance-count JDs in eval/sample_jds/ are each
scored --runs times; reports the sample standard deviation (and range) of
`final_score` per JD and averaged over JDs.

Part B (agreement): for each entry in eval/jd_manual_ratings.json with a
manual_score, the pipeline scores the JD once. Pipeline verdict is derived from
the score with the same bands the UI shows (>=75 strong_fit, >=50 moderate_fit,
else weak_fit), so manual_verdict must be one of strong_fit / moderate_fit /
weak_fit. Fill the ratings in BEFORE looking at pipeline output.

Transport: by default it calls POST /match on --base-url, which is subject to
the /match rate limit (5/min, 20/hr by default -- 25 variance runs will not fit
in an hour). Either start a local backend with MATCH_RATE_LIMIT_SHORT /
MATCH_RATE_LIMIT_LONG raised, or use --direct to call backend.ats.score_match
in-process (no HTTP, no limits; needs GROQ_API_KEY locally).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

from eval import _common as C

# Wire-format contract with backend/ats.py (separate process over HTTP).
RESULT_MARKER = "<<<JD_MATCH_RESULT>>>"
JD_EXTS = {".txt", ".pdf", ".docx"}


def score_http(base_url: str, path: Path, timeout: float) -> dict:
    files = {"file": (path.name, path.read_bytes(), "application/octet-stream")}
    buffer = ""
    with httpx.stream("POST", f"{base_url}/match", files=files, timeout=timeout) as resp:
        if resp.status_code != 200:
            resp.read()
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        for chunk in resp.iter_text():
            buffer += chunk
    _analysis, marker, tail = buffer.partition(RESULT_MARKER)
    if not marker:
        return {"error": "stream ended without a result marker"}
    return json.loads(tail)


def score_direct(path: Path) -> dict:
    from backend.ats import extract_text, score_match
    return score_match(extract_text(path.name, path.read_bytes()))


def make_scorer(args):
    if args.direct:
        return lambda p: score_direct(p)
    return lambda p: score_http(args.base_url, p, args.timeout)


def numeric_score(result: dict) -> float | None:
    s = result.get("final_score")
    return float(s) if isinstance(s, (int, float)) else None


def score_stats(scores: list[float]) -> dict:
    """Spread of repeated scores for one JD (sample std dev, needs >=2 values)."""
    if not scores:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "range": None}
    return {
        "n": len(scores), "mean": round(statistics.mean(scores), 2),
        "std": round(statistics.stdev(scores), 2) if len(scores) >= 2 else None,
        "min": min(scores), "max": max(scores), "range": max(scores) - min(scores),
    }


def run_variance(args, scorer):
    jd_dir = Path(args.jd_dir)
    jds = sorted(p for p in jd_dir.iterdir() if p.suffix.lower() in JD_EXTS)[:args.variance_count]
    if not jds:
        print(f"No JD files found in {jd_dir}. Add .txt/.pdf/.docx files first.")
        return None

    per_jd = []
    for jd in jds:
        scores, failures = [], 0
        for r in range(args.runs):
            try:
                result = scorer(jd)
                s = numeric_score(result)
            except Exception as e:
                print(f"  {jd.name} run {r + 1}: error {e!r}")
                s = None
            if s is None:
                failures += 1
            else:
                scores.append(s)
            if args.sleep:
                time.sleep(args.sleep)
        stats = score_stats(scores)
        per_jd.append({"jd_file": str(jd), "scores": scores, "failures": failures, **stats})
        print(f"  {jd.name}: scores={scores} std={stats['std']} range={stats['range']} failures={failures}")

    stds = [j["std"] for j in per_jd if j["std"] is not None]
    summary = {
        "jds": len(per_jd), "runs_per_jd": args.runs,
        "std_definition": "sample standard deviation (n-1) of final_score across runs, per JD",
        "avg_std": round(statistics.mean(stds), 2) if stds else None,
        "max_std": max(stds) if stds else None,
        "max_range": max((j["range"] for j in per_jd if j["range"] is not None), default=None),
        "total_failed_runs": sum(j["failures"] for j in per_jd),
    }
    print("\nVariance")
    C.print_table(["JD", "mean", "std", "range", "failed"],
                  [[Path(j["jd_file"]).name, j["mean"], j["std"], j["range"], j["failures"]] for j in per_jd])
    print(f"avg std across JDs = {summary['avg_std']}   (max std {summary['max_std']}, max range {summary['max_range']})")
    return {"summary": summary, "per_jd": per_jd}


def run_agreement(args, scorer):
    ratings = [r for r in C.load_json(args.ratings) if r.get("manual_score") is not None]
    if not ratings:
        print(f"No manual_score values filled in {args.ratings}. Rate the JDs first (before looking at pipeline output).")
        return None

    rows = []
    for r in ratings:
        path = Path(r["jd_file"])
        if not path.exists():
            print(f"  skipping {path}: file not found")
            continue
        try:
            s = numeric_score(scorer(path))
        except Exception as e:
            print(f"  {path.name}: error {e!r}")
            s = None
        if s is None:
            rows.append({"jd_file": r["jd_file"], "manual_score": r["manual_score"], "pipeline_score": None})
        else:
            rows.append({
                "jd_file": r["jd_file"], "manual_score": r["manual_score"], "pipeline_score": s,
                "abs_error": abs(s - r["manual_score"]),
                "within_10": abs(s - r["manual_score"]) <= 10,
                "manual_verdict": r.get("manual_verdict"),
                "pipeline_verdict": C.verdict_from_score(s),
                "verdict_match": r.get("manual_verdict") == C.verdict_from_score(s),
            })
        if args.sleep:
            time.sleep(args.sleep)

    scored = [x for x in rows if x.get("pipeline_score") is not None]
    if not scored:
        print("No JD produced a score.")
        return None
    with_verdict = [x for x in scored if x["manual_verdict"] in C.VERDICTS]
    summary = {
        "n_rated": len(rows), "n_scored": len(scored),
        "mean_absolute_error": round(statistics.mean(x["abs_error"] for x in scored), 2),
        "pct_within_10": C.ratio(sum(x["within_10"] for x in scored), len(scored)),
        "pct_verdict_match": C.ratio(sum(x["verdict_match"] for x in with_verdict), len(with_verdict)),
        "n_with_manual_verdict": len(with_verdict),
    }
    print("\nAgreement with manual ratings")
    C.print_table(["JD", "manual", "pipeline", "|diff|", "verdicts"], [
        [Path(x["jd_file"]).name, x["manual_score"], x["pipeline_score"], x["abs_error"],
         f"{x['manual_verdict']} / {x['pipeline_verdict']}"] for x in scored])
    print(f"MAE = {summary['mean_absolute_error']}   within +/-10 = {C.fmt_pct(summary['pct_within_10'])}   "
          f"verdict agreement = {C.fmt_pct(summary['pct_verdict_match'])}")
    return {"summary": summary, "per_jd": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["variance", "agreement", "both"], default="both")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--direct", action="store_true", help="call backend.ats in-process instead of HTTP")
    ap.add_argument("--jd-dir", default="eval/sample_jds")
    ap.add_argument("--ratings", default="eval/jd_manual_ratings.json")
    ap.add_argument("--variance-count", type=int, default=5, help="how many JDs for the variance test")
    ap.add_argument("--runs", type=int, default=5, help="repeats per JD in the variance test")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between /match calls (5/min limit => 13)")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    scorer = make_scorer(args)
    out = {"transport": "direct" if args.direct else args.base_url}
    if args.mode in ("variance", "both"):
        print("Part A: variance")
        out["variance"] = run_variance(args, scorer)
    if args.mode in ("agreement", "both"):
        print("\nPart B: agreement")
        out["agreement"] = run_agreement(args, scorer)

    if not any(out.get(k) for k in ("variance", "agreement")):
        return 2
    print(f"\nwrote {C.write_results('jd_matcher', out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
