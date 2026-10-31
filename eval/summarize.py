"""1.9 Print the README "Results" table from the newest files in eval/results/.

    python -m eval.summarize

Every number comes from a results JSON (and `pytest --collect-only` for the
test count), so the README never contains a figure that wasn't measured.
Paste the output under "## Results" in README.md. Anything not yet measured
shows as "not run".
"""
import re
import subprocess
import sys

from eval import _common as C


def latest(prefix: str) -> dict | None:
    files = sorted(C.RESULTS_DIR.glob(f"{prefix}_*.json"))
    return C.load_json(files[-1]) if files else None


def test_count() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                             cwd=C.REPO_ROOT, capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"(\d+) tests? collected", out)
        return m.group(1) if m else "not run"
    except Exception:
        return "not run"


def f(x, suffix="", digits=2):
    return "not run" if x is None else (f"{x:.{digits}f}{suffix}" if isinstance(x, float) else f"{x}{suffix}")


def main() -> int:
    ret, faith, lat = latest("retrieval"), latest("faithfulness"), latest("latency")
    jd, agent = latest("jd_matcher"), latest("agent")

    rows = []
    if ret:
        rows.append((f"Retrieval hit-rate@{ret['k']}", C.fmt_pct(ret["overall_hit_rate"]),
                     f"{ret['n_evaluated']} questions"))
    if faith:
        rows.append(("Answers grounded in retrieved context", C.fmt_pct(faith["pct_grounded"]),
                     f"{faith['n_in_scope']} in-scope questions, judge {faith['judge_model']}"))
        rows.append(("Out-of-scope questions correctly declined", C.fmt_pct(faith["pct_correct_refusals"]),
                     f"{faith['n_out_of_scope']} questions"))
    if lat:
        seq = lat["passes"]["sequential"]["summary"]
        if seq["ttft"]:
            rows.append(("TTFT p50 / p95 (/chat, streaming)",
                         f"{seq['ttft']['p50']:.2f}s / {seq['ttft']['p95']:.2f}s",
                         f"{seq['ok']} requests; limits: {lat['rate_limit_note']}"))
    if jd and jd.get("variance"):
        v = jd["variance"]["summary"]
        rows.append(("JD score std dev across repeat runs (avg)", f(v["avg_std"]),
                     f"{v['jds']} JDs x {v['runs_per_jd']} runs"))
    if jd and jd.get("agreement"):
        a = jd["agreement"]["summary"]
        rows.append(("JD score within +/-10 of manual rating", C.fmt_pct(a["pct_within_10"]),
                     f"{a['n_scored']} JDs, MAE {a['mean_absolute_error']}"))
        rows.append(("JD verdict agreement with manual rating", C.fmt_pct(a["pct_verdict_match"]),
                     f"{a['n_with_manual_verdict']} JDs"))
    if agent:
        rows.append(("GitHub agent: avg tool calls / answer", f(agent["avg_tool_calls"]),
                     f"{agent['answered']} questions"))
        rows.append(("GitHub agent: expected file read", C.fmt_pct(agent["pct_expected_file_read"]), ""))
    rows.append(("pytest tests", test_count(), ""))

    commits = {d["commit"][:8] for d in (ret, faith, lat, jd, agent) if d}
    dates = sorted(d["generated_at"][:10] for d in (ret, faith, lat, jd, agent) if d)
    print("## Results\n")
    print(f"_Measured {dates[0] if dates else 'n/a'}"
          f"{' – ' + dates[-1] if len(set(dates)) > 1 else ''}, "
          f"commit {', '.join(sorted(commits)) or 'n/a'}. Scripts: `eval/`; raw output: `eval/results/`._\n")
    print("| Metric | Result | Set / notes |\n|---|---|---|")
    for metric, value, notes in rows:
        print(f"| {metric} | {value} | {notes} |")
    missing = [n for n, d in [("retrieval", ret), ("faithfulness", faith), ("latency", lat),
                              ("jd_matcher", jd), ("agent", agent)] if not d]
    if missing:
        print(f"\n_Not yet run: {', '.join(missing)}._")
    return 0


if __name__ == "__main__":
    sys.exit(main())
