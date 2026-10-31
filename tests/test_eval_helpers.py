"""The eval scripts produce the numbers that go on the resume, so their own
arithmetic and bookkeeping is tested here (offline, no live LLM)."""
import json
from argparse import Namespace
from pathlib import Path

import pytest

from eval import _common as C
from eval import jd_matcher_eval, latency_bench, retrieval_eval

FIXTURE_INFO = str(Path(__file__).parent / "fixtures" / "info.txt")


# --------------------------------------------------------------------------
# _common
# --------------------------------------------------------------------------
def test_expected_list_accepts_none_string_and_list():
    assert C.expected_list({"expected_substring": None}) == []
    assert C.expected_list({}) == []
    assert C.expected_list({"expected_substring": "Qdrant"}) == ["Qdrant"]
    assert C.expected_list({"expected_substring": ["a", "", "b"]}) == ["a", "b"]


def test_substring_hit_is_case_insensitive_and_any_of():
    assert C.substring_hit(["qdrant"], ["Stores data in QDRANT."])
    assert C.substring_hit(["nope", "faiss"], ["x", "uses FAISS"])
    assert not C.substring_hit(["rust"], ["python only"])


@pytest.mark.parametrize("score,verdict", [
    (100, "strong_fit"), (75, "strong_fit"), (74.9, "moderate_fit"),
    (50, "moderate_fit"), (49, "weak_fit"), (0, "weak_fit")])
def test_verdict_bands_match_the_ui_thresholds(score, verdict):
    assert C.verdict_from_score(score) == verdict


def test_parse_json_object_handles_fences_and_prose():
    assert C.parse_json_object('```json\n{"verdict": "GROUNDED"}\n```') == {"verdict": "GROUNDED"}
    assert C.parse_json_object("no json here") is None
    assert C.parse_json_object("{broken") is None


def test_write_results_stamps_commit_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    path = C.write_results("unit", {"x": 1})
    body = json.loads(path.read_text())
    assert body["x"] == 1 and body["generated_at"] and "commit" in body
    assert path.name.startswith("unit_") and path.suffix == ".json"


# --------------------------------------------------------------------------
# latency math
# --------------------------------------------------------------------------
def test_percentiles_use_numpy_semantics():
    p = latency_bench.percentiles([float(i) for i in range(1, 101)])
    assert p == {"p50": 50.5, "p95": 95.05, "p99": 99.01}
    assert latency_bench.percentiles([]) is None


def test_latency_summary_separates_429s_from_other_failures():
    samples = [
        {"ok": True, "status": 200, "ttft": 0.5, "total": 2.0},
        {"ok": True, "status": 200, "ttft": 1.5, "total": 4.0},
        {"ok": False, "status": 429, "ttft": None, "total": 0.01},
        {"ok": False, "status": None, "ttft": None, "total": 3.0},
    ]
    s = latency_bench.summarize(samples)
    assert (s["requests"], s["ok"], s["rate_limited_429"], s["other_failures"]) == (4, 2, 1, 1)
    assert s["ttft"]["p50"] == 1.0          # failures excluded from the percentiles


# --------------------------------------------------------------------------
# JD matcher stats
# --------------------------------------------------------------------------
def test_score_stats_uses_sample_std_dev():
    s = jd_matcher_eval.score_stats([80, 82, 78])
    assert s["mean"] == 80 and s["std"] == 2.0 and s["range"] == 4
    assert jd_matcher_eval.score_stats([80])["std"] is None
    assert jd_matcher_eval.score_stats([])["n"] == 0


def test_numeric_score_ignores_failed_results():
    assert jd_matcher_eval.numeric_score({"final_score": 71}) == 71.0
    assert jd_matcher_eval.numeric_score({"final_score": None, "error": "x"}) is None


def test_agreement_metrics(tmp_path):
    files = []
    for name in ("a.txt", "b.txt", "c.txt"):
        p = tmp_path / name
        p.write_text("jd")
        files.append(p)
    ratings = tmp_path / "ratings.json"
    ratings.write_text(json.dumps([
        {"jd_file": str(files[0]), "manual_score": 80, "manual_verdict": "strong_fit"},
        {"jd_file": str(files[1]), "manual_score": 60, "manual_verdict": "moderate_fit"},
        {"jd_file": str(files[2]), "manual_score": 30, "manual_verdict": "weak_fit"},
        {"jd_file": str(tmp_path / "unrated.txt"), "manual_score": None, "manual_verdict": None},
    ]))
    pipeline = {"a.txt": 85, "b.txt": 40, "c.txt": 32}          # diffs 5, 20, 2
    out = jd_matcher_eval.run_agreement(
        Namespace(ratings=str(ratings), sleep=0),
        lambda p: {"final_score": pipeline[p.name]})

    s = out["summary"]
    assert s["n_scored"] == 3
    assert s["mean_absolute_error"] == pytest.approx((5 + 20 + 2) / 3, abs=0.01)
    assert s["pct_within_10"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["pct_verdict_match"] == pytest.approx(2 / 3, abs=1e-3)   # b: moderate vs weak


# --------------------------------------------------------------------------
# retrieval eval
# --------------------------------------------------------------------------
QUESTIONS = [
    {"id": "t1", "type": "project_fact", "query": "Which vector database does ScrubGPT use?",
     "expected_substring": "Qdrant"},
    {"id": "t2", "type": "project_fact", "query": "What does the house price model predict?",
     "expected_substring": ["housing prices"]},
    {"id": "t3", "type": "skill_fact", "query": "Which fish is best?", "expected_substring": "salmon"},
    {"id": "t4", "type": "profile_fact", "query": "unfilled", "expected_substring": None},
    {"id": "t5", "type": "out_of_scope", "query": "poem about clouds", "expected_substring": None},
]


def test_retrieval_run_scores_hits_and_skips_unfilled_and_out_of_scope():
    per_q, unfilled, other = retrieval_eval.run(QUESTIONS, k=3, followups="skip", info_path=FIXTURE_INFO)

    by_id = {r["id"]: r for r in per_q}
    assert set(by_id) == {"t1", "t2", "t3"}
    assert by_id["t1"]["hit"] and by_id["t2"]["hit"]
    assert not by_id["t3"]["hit"]                       # genuine miss is reported as a miss
    assert (unfilled, other) == (1, 1)
    assert all(len(r["retrieved_ids"]) == 3 for r in per_q)


def test_retrieval_run_restores_global_config(isolated_env):
    from backend import config
    before = (config.INFO_TXT_PATH, config.SQLITE_DB_PATH, config.FAISS_INDEX_PATH)
    retrieval_eval.run(QUESTIONS[:1], k=3, followups="skip", info_path=FIXTURE_INFO)
    assert (config.INFO_TXT_PATH, config.SQLITE_DB_PATH, config.FAISS_INDEX_PATH) == before


def test_retrieval_summarize_breaks_down_by_type():
    per_q = [{"type": "a", "hit": True}, {"type": "a", "hit": False}, {"type": "b", "hit": True}]
    overall, by_type, _ = retrieval_eval.summarize(per_q)
    assert overall == pytest.approx(2 / 3, abs=1e-3)
    assert by_type == {"a": 0.5, "b": 1.0}


def test_retrieval_cli_writes_results_and_enforces_min_hit_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(QUESTIONS[:2]))
    base = ["--questions", str(qfile), "--k", "3", "--info-path", FIXTURE_INFO, "--followups", "skip"]

    assert retrieval_eval.main(base + ["--min-hit-rate", "0.5"]) == 0
    written = list((tmp_path / "results").glob("retrieval_*.json"))
    assert len(written) == 1
    body = json.loads(written[0].read_text())
    assert body["overall_hit_rate"] == 1.0 and body["k"] == 3 and body["commit"]

    assert retrieval_eval.main(base + ["--min-hit-rate", "1.01"]) == 1


def test_retrieval_cli_refuses_to_report_when_nothing_is_filled_in(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps([QUESTIONS[3]]))
    assert retrieval_eval.main(["--questions", str(qfile), "--info-path", FIXTURE_INFO]) == 2
    assert not (tmp_path / "results").exists()
