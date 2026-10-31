import json

import pytest

from backend import ats, config
from backend.ats import RESULT_MARKER
from conftest import DEFAULT_ANALYSIS


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ext", ["txt", "pdf", "docx"])
def test_extract_text_returns_nonempty_text_for_each_format(sample_jds, ext):
    name, data = sample_jds[ext]
    text = ats.extract_text(name, data)
    assert text.strip()
    assert "FastAPI" in text


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        ats.extract_text("jd.exe", b"whatever")


def test_extract_text_rejects_missing_filename_and_extension():
    with pytest.raises(ValueError):
        ats.extract_text(None, b"x")
    with pytest.raises(ValueError):
        ats.extract_text("noextension", b"x")


@pytest.mark.parametrize("name", ["bad.pdf", "bad.docx"])
def test_extract_text_turns_corrupt_files_into_valueerror(name):
    with pytest.raises(ValueError):
        ats.extract_text(name, b"this is not really a document")


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------
def test_extract_json_object_handles_nesting_strings_and_surrounding_text():
    raw = 'Sure! Here you go: {"a": {"b": "has } brace and \\" quote"}, "c": [1, 2]} trailing words'
    obj = ats._extract_json_object(raw)
    assert json.loads(obj) == {"a": {"b": 'has } brace and " quote'}, "c": [1, 2]}


def test_extract_json_object_returns_none_when_unbalanced():
    assert ats._extract_json_object('{"a": {"b": 1}') is None
    assert ats._extract_json_object("no json at all") is None


def test_parse_response_salvages_truncated_json():
    raw = ('{"topic_verdicts": [{"topic": "Skills", "verdict": "Strong"}], '
           '"final_verdict": "Good fit, but')
    parsed = ats._parse_comparison_response(raw)
    assert parsed["_parse_failed"] is False
    assert parsed["topic_verdicts"] == [{"topic": "Skills", "verdict": "Strong"}]


def test_parse_response_flags_garbage_as_failed():
    assert ats._parse_comparison_response("sorry, no JSON")["_parse_failed"] is True


def test_build_result_on_parse_failure_gives_error_and_no_fake_score():
    result = ats._build_result(55.5, {"_parse_failed": True})
    assert result["final_score"] is None
    assert result["error"]
    assert result["embedding_score"] == 55.5


def test_build_result_uses_llm_score_not_embedding_score():
    result = ats._build_result(40.0, {**DEFAULT_ANALYSIS, "_parse_failed": False})
    assert result["final_score"] == DEFAULT_ANALYSIS["score"]
    assert result["embedding_score"] == 40.0


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def test_embedding_similarity_is_within_0_100_and_identical_text_is_max():
    same = ats._embedding_similarity("python fastapi sql", "python fastapi sql")
    other = ats._embedding_similarity("python fastapi sql", "welding metal fabrication")
    assert 0 <= other < same <= 100
    assert same == pytest.approx(100.0, abs=0.01)


def test_score_match_stream_ends_with_marker_chunk_carrying_valid_result():
    chunks = list(ats.score_match_stream("Looking for a Python engineer."))
    assert chunks[-1].startswith(RESULT_MARKER)
    assert not any(RESULT_MARKER in c for c in chunks[:-1])

    result = json.loads(chunks[-1][len(RESULT_MARKER):])
    assert result["final_score"] == DEFAULT_ANALYSIS["score"]


def test_score_match_wrapper_returns_structured_result():
    result = ats.score_match("Looking for a Python engineer.")
    assert 0 <= result["final_score"] <= 100
    assert result["final_verdict"]


def test_match_uses_the_stronger_match_model(mock_llm):
    ats.score_match("Looking for a Python engineer.")
    assert mock_llm.calls[-1]["model"] == config.MATCH_MODEL


# --------------------------------------------------------------------------
# /match endpoint
# --------------------------------------------------------------------------
def _post(client, name, data):
    return client.post("/match", files={"file": (name, data, "application/octet-stream")})


@pytest.mark.parametrize("ext", ["txt", "pdf", "docx"])
def test_match_endpoint_returns_structured_result(client, sample_jds, ext):
    name, data = sample_jds[ext]
    resp = _post(client, name, data)

    assert resp.status_code == 200
    analysis_text, marker, result_json = resp.text.partition(RESULT_MARKER)
    assert marker == RESULT_MARKER and analysis_text.strip()

    result = json.loads(result_json)
    assert 0 <= result["final_score"] <= 100
    assert isinstance(result["genuine_gaps"], list)
    assert isinstance(result["transferable_skills"], list)
    assert isinstance(result["topic_verdicts"], list)
    assert isinstance(result["final_verdict"], str) and result["final_verdict"]


def test_match_endpoint_surfaces_parse_failure_as_error_field(client, sample_jds, mock_llm):
    mock_llm.when("supportive, realistic technical reviewer", "I refuse to output JSON.")
    name, data = sample_jds["txt"]
    resp = _post(client, name, data)

    assert resp.status_code == 200
    result = json.loads(resp.text.partition(RESULT_MARKER)[2])
    assert result["final_score"] is None and result["error"]


def test_match_rejects_empty_upload_with_4xx(client):
    resp = _post(client, "empty.txt", b"")
    assert 400 <= resp.status_code < 500


def test_match_rejects_unsupported_type_with_4xx(client):
    resp = _post(client, "jd.exe", b"MZ...")
    assert resp.status_code == 400


@pytest.mark.parametrize("name", ["corrupt.pdf", "corrupt.docx"])
def test_match_rejects_corrupt_file_with_4xx_not_500(client, name):
    resp = _post(client, name, b"definitely not a real document")
    assert 400 <= resp.status_code < 500


def test_match_rejects_oversized_upload_with_413(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    resp = _post(client, "jd.txt", b"x" * 100)
    assert resp.status_code == 413


def test_match_requires_a_file(client):
    assert client.post("/match").status_code == 422
