import hashlib

from backend import config, rag


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------
def test_short_paragraphs_stay_whole_one_chunk_each():
    text = "First short paragraph.\n\nSecond short paragraph, also tiny."
    assert rag.chunk_text(text, max_words=50, overlap_words=0) == [
        "First short paragraph.",
        "Second short paragraph, also tiny.",
    ]


def test_blank_lines_and_whitespace_never_produce_empty_chunks():
    chunks = rag.chunk_text(
        "\n\n  Alpha beta.  \n\n\n\n  \n\nGamma delta.\n\n", 50, 0)
    assert chunks == ["Alpha beta.", "Gamma delta."]


def test_long_paragraph_splits_on_sentence_boundaries_only():
    # 6 words each
    sentences = [f"Sentence number {i} ends right here." for i in range(10)]
    chunks = rag.chunk_text(" ".join(sentences), max_words=20, overlap_words=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 20
        # never starts mid-sentence
        assert chunk.startswith("Sentence number")
        assert chunk.endswith("here.")               # never ends mid-sentence
    # nothing lost, nothing duplicated (overlap is 0)
    assert " ".join(chunks).split() == " ".join(sentences).split()


def test_overlap_carries_trailing_words_into_next_chunk():
    sentences = [f"Sentence number {i} ends right here." for i in range(10)]
    chunks = rag.chunk_text(" ".join(sentences), max_words=20, overlap_words=3)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[:3] == prev.split()[-3:]
    assert all(len(c.split()) <= 20 for c in chunks)


def test_single_oversized_sentence_falls_back_to_word_slices():
    # no punctuation -> one 50-word "sentence"
    words = [f"w{i}" for i in range(50)]
    chunks = rag.chunk_text(" ".join(words), max_words=20, overlap_words=5)

    assert all(len(c.split()) <= 20 for c in chunks)
    assert chunks[0].split() == words[:20]
    assert chunks[1].split()[:5] == words[15:20]    # 5-word overlap
    assert set(" ".join(chunks).split()) == set(words)


def test_paragraphs_are_not_merged_across_boundaries():
    text = "Topic A is about apples.\n\nTopic B is about bananas."
    chunks = rag.chunk_text(text, max_words=100, overlap_words=0)
    assert len(chunks) == 2 and "bananas" not in chunks[0]


# --------------------------------------------------------------------------
# index fingerprint / rebuild behaviour
# --------------------------------------------------------------------------
def test_fingerprint_is_sha256_of_info_txt_and_changes_with_it(isolated_env):
    before = rag._info_source_fingerprint()
    with open(isolated_env.info_path, "rb") as f:
        assert before == hashlib.sha256(f.read()).hexdigest()

    with open(isolated_env.info_path, "a", encoding="utf-8") as f:
        f.write("\n\nA brand new paragraph about Rust.")
    assert rag._info_source_fingerprint() != before


def test_editing_info_txt_triggers_rebuild(isolated_env, monkeypatch):
    calls = []
    real_build = rag._build_index_from_source

    def spy():
        calls.append(1)
        return real_build()

    monkeypatch.setattr(rag, "_build_index_from_source", spy)

    rag.get_index()                      # cold start: nothing cached -> build
    assert len(calls) == 1
    rag.get_index()                      # unchanged -> served from memory
    assert len(calls) == 1

    with open(isolated_env.info_path, "a", encoding="utf-8") as f:
        f.write("\n\nA brand new paragraph about Rust.")

    _index, chunks = rag.get_index()     # fingerprint differs -> rebuild
    assert len(calls) == 2
    assert any("Rust" in c for c in chunks)


def test_unchanged_info_txt_loads_persisted_index_without_rebuild(monkeypatch):
    rag.get_index()                      # builds + writes index, mapping, fingerprint

    # simulate a fresh process: in-memory caches gone, files on disk remain
    monkeypatch.setattr(rag, "_index_cache", None)
    monkeypatch.setattr(rag, "_chunks_cache", None)
    monkeypatch.setattr(rag, "_index_source_fingerprint", None)

    def must_not_rebuild():
        raise AssertionError(
            "index was rebuilt although info.txt is unchanged")

    monkeypatch.setattr(rag, "_build_index_from_source", must_not_rebuild)
    index, chunks = rag.get_index()
    assert index.ntotal == len(chunks) > 0


def test_corrupt_chunk_mapping_triggers_rebuild(isolated_env, monkeypatch):
    rag.get_index()
    with open(config.CHUNK_MAPPING_PATH, "wb") as f:
        f.write(b"not a pickle")
    monkeypatch.setattr(rag, "_index_cache", None)
    monkeypatch.setattr(rag, "_chunks_cache", None)
    monkeypatch.setattr(rag, "_index_source_fingerprint", None)

    index, chunks = rag.get_index()      # must recover, not crash
    assert index.ntotal == len(chunks) > 0


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def test_known_query_returns_expected_top_chunk():
    index, chunks = rag.get_index()
    top = rag.retrieve(
        "Which vector database does ScrubGPT use?", index, chunks, k=1)
    assert len(top) == 1 and "Qdrant" in top[0]


def test_retrieve_with_ids_matches_retrieve_and_ids_index_the_mapping():
    index, chunks = rag.get_index()
    q = "Which tools does the portfolio chatbot HireMeAI use?"
    with_ids = rag.retrieve_with_ids(q, index, chunks, k=3)

    assert [t for _i, t in with_ids] == rag.retrieve(q, index, chunks, k=3)
    for i, text in with_ids:
        assert chunks[i] == text
    assert "HireMeAI" in with_ids[0][1]


def test_retrieve_respects_k():
    index, chunks = rag.get_index()
    assert len(rag.retrieve("python", index, chunks, k=2)) == 2


# --------------------------------------------------------------------------
# scope gate + prompt
# --------------------------------------------------------------------------
def test_scope_gate_returns_false_when_llm_says_no(mock_llm):
    mock_llm.when("scope gate", "NO")
    assert rag.is_relevant_query(
        "write me a poem about clouds", has_history=False) is False


def test_scope_gate_returns_true_when_llm_says_yes(mock_llm):
    mock_llm.when("scope gate", "YES")
    assert rag.is_relevant_query(
        "what are your skills", has_history=False) is True


def test_scope_gate_fails_open_when_llm_errors(mock_llm):
    mock_llm.error = RuntimeError("groq down")
    assert rag.is_relevant_query("anything", has_history=False) is True


def test_answer_prompt_contains_context_question_and_history():
    prompt = rag.build_answer_prompt(
        ["Chunk about Qdrant."], "What DB?", [{"role": "user", "content": "earlier question"}])
    assert "Chunk about Qdrant." in prompt
    assert "What DB?" in prompt
    assert "earlier question" in prompt


def test_age_question_uses_birthdate_in_profile_text():
    profile = "Abhigya Narain was born on 23/04/2003. He completed his schooling..."
    answer = rag.answer_direct_profile_fact("how old are you?", profile)
    assert answer is not None
    assert "23/04/2003" in answer
    assert "23" in answer or "years old" in answer.lower()
