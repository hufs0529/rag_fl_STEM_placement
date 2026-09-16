from src.nq_data import AnswerIndex, build_eval_set, question_id, split_usable


QUESTIONS = [
    {"qid": "a", "question": "who wrote hamlet", "answers": ["William Shakespeare"]},
    {"qid": "b", "question": "capital of france", "answers": ["Paris", "Paris, France"]},
]


def test_question_id_is_stable_under_whitespace_and_case():
    assert question_id(" Who Wrote Hamlet ") == question_id("who wrote hamlet")


def test_answer_index_matches_normalised_multi_token_answers():
    # 1단계(정답 문자열 포함)만 검증하므로 겹침 필터를 끈다
    index = AnswerIndex(QUESTIONS, min_question_overlap=0)
    assert index.match("Hamlet was written by William Shakespeare.") == ["a"]
    assert index.match("The Eiffel Tower stands in Paris.") == ["b"]
    assert index.match("An unrelated sentence.") == []


def test_answer_index_reports_each_question_once():
    index = AnswerIndex(QUESTIONS, min_question_overlap=0)
    assert index.match("Paris. Paris, France. Paris again.") == ["b"]


def test_overlong_answers_are_skipped_rather_than_indexed():
    long_answer = [{"qid": "c", "question": "q", "answers": [" ".join(["word"] * 20)]}]
    index = AnswerIndex(long_answer, max_answer_tokens=8, min_question_overlap=0)
    assert len(index) == 0
    assert index.skipped_answers == 1


def test_split_usable_discards_questions_with_no_gold_passage():
    usable, discarded = split_usable(QUESTIONS, {"a": ["p1", "p2"]})
    assert [q["qid"] for q in usable] == ["a"]
    assert usable[0]["gold_passage_ids"] == ["p1", "p2"]
    assert [q["qid"] for q in discarded] == ["b"]


def test_eval_set_is_identical_across_calls_and_independent_of_run_seed():
    pool = [{"qid": f"q{i}", "question": str(i), "answers": ["x"]} for i in range(100)]
    first = build_eval_set(pool, size=10)
    second = build_eval_set(pool, size=10)
    assert [q["qid"] for q in first] == [q["qid"] for q in second]
    assert len(first) == 10


# --- 2단계 gold 판정 (Subtask 1.1) -----------------------------------------

from src.nq_data import content_terms

OVERLAP_QUESTIONS = [
    {"qid": "yr", "question": "when was puerto rico added to the usa", "answers": ["1950"]},
    {"qid": "wr", "question": "who wrote the novel hamlet", "answers": ["William Shakespeare"]},
]


def test_content_terms_drop_question_words_and_short_tokens():
    # 의문사가 남으면 어떤 passage 든 겹쳐버려 필터가 무력해진다
    assert content_terms("when was puerto rico added to the usa") == {
        "puerto", "rico", "added", "usa"}


def test_a_single_token_answer_alone_is_not_enough_to_be_gold():
    # NQ 정답의 69% 가 한 토큰이라, 포함만 보면 코퍼스의 98% 가 gold 가 된다
    index = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=2)
    kept, raw = index.match_detailed("The 1950 census of rural Norway recorded growth.")
    assert raw == 1          # 1단계는 통과했고
    assert kept == []        # 2단계에서 걸러졌다


def test_a_genuinely_related_passage_still_counts_as_gold():
    index = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=2)
    assert index.match("Puerto Rico was added to the usa territories in 1950.") == ["yr"]


def test_the_title_contributes_to_the_overlap():
    index = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=2)
    assert index.match("The territory joined in 1950.", title="Puerto Rico history") == ["yr"]


def test_disabling_the_filter_restores_the_permissive_behaviour():
    index = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=0)
    assert index.match("The 1950 census of rural Norway recorded growth.") == ["yr"]


def test_raising_the_threshold_tightens_the_criterion():
    loose = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=1)
    strict = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=3)
    passage = "Puerto history records the year 1950."
    assert loose.match(passage) == ["yr"]
    assert strict.match(passage) == []


def test_match_detailed_reports_how_many_were_rejected():
    index = AnswerIndex(OVERLAP_QUESTIONS, min_question_overlap=2)
    kept, raw = index.match_detailed("In 1950 William Shakespeare was quoted in Norway.")
    assert raw == 2 and kept == []
