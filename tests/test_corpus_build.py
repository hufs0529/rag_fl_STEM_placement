import pytest

from src.corpus_build import (
    bm25_hard_negatives, build_corpus, collect_gold_passages,
    fill_random_remainder, verify_no_answer_bearing_negatives,
)
from src.data import PassageStore


def _store():
    passages = [
        {"pid": "g0", "title": "France", "text": "The capital of France is Paris."},
        {"pid": "g1", "title": "Paris", "text": "Paris has been the capital since 987."},
        # 어휘적으로 가깝지만 정답이 없는 passage - 진짜 hard negative
        {"pid": "n0", "title": "France", "text": "The capital of France moved several times."},
        {"pid": "n1", "title": "Capitals", "text": "A capital city hosts the government of France."},
    ]
    passages += [
        {"pid": f"r{i}", "title": "Misc", "text": f"unrelated filler sentence number {i}"}
        for i in range(20)
    ]
    return PassageStore(passages)


QUESTIONS = [{
    "qid": "q0", "question": "what is the capital of france",
    "answers": ["Paris"], "gold_passage_ids": ["g0", "g1"],
}]


def test_gold_passages_are_collected_without_duplicates():
    store = _store()
    questions = QUESTIONS + [{**QUESTIONS[0], "qid": "q1"}]
    assert collect_gold_passages(questions, store) == ["g0", "g1"]


def test_hard_negatives_never_contain_the_answer_string():
    store = _store()
    negatives = bm25_hard_negatives(
        QUESTIONS, store, [p for p in store.ids() if not p.startswith("g")], per_question=3
    )
    assert negatives
    assert "g0" not in negatives and "g1" not in negatives
    for pid in negatives:
        assert "paris" not in store.get(pid)["text"].lower()


def test_hard_negatives_are_lexically_close_not_random():
    store = _store()
    negatives = bm25_hard_negatives(
        QUESTIONS, store, [p for p in store.ids() if not p.startswith("g")], per_question=2
    )
    # BM25 상위는 "capital"/"France" 를 공유하는 n0/n1 이어야 한다
    assert set(negatives) <= {"n0", "n1"}


def test_random_remainder_fills_to_the_target_size():
    store = _store()
    filled = fill_random_remainder(["g0"], store.ids(), target_size=10)
    assert len(filled) == 10
    assert filled[0] == "g0"
    assert len(set(filled)) == 10


def test_remainder_never_exceeds_the_available_pool():
    filled = fill_random_remainder(["a"], ["a", "b", "c"], target_size=100)
    assert len(filled) == 3


def test_build_corpus_reports_its_composition():
    result = build_corpus(QUESTIONS, _store(), target_size=12,
                          hard_negatives_per_question=2, gold_per_question=None,
                          log=lambda *a: None)
    comp = result["composition"]
    assert comp["gold"] == 2
    assert comp["total"] == 12
    assert comp["gold"] + comp["hard_negatives"] + comp["random_remainder"] == comp["total"]


# --- 예산 배분과 비용 상한 -------------------------------------------------

def _many_questions(n=10):
    return [
        {"qid": f"q{i}", "question": "what is the capital of france",
         "answers": ["Paris"], "gold_passage_ids": ["g0", "g1"]}
        for i in range(n)
    ]


def test_gold_is_capped_per_question_so_it_cannot_eat_the_budget():
    # 상한이 없으면 6만 질문 x 최대 20 gold 가 20만 예산을 혼자 다 먹는다
    store = _store()
    assert len(collect_gold_passages(QUESTIONS, store, max_per_question=1)) == 1
    assert len(collect_gold_passages(QUESTIONS, store, max_per_question=None)) == 2


def test_gold_exceeding_the_budget_is_an_error_not_a_silent_overshoot():
    store = _store()
    with pytest.raises(ValueError, match="exceed the corpus budget"):
        build_corpus(QUESTIONS, store, target_size=1, gold_per_question=None,
                     log=lambda *a: None)


def test_remainder_comes_from_the_answer_free_pool_only():
    store = _store()
    random_pool = [pid for pid in store.ids() if pid.startswith("r")]
    result = build_corpus(QUESTIONS, store, target_size=10, hard_negatives_per_question=2,
                          gold_per_question=1, random_pool=random_pool, log=lambda *a: None)
    comp = result["composition"]
    chosen = result["passage_ids"][comp["gold"] + comp["hard_negatives"]:]
    # remainder 에는 gold 가 섞이면 안 된다 - 섞이면 recall 이 false positive 를 낸다
    assert all(pid.startswith("r") for pid in chosen)


def test_a_pool_too_small_to_fill_the_corpus_is_reported_not_hidden():
    store = _store()
    result = build_corpus(QUESTIONS, store, target_size=500, hard_negatives_per_question=1,
                          gold_per_question=1, random_pool=["r0", "r1"], log=lambda *a: None)
    assert result["composition"]["shortfall"] > 0


def test_bm25_question_cap_bounds_the_cost():
    # BM25 는 질의마다 코퍼스 전체를 훑으므로 질문 수가 그대로 비용이 된다
    store = _store()
    result = build_corpus(_many_questions(10), store, target_size=12,
                          hard_negatives_per_question=1, gold_per_question=1,
                          hard_negative_questions=3, log=lambda *a: None)
    assert result["composition"]["hard_negative_questions"] == 3


def test_bm25_pool_cap_shrinks_the_index():
    store = _store()
    candidates = [pid for pid in store.ids() if pid.startswith("r")]
    negatives = bm25_hard_negatives(QUESTIONS, store, candidates, per_question=2,
                                    pool_size=5, log=lambda *a: None)
    # 상한 안에서만 뽑히고, 정답은 여전히 들어가지 않는다
    assert len(negatives) <= 2
    assert all(not pid.startswith("g") for pid in negatives)


def test_bm25_with_no_questions_or_no_candidates_returns_nothing():
    store = _store()
    assert bm25_hard_negatives([], store, store.ids(), log=lambda *a: None) == []
    assert bm25_hard_negatives(QUESTIONS, store, [], log=lambda *a: None) == []


def test_verification_flags_an_answer_bearing_negative():
    violations = verify_no_answer_bearing_negatives(QUESTIONS, ["g0", "n0"], _store())
    assert [v["pid"] for v in violations] == ["g0"]


def test_gold_plus_negatives_over_budget_is_an_error_not_a_vanished_remainder():
    """gold 단독은 예산 안이지만 negative 를 더하면 넘는 경우.

    예전에는 조용히 넘어가서 remainder 가 0 이 되고 코퍼스가 목표를 초과했다 -
    remainder 를 따로 모은 이유 자체가 무력해지는 실패다.
    """
    import pytest

    store = _store()
    with pytest.raises(ValueError, match="no random remainder"):
        build_corpus(QUESTIONS, store, target_size=3, hard_negatives_per_question=3,
                     gold_per_question=2, random_pool=[f"r{i}" for i in range(20)],
                     log=lambda *a: None)


def test_the_default_gold_per_question_leaves_room_for_a_remainder():
    # 설정 기본값이 실제 질문 규모에서 예산에 맞는지 - 산술로 확인
    from src.config import load_config
    cfg = load_config()
    questions, negatives = 91_535, cfg.data.hard_negative_questions * cfg.data.hard_negatives_per_question
    gold = questions * cfg.data.gold_per_question_in_corpus
    assert gold + negatives < cfg.data.corpus_size, (
        f"gold {gold:,} + negatives {negatives:,} >= budget {cfg.data.corpus_size:,}")


def test_the_budget_guard_names_question_limit_first():
    """축소 리허설에서 corpus_size 만 줄이면 질문 수가 그대로여서 가드가 걸린다.

    메시지가 'gold_per_question 을 낮춰라' 만 말하면, 실제로 필요한 손잡이
    (question_limit) 를 찾지 못한다. 실측으로 --dev 가 여기서 멈췄다.
    """
    import pytest
    from src.corpus_build import build_corpus
    from src.data import PassageStore

    store = PassageStore([{"pid": str(i), "title": "t", "text": f"x{i}"} for i in range(50)])
    questions = [{"qid": f"q{i}", "question": "q", "answers": ["a"],
                  "gold_passage_ids": [str(i)]} for i in range(20)]
    with pytest.raises(ValueError) as e:
        build_corpus(questions, store, target_size=5, gold_per_question=1,
                     random_pool=[str(i) for i in range(30, 50)], log=lambda *_a: None)
    msg = str(e.value)
    assert "data.question_limit" in msg
    assert msg.index("data.question_limit") < msg.index("data.gold_per_question_in_corpus")
    assert "20 questions" in msg
