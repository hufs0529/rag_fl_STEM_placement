import random

import pytest

from src.probe_corpus import plan_probe_corpus, select_question_gold, week2_gold_share


def q(qid, golds):
    return {"qid": qid, "question": f"q{qid}", "gold_passage_ids": list(golds)}


def test_gold_share_is_derived_from_the_measured_question_count():
    # Week 2: 80,720 질문 x gold 1 권 / 20 만 권
    assert week2_gold_share(80_720, 200_000, 1) == pytest.approx(0.4036)
    assert week2_gold_share(80_720, 200_000, 2) == pytest.approx(0.8072)
    assert week2_gold_share(0, 200_000) == 0.0
    assert week2_gold_share(500_000, 200_000) == 1.0          # 1.0 을 넘지 않는다
    assert week2_gold_share(10, 0) == 0.0                     # 0 으로 나누지 않는다


def test_one_gold_per_question_is_kept_and_the_rest_are_forbidden():
    questions = [q("a", ["p1", "p2", "p3"]), q("b", ["p4", "p5"])]
    kept, gold_ids, forbidden = select_question_gold(questions, lambda _p: True, 1)
    assert [k["gold_passage_ids"] for k in kept] == [["p1"], ["p4"]]
    assert gold_ids == ["p1", "p4"]
    # 유지된 1 권만이 아니라 원본 gold 전체가 금지된다
    assert forbidden == {"p1", "p2", "p3", "p4", "p5"}


def test_questions_with_no_gold_in_the_store_are_dropped():
    questions = [q("a", ["p1"]), q("b", ["missing"])]
    kept, gold_ids, _ = select_question_gold(questions, lambda p: p == "p1", 1)
    assert [k["qid"] for k in kept] == ["a"]
    assert gold_ids == ["p1"]


def test_the_original_question_dicts_are_not_mutated():
    questions = [q("a", ["p1", "p2"])]
    select_question_gold(questions, lambda _p: True, 1)
    assert questions[0]["gold_passage_ids"] == ["p1", "p2"]   # 금지 목록이 쓸 원본


def test_no_sampled_question_gold_leaks_into_the_filler():
    """핵심 불변식. 표본 질문의 gold 가 라벨 없이 코퍼스에 앉으면 recall 이 과소계상된다."""
    questions = [q("a", ["p1", "p2", "p3"]), q("b", ["p4", "p5", "p6"])]
    all_gold = [f"p{i}" for i in range(1, 40)]
    plan = plan_probe_corpus(
        questions, lambda _p: True, all_gold, filler_ids=["r1", "r2", "r3"],
        size=20, gold_share=0.5, gold_per_question=1,
    )
    forbidden = {"p1", "p2", "p3", "p4", "p5", "p6"}
    assert plan["question_gold"] == ["p1", "p4"]
    assert not (set(plan["other_question_gold"]) & forbidden)
    assert not (set(plan["answer_free_filler"]) & forbidden)


def test_gold_share_matches_week2_and_the_rest_is_answer_free():
    questions = [q(str(i), [f"g{i}"]) for i in range(100)]
    all_gold = [f"g{i}" for i in range(100)] + [f"o{i}" for i in range(5000)]
    plan = plan_probe_corpus(
        questions, lambda _p: True, all_gold,
        filler_ids=[f"r{i}" for i in range(5000)],
        size=1000, gold_share=0.4036, gold_per_question=1,
        shuffle=random.Random(0).shuffle,
    )
    assert plan["size"] == 1000
    assert plan["gold_share"] == pytest.approx(0.404, abs=0.001)
    assert len(plan["question_gold"]) == 100
    assert len(plan["other_question_gold"]) == 304            # 404 - 100
    assert len(plan["answer_free_filler"]) == 596


def test_a_short_filler_pool_is_reported_as_a_shortfall():
    questions = [q("a", ["p1"])]
    plan = plan_probe_corpus(
        questions, lambda _p: True, ["p1"], filler_ids=["r1"],
        size=100, gold_share=0.0, gold_per_question=1,
    )
    assert plan["size"] == 2 and plan["shortfall"] == 98      # 조용히 넘어가지 않는다


def test_zero_gold_share_gives_the_old_composition():
    questions = [q(str(i), [f"g{i}"]) for i in range(10)]
    plan = plan_probe_corpus(
        questions, lambda _p: True, [f"g{i}" for i in range(10)],
        filler_ids=[f"r{i}" for i in range(100)],
        size=50, gold_share=0.0, gold_per_question=1,
    )
    assert plan["other_question_gold"] == []
    assert len(plan["answer_free_filler"]) == 40
