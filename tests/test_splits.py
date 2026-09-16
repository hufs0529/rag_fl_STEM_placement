"""train / val / test 가 서로 겹치지 않는지.

실측 사고: 게이트 표본(val)과 평가셋(test)을 같은 80,720 개에서 독립적으로 뽑아
24 개가 겹쳤다(기대 24.8). 결과를 내는 모델이 test 를 학습하지는 않지만, 설정을
정한 질문과 최종 비교에 쓰는 질문이 겹치면 "세 집합이 분리됐다" 고 말할 수 없다.
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.nq_data import build_eval_set


def _questions(n):
    return [{"qid": f"q{i}", "question": f"question {i}", "answers": ["a"],
             "gold_passage_ids": [f"p{i}"]} for i in range(n)]


def test_probe_set_reproduces_the_gate_sampling():
    from make_splits import probe_qids
    qs = _questions(5000)
    # 게이트와 같은 호출: random.Random(seed).sample(questions, n)
    expected = set()
    for n in (100, 300, 2000):
        expected.update(q["qid"] for q in random.Random(42).sample(qs, n))
    assert probe_qids(qs, seed=42) == expected


def test_probe_set_unions_rather_than_assuming_nesting():
    """합집합을 쓰는 이유: 중첩이 **보장되지 않는다.**

    CPython 의 random.sample 은 k 와 모집단 크기의 비에 따라 알고리즘을 바꾼다.
    실측:
      모집단 80,720 (실제)  sample(100) ⊂ sample(300) ⊂ sample(2000)  -> 합집합 2,000
      모집단  5,000         sample(300) 은 sample(2000) 의 접두사가 아니다 -> 합집합 2,009
    큰 표본 하나만 제외하면 두 번째 경우에서 9 개를 놓친다. 합집합은 어느 쪽이든 맞다.
    """
    from make_splits import probe_qids
    for pop in (5000, 80720):
        qs = _questions(pop)
        parts = [{q["qid"] for q in random.Random(42).sample(qs, n)}
                 for n in (100, 300, 2000)]
        union = probe_qids(qs, 42)
        for part in parts:
            assert part <= union                  # 어떤 n 을 썼든 전부 제외된다
        assert union == set().union(*parts)

    # 중첩이 깨지는 경우가 실제로 있다 - 합집합이 과한 조치가 아니라는 확인
    qs = _questions(5000)
    s300 = {q["qid"] for q in random.Random(42).sample(qs, 300)}
    s2000 = {q["qid"] for q in random.Random(42).sample(qs, 2000)}
    assert not s300 <= s2000


def test_test_set_excludes_the_val_set():
    from make_splits import probe_qids
    qs = _questions(5000)
    val = probe_qids(qs, seed=42)
    test = build_eval_set(qs, size=200, seed=42, exclude=val)
    test_ids = {q["qid"] for q in test}
    assert len(test_ids) == 200
    assert not (test_ids & val)


def test_without_exclusion_they_do_overlap():
    """제외하지 않으면 실제로 겹친다 - 수정이 의미 있다는 확인."""
    from make_splits import probe_qids
    qs = _questions(5000)
    val = probe_qids(qs, seed=42)
    naive = {q["qid"] for q in build_eval_set(qs, size=200, seed=42)}
    assert naive & val                            # 겹친다


def test_three_sets_partition_the_pool():
    """train = pool − test, 그리고 test ∩ val = 0."""
    from make_splits import probe_qids
    qs = _questions(3000)
    val = probe_qids(qs, seed=42)
    test = {q["qid"] for q in build_eval_set(qs, size=100, seed=42, exclude=val)}
    pool = {q["qid"] for q in qs}
    train = pool - test
    assert train | test == pool
    assert not (train & test)
    assert not (test & val)
    # val 은 train 안에 있다 - 게이트는 학습 질문으로 설정을 정했다
    assert val <= train


def test_the_test_set_is_stable_across_run_seeds():
    """test 는 run seed 와 무관해야 한다 - 조건 간 평가셋 분산을 없앤 장치."""
    qs = _questions(2000)
    a = [q["qid"] for q in build_eval_set(qs, size=50, seed=42)]
    b = [q["qid"] for q in build_eval_set(qs, size=50, seed=42)]
    assert a == b
