from src.schedule import eval_rounds, is_eval_round


def test_early_rounds_are_dense_and_late_rounds_sparse():
    rounds = eval_rounds(60, dense_until=10, sparse_every=5)
    assert rounds[:10] == list(range(1, 11))
    assert rounds[10:] == [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def test_the_final_round_is_always_evaluated():
    # 고정 예산의 끝점이 곡선에 없으면 "그 예산이 무엇을 사는가"를 못 읽는다
    assert eval_rounds(12, 10, 5)[-1] == 12
    assert eval_rounds(61, 10, 5)[-1] == 61


def test_short_runs_are_evaluated_every_round():
    assert eval_rounds(6, 10, 5) == [1, 2, 3, 4, 5, 6]


def test_schedule_is_identical_for_every_condition():
    # 조건마다 평가 지점이 다르면 Week 5 의 보간이 비교 가능성을 잃는다
    assert eval_rounds(40, 10, 5) == eval_rounds(40, 10, 5)


def test_zero_rounds_gives_an_empty_schedule():
    assert eval_rounds(0) == []


def test_is_eval_round_matches_the_schedule():
    assert is_eval_round(15, 60, 10, 5)
    assert not is_eval_round(14, 60, 10, 5)
