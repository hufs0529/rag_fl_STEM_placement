import pytest

from src.selection import promotion_rate, rank_change_rate, select_passages

IDS = [f"p{i}" for i in range(50)]
# dense 순위를 그대로 따르는 점수 + p40 만 크게 높은 점수
SCORES = {pid: -i for i, pid in enumerate(IDS)}
SCORES["p40"] = 100.0


def test_depth_zero_is_plain_dense_retrieval():
    assert select_passages(IDS, SCORES, depth=0) == ["p0", "p1", "p2"]


def test_depth_three_control_cannot_promote_a_new_passage():
    selected = select_passages(IDS, SCORES, depth=3)
    assert set(selected) == {"p0", "p1", "p2"}
    assert promotion_rate(IDS, selected) == 0.0


def test_depth_three_control_can_still_reorder():
    scores = dict(SCORES)
    scores["p2"] = 50.0
    assert select_passages(IDS, scores, depth=3) == ["p2", "p0", "p1"]


def test_deeper_reranking_promotes_passages_dense_ranked_low():
    selected = select_passages(IDS, SCORES, depth=50)
    assert selected[0] == "p40"
    assert promotion_rate(IDS, selected) == pytest.approx(1 / 3)


def test_every_depth_returns_exactly_top_k():
    for depth in (0, 3, 10, 50):
        assert len(select_passages(IDS, SCORES, depth)) == 3


def test_ties_break_on_dense_rank_so_selection_is_deterministic():
    flat = {pid: 1.0 for pid in IDS}
    assert select_passages(IDS, flat, depth=50) == ["p0", "p1", "p2"]


def test_missing_rerank_scores_raise_rather_than_silently_degrade():
    partial = {pid: 0.0 for pid in IDS[:5]}
    with pytest.raises(KeyError):
        select_passages(IDS, partial, depth=10)


def test_rank_change_rate_is_zero_for_the_baseline():
    assert rank_change_rate(IDS, select_passages(IDS, SCORES, depth=0)) == 0.0
