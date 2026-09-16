import numpy as np

from src.retrieval_cache import RetrievalCache
from src.selection import select_passages


def _cache(pool=50, n=3):
    hits = [
        [(f"q{q}_p{i}", 1.0 - 0.01 * i) for i in range(pool)]
        for q in range(n)
    ]
    scores = [{f"q{q}_p{i}": float(pool - i) for i in range(pool)} for q in range(n)]
    scores[0][f"q0_p40"] = 999.0
    return RetrievalCache.build([f"q{i}" for i in range(n)], hits, scores, {"pool": pool})


def test_cache_returns_candidates_in_dense_order():
    cache = _cache()
    candidates, _ = cache.get("q0")
    assert candidates[:3] == ["q0_p0", "q0_p1", "q0_p2"]
    assert len(candidates) == 50


def test_one_common_pool_serves_every_depth():
    # d=10 의 점수는 d=50 점수의 부분집합이어야 한다 - 조건별 캐시를 따로 만들면
    # 후보 풀이 갈라질 위험이 생긴다
    cache = _cache()
    candidates, scores = cache.get("q0")
    assert select_passages(candidates, scores, 10)[0] == "q0_p0"
    assert select_passages(candidates, scores, 50)[0] == "q0_p40"


def test_cache_round_trips_through_disk(tmp_path):
    cache = _cache()
    path = cache.save(tmp_path / "client_00.npz")
    reloaded = RetrievalCache.load(path)
    assert len(reloaded) == len(cache)
    assert reloaded.pool_size == 50
    assert reloaded.meta["pool"] == 50
    assert reloaded.get("q1") == cache.get("q1")


def test_missing_questions_are_reported_not_guessed():
    cache = _cache()
    assert "q0" in cache
    assert "nope" not in cache


def test_short_candidate_lists_are_padded_without_affecting_selection():
    hits = [[("a", 0.9), ("b", 0.5)], [("c", 0.8), ("d", 0.7), ("e", 0.6)]]
    scores = [{"a": 1.0, "b": 2.0}, {"c": 1.0, "d": 3.0, "e": 2.0}]
    cache = RetrievalCache.build(["q0", "q1"], hits, scores)
    candidates, score_map = cache.get("q0")
    assert candidates[:2] == ["a", "b"]
    # 패딩된 자리는 -inf 점수라 절대 선택되지 않는다
    assert score_map[candidates[2]] == float("-inf") or candidates[2] == "b"


def test_dense_scores_are_kept_alongside_rerank_scores():
    cache = _cache()
    dense = cache.dense_score_map("q0")
    assert dense["q0_p0"] > dense["q0_p1"]


# --- 후보 풀이 만드는 상한 (Subtask 1.2 진단) -------------------------------

from src.metrics import gold_recall_at_k


def test_reranking_cannot_find_gold_that_is_not_in_the_pool():
    """depth 는 후보 풀 **안에서만** 고른다 - 풀 밖의 정답지는 d=50 도 못 찾는다.

    그래서 "풀 안에 정답지가 있는 질문의 비율"이 모든 depth 의 상한이고,
    그 값을 모르면 d=50 의 recall 이 잘한 건지 못한 건지 판단할 수 없다.
    """
    pool = ["p0", "p1", "p2"]                 # 후보 풀 (정답지 없음)
    gold = ["g9"]                             # 정답지는 풀 밖에 있다
    scores = {"p0": 1.0, "p1": 2.0, "p2": 3.0}

    assert gold_recall_at_k(select_passages(pool, scores, len(pool)), gold, 3) == 0.0
    # 풀 전체를 봐도 0 - 상한 자체가 0 이다
    assert gold_recall_at_k(pool, gold, len(pool)) == 0.0


def test_depth_recall_can_never_exceed_the_pool_ceiling():
    pool = [f"p{i}" for i in range(10)]
    gold = ["p9"]                             # 풀의 맨 뒤에 있다
    scores = {p: float(i) for i, p in enumerate(pool)}      # p9 가 최고점

    ceiling = gold_recall_at_k(pool, gold, len(pool))
    assert ceiling == 1.0
    for depth in (0, 3, 10):
        assert gold_recall_at_k(select_passages(pool, scores, depth), gold, 3) <= ceiling


def test_a_pool_that_contains_gold_sets_a_reachable_ceiling():
    # 깊게 읽으면 풀 안의 정답지를 끌어올릴 수 있다
    pool = [f"p{i}" for i in range(10)]
    gold = ["p9"]
    scores = {p: (99.0 if p == "p9" else 0.0) for p in pool}
    assert gold_recall_at_k(select_passages(pool, scores, 0), gold, 3) == 0.0
    assert gold_recall_at_k(select_passages(pool, scores, 10), gold, 3) == 1.0


def test_recall_is_not_fooled_by_a_random_passage_that_happens_to_contain_the_answer():
    """코퍼스의 "아무 책"도 거의 전부 어떤 질문의 정답 문자열을 담고 있다.

    id 로 대조하므로 그런 passage 를 가져와도 성공으로 세지 않는다.
    """
    assert gold_recall_at_k(["random_pid"], ["gold_pid"], 3) == 0.0


# --- 캐시 경로는 분할 시드를 포함해야 한다 ---------------------------------

def test_cache_path_separates_partition_seeds():
    """실측 사고: 경로에 시드가 없어 시드 1002 의 precompute 가 1001 의 캐시를
    덮어썼고(66분 소실), Week 3 은 분할은 시드별로 캐시는 시드 없이 읽어
    클라이언트가 자기가 갖지도 않은 passage 의 후보를 받았다.
    """
    from src.retrieval_cache import cache_path
    a = cache_path("results/cache", 0, 1001)
    b = cache_path("results/cache", 0, 1002)
    assert a != b
    assert "seed_1001" in a and "seed_1002" in b
    assert a.endswith("client_00.npz") and b.endswith("client_00.npz")


def test_cache_path_requires_the_seed():
    """기본값을 주면 호출부가 시드를 잊어도 조용히 돌아간다 - 필수로 둔다."""
    import pytest
    from src.retrieval_cache import cache_path
    with pytest.raises(TypeError):
        cache_path("results/cache", 0)


def test_load_all_requires_the_seed():
    import inspect
    from src.retrieval_cache import load_all
    params = inspect.signature(load_all).parameters
    assert "partition_seed" in params
    assert params["partition_seed"].default is inspect.Parameter.empty
