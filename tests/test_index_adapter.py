"""Qdrant 어댑터가 BruteForceRetriever 와 같은 결과를 내는지.

qdrant-client 1.12 가 search()/search_batch() 를 폐기하고 1.19 가 **제거**했다.
게이트는 BruteForce 로 측정하고 Week 2 는 Qdrant 로 측정하므로, 두 경로가 다른
결과를 내면 게이트의 결론이 본실험으로 이어지지 않는다. 그래서 동등성을 고정한다.
"""
import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from src.index import QdrantIndex, QdrantRetriever, build_client_index
from src.retrieval import BruteForceRetriever


def _unit(rng, *shape):
    v = rng.normal(size=shape).astype(np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture
def fixtures(tmp_path):
    rng = np.random.default_rng(0)
    n, dim = 300, 32
    vectors = _unit(rng, n, dim)
    pids = [f"p{i}" for i in range(n)]
    index = QdrantIndex(path=str(tmp_path / "q"), dim=dim, distance="cosine")
    name = build_client_index(index, 0, pids, vectors, log=lambda *_a: None)
    yield index, name, pids, vectors, rng, dim
    index.close()


def test_qdrant_matches_brute_force_order_and_scores(fixtures):
    index, name, pids, vectors, rng, dim = fixtures
    queries = _unit(rng, 9, dim)
    got = QdrantRetriever(index, name).search(queries, 10)
    want = BruteForceRetriever(pids, vectors).search(queries, 10)
    assert len(got) == len(want) == 9
    for g, w in zip(got, want):
        assert [x[0] for x in g] == [x[0] for x in w]
        for (_, gs), (_, ws) in zip(g, w):
            assert gs == pytest.approx(ws, abs=1e-5)


def test_a_single_query_vector_is_accepted(fixtures):
    """BruteForceRetriever 와 같이 1차원 입력도 받아야 한다."""
    index, name, _pids, _v, rng, dim = fixtures
    one = _unit(rng, dim)
    out = QdrantRetriever(index, name).search(one, 5)
    assert len(out) == 1 and len(out[0]) == 5


def test_batching_covers_more_queries_than_one_chunk(fixtures):
    """내부 청크(256)보다 많은 질의를 넣어도 전부 돌아와야 한다."""
    index, name, pids, vectors, rng, dim = fixtures
    queries = _unit(rng, 300, dim)
    got = QdrantRetriever(index, name).search(queries, 3)
    want = BruteForceRetriever(pids, vectors).search(queries, 3)
    assert len(got) == 300
    assert [x[0] for x in got[299]] == [x[0] for x in want[299]]
