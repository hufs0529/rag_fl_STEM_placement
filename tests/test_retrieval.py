import numpy as np

from src.retrieval import BruteForceRetriever, as_score_map, dense_ranked_ids


def _unit(rows):
    arr = np.asarray(rows, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_search_returns_candidates_in_descending_similarity():
    embeddings = _unit([[1, 0], [0.9, 0.1], [0, 1]])
    retriever = BruteForceRetriever(["p0", "p1", "p2"], embeddings)
    hits = retriever.search(_unit([[1, 0]]), top_n=3)[0]
    assert dense_ranked_ids(hits) == ["p0", "p1", "p2"]
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_top_n_is_clipped_to_the_corpus_size():
    retriever = BruteForceRetriever(["p0"], _unit([[1, 0]]))
    assert len(retriever.search(_unit([[1, 0]]), top_n=50)[0]) == 1


def test_score_map_round_trips_the_hits():
    hits = [("p0", 0.9), ("p1", 0.4)]
    assert as_score_map(hits) == {"p0": 0.9, "p1": 0.4}
