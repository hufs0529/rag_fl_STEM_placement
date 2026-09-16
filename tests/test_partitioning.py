import numpy as np

from src.partitioning import (
    allocate_clusters, assign_questions, balance_clients, partition, partition_summary,
)


def test_clusters_are_allocated_to_the_lightest_client():
    assignment = allocate_clusters({0: 10, 1: 8, 2: 6, 3: 6}, num_clients=2)
    loads = [0, 0]
    for cluster, client in assignment.items():
        loads[client] += {0: 10, 1: 8, 2: 6, 3: 6}[cluster]
    assert abs(loads[0] - loads[1]) <= 4


def test_a_cluster_is_never_split_across_clients():
    # 주제 분리는 클러스터를 통째로 배정하기 때문에 구조적으로 보장된다
    assignment = allocate_clusters({c: 1 for c in range(10)}, num_clients=3)
    assert len(assignment) == 10
    assert all(isinstance(v, int) for v in assignment.values())


def test_question_goes_to_the_client_holding_most_of_its_gold():
    questions = [{"qid": "q", "question": "?", "answers": ["a"],
                  "gold_passage_ids": ["p0", "p1", "p2"]}]
    clients = assign_questions(questions, {"p0": 0, "p1": 1, "p2": 1})
    assert list(clients) == [1]
    # 배정된 클라이언트가 검색할 수 없는 gold 는 제거된다
    assert clients[1][0]["gold_passage_ids"] == ["p1", "p2"]


def test_question_with_no_gold_in_the_corpus_is_dropped():
    questions = [{"qid": "q", "question": "?", "answers": ["a"], "gold_passage_ids": ["zz"]}]
    assert assign_questions(questions, {"p0": 0}) == {}


def test_balancing_makes_every_client_exactly_equal():
    clients = {0: [{"qid": str(i)} for i in range(10)], 1: [{"qid": "x"}] * 4}
    balanced = balance_clients(clients, num_clients=2, seed=0)
    assert len(balanced[0]) == len(balanced[1]) == 4


def test_end_to_end_partition_is_equal_size_and_gold_reachable():
    rng = np.random.default_rng(0)
    embeddings = np.concatenate(
        [rng.normal(loc=i * 10, size=(40, 4)) for i in range(4)]
    ).astype(np.float32)
    passages = [{"pid": f"p{i}", "title": "t", "text": "body"} for i in range(160)]
    questions = [
        {"qid": f"q{i}", "question": "?", "answers": ["a"], "gold_passage_ids": [f"p{i}"]}
        for i in range(160)
    ]
    result = partition(passages, embeddings, questions, num_clients=4,
                       n_clusters=8, seed=1001, log=lambda *a: None)
    summary = partition_summary(result)
    assert summary["question_size_spread"] == 0
    # 모든 질문이 자기 클라이언트에서 자기 gold 를 검색할 수 있어야 한다:
    # 그래야 측정된 retrieval 실패가 파티션 운이 아니라 순위 난이도를 반영한다
    assert summary["gold_reachable_fraction"] == 1.0


def test_two_partition_seeds_give_different_allocations():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(120, 6)).astype(np.float32)
    passages = [{"pid": f"p{i}", "title": "t", "text": "body"} for i in range(120)]
    questions = [
        {"qid": f"q{i}", "question": "?", "answers": ["a"], "gold_passage_ids": [f"p{i}"]}
        for i in range(120)
    ]
    kwargs = dict(num_clients=4, n_clusters=12, log=lambda *a: None)
    a = partition(passages, embeddings, questions, seed=1001, **kwargs)
    b = partition(passages, embeddings, questions, seed=1002, **kwargs)
    assert a["passage_to_client"] != b["passage_to_client"]
