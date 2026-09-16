"""주제 클러스터 기반 클라이언트 파티션 (Subtask 1.1).

질문과 코퍼스를 **따로가 아니라 함께** 나눈다:
  1) passage 를 주제별로 클러스터링
  2) 클러스터를 8 클라이언트에 균등 배분
  3) 각 질문은 자신의 gold passage 를 가진 클라이언트에 배정

결과적으로 클라이언트는 서로 겹치지 않는 주제 범위를 갖게 되어 federated
learning 이 다루는 data-silo 구조를 재현하면서도, 모든 클라이언트가 원리적으로
자기 gold evidence 를 검색할 수 있다. 따라서 측정된 retrieval 실패는
파티션 운(luck)이 아니라 **순위 매기기의 난이도**를 반영한다.

왜 Dirichlet 이 아닌가: Dirichlet 파티션은 두 가지를 동시에 바꾼다 - 주제 범위를
좁히는 것(의도)과 클라이언트 데이터 크기를 불균등하게 만드는 것(의도치 않은
평균화 불안정성의 원천). 게다가 시드마다 다시 뽑히므로, 시드가 안정시키려는
추정치에 파티션 분산을 주입한다. 크기 균등·주제 분리 파티션을 고정하면 설계가
1차원으로 유지된다. 시드 5개에 파티션 2개를 배정해 결과가 한 배정에 묶이지
않게 한다.

Topic-cluster client partitioning. Questions and corpus are partitioned jointly
rather than independently, so clients specialise in disjoint topic ranges while
every client can in principle retrieve its own gold evidence. Client
heterogeneity is a fixed background condition, not an experimental factor.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np


def cluster_passages(
    embeddings: "np.ndarray", n_clusters: int = 256, batch_size: int = 4096, seed: int = 0
) -> "np.ndarray":
    """passage 임베딩을 주제 클러스터로. 200k x 384 라 MiniBatchKMeans 를 쓴다."""
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(n_clusters, len(embeddings))
    model = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, random_state=seed, n_init=3
    )
    return model.fit_predict(np.asarray(embeddings, dtype=np.float32))


def allocate_clusters(
    cluster_weights: Dict[int, int], num_clients: int
) -> Dict[int, int]:
    """클러스터 -> 클라이언트 배정. 큰 클러스터부터 가장 가벼운 클라이언트에 넣는다.

    (LPT 그리디) 클러스터 단위로 통째로 배정하므로 주제 분리가 유지되고,
    부하 균형 덕분에 클라이언트 크기가 거의 균등해진다. 완전 균등화는
    balance_clients() 가 마무리한다.
    """
    loads = [0] * num_clients
    assignment: Dict[int, int] = {}
    for cluster in sorted(cluster_weights, key=lambda c: -cluster_weights[c]):
        target = min(range(num_clients), key=lambda i: (loads[i], i))
        assignment[cluster] = target
        loads[target] += cluster_weights[cluster]
    return assignment


def assign_questions(
    questions: Sequence[dict],
    passage_to_client: Dict[str, int],
) -> Dict[int, List[dict]]:
    """각 질문을 gold 를 가장 많이 보유한 클라이언트에 배정.

    gold 가 여러 클라이언트에 흩어진 질문은, 배정된 클라이언트가 가진 gold 만
    남긴다. 자기 인덱스에서 검색할 수 없는 gold 를 recall 분모에 남겨두면
    "검색 난이도"가 아니라 "파티션 운"을 재게 되기 때문이다.

    Questions are assigned to the client holding most of their gold passages,
    and their gold set is restricted to that client's passages: a gold the
    client cannot retrieve would measure partition luck, not ranking difficulty.
    """
    clients: Dict[int, List[dict]] = defaultdict(list)
    for q in questions:
        owners = Counter(
            passage_to_client[pid] for pid in q["gold_passage_ids"] if pid in passage_to_client
        )
        if not owners:
            continue                                  # 코퍼스에 gold 가 없으면 버린다
        client = owners.most_common(1)[0][0]
        local_gold = [
            pid for pid in q["gold_passage_ids"] if passage_to_client.get(pid) == client
        ]
        clients[client].append({**q, "gold_passage_ids": local_gold, "client_id": client})
    return dict(clients)


def balance_clients(
    clients: Dict[int, List[dict]], num_clients: int, seed: int = 0
) -> Dict[int, List[dict]]:
    """질문 수를 클라이언트 간 정확히 동일하게 맞춘다(가장 작은 쪽으로 절삭).

    크기 균등은 설계상의 요구다: 크기가 다르면 FedAvg 평균화가 불안정해지고,
    그 불안정성이 처리 효과와 섞인다. 절삭은 무작위로 하되 시드를 고정한다.
    """
    rng = np.random.default_rng(seed)
    sizes = [len(clients.get(i, [])) for i in range(num_clients)]
    target = min(sizes) if sizes else 0
    out = {}
    for i in range(num_clients):
        group = list(clients.get(i, []))
        if len(group) > target:
            keep = rng.choice(len(group), size=target, replace=False)
            group = [group[j] for j in sorted(keep)]
        out[i] = group
    return out


def partition(
    passages: Sequence[dict],
    embeddings: "np.ndarray",
    questions: Sequence[dict],
    num_clients: int = 8,
    n_clusters: int = 256,
    batch_size: int = 4096,
    seed: int = 0,
    equal_size: bool = True,
    log=print,
) -> dict:
    """전체 파티션 파이프라인. 시드는 partition_seeds 중 하나(1001/1002)."""
    labels = cluster_passages(embeddings, n_clusters, batch_size, seed)
    log(f"  clustered {len(passages):,} passages into {len(set(labels))} topics")

    # 클러스터 가중치는 passage 수가 아니라 **질문 수**로 잡는다. 균등하게 맞춰야
    # 하는 것은 학습 데이터 크기이지 인덱스 크기가 아니기 때문이다.
    pid_to_cluster = {p["pid"]: int(c) for p, c in zip(passages, labels)}
    question_weight = Counter()
    for q in questions:
        for pid in q["gold_passage_ids"]:
            if pid in pid_to_cluster:
                question_weight[pid_to_cluster[pid]] += 1
                break
    for cluster in set(pid_to_cluster.values()):
        question_weight.setdefault(cluster, 0)

    cluster_to_client = allocate_clusters(dict(question_weight), num_clients)
    passage_to_client = {pid: cluster_to_client[c] for pid, c in pid_to_cluster.items()}

    clients = assign_questions(questions, passage_to_client)
    if equal_size:
        clients = balance_clients(clients, num_clients, seed)

    client_passages: Dict[int, List[str]] = defaultdict(list)
    for pid, client in passage_to_client.items():
        client_passages[client].append(pid)

    sizes = {i: len(clients.get(i, [])) for i in range(num_clients)}
    log("  questions per client: " + ", ".join(f"{i}:{n}" for i, n in sizes.items()))
    log("  passages  per client: "
        + ", ".join(f"{i}:{len(client_passages.get(i, []))}" for i in range(num_clients)))

    return {
        "seed": seed,
        "num_clients": num_clients,
        "n_clusters": int(len(set(labels))),
        "questions_per_client": sizes,
        "client_questions": {str(i): clients.get(i, []) for i in range(num_clients)},
        "client_passages": {str(i): client_passages.get(i, []) for i in range(num_clients)},
        "passage_to_client": passage_to_client,
    }


def partition_summary(result: dict) -> dict:
    """보고서용 요약 - 주제 분리와 크기 균등이 실제로 지켜졌는지."""
    sizes = list(result["questions_per_client"].values())
    passage_counts = [len(v) for v in result["client_passages"].values()]
    gold_reachable = 0
    total = 0
    for client, questions in result["client_questions"].items():
        owned = set(result["client_passages"][client])
        for q in questions:
            total += 1
            gold_reachable += any(pid in owned for pid in q["gold_passage_ids"])
    return {
        "questions_per_client": sizes,
        "question_size_spread": (max(sizes) - min(sizes)) if sizes else 0,
        "passages_per_client": passage_counts,
        "gold_reachable_fraction": gold_reachable / total if total else 0.0,
        "clients_are_topic_disjoint": True,   # 클러스터를 통째로 배정하므로 구조적으로 보장
    }
