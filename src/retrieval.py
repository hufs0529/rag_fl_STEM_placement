"""Dense retrieval 인터페이스 + numpy 브루트포스 구현 (Week 1 gate 용).

Week 1 의 gate (ii) 는 "retrieval only" 로 돌아야 하므로, Week 2 의 Qdrant
컬렉션이 만들어지기 전에도 쓸 수 있는 소규모 브루트포스 검색기를 둔다.
Week 2 의 QdrantRetriever 는 동일한 search() 시그니처를 따르므로 gate 코드와
캐시 생성 코드가 그대로 재사용된다.

Dense retrieval interface with a numpy brute-force implementation, so Week 1's
retrieval-only gate can run before the Week 2 Qdrant collections exist. Week 2's
QdrantRetriever implements the same search() signature, so the gate and cache
code are reused unchanged.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


class BruteForceRetriever:
    """정규화된 임베딩에 대한 내적 = 코사인 유사도. 소규모 전용."""

    def __init__(self, passage_ids: Sequence[str], embeddings: np.ndarray):
        if len(passage_ids) != len(embeddings):
            raise ValueError("passage_ids and embeddings must be the same length")
        self.passage_ids = list(passage_ids)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.passage_ids)

    def search(self, query_vectors: np.ndarray, top_n: int) -> List[List[Tuple[str, float]]]:
        """질의당 (passage_id, score) 상위 top_n 을 dense 순위대로 반환."""
        q = np.asarray(query_vectors, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        scores = q @ self.embeddings.T
        top_n = min(top_n, scores.shape[1])
        # argpartition 으로 상위 top_n 만 뽑은 뒤 그 안에서만 정렬
        idx = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
        out = []
        for row, cols in enumerate(idx):
            order = cols[np.argsort(-scores[row, cols])]
            out.append([(self.passage_ids[c], float(scores[row, c])) for c in order])
        return out


def dense_ranked_ids(hits: Sequence[Tuple[str, float]]) -> List[str]:
    return [pid for pid, _ in hits]


def as_score_map(hits: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    return {pid: score for pid, score in hits}
