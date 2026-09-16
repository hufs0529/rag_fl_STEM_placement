"""사전 계산된 retrieval 캐시 (Subtask 1.2).

임베더와 재채점기가 모두 동결이고 코퍼스가 정적이므로, 후보 리스트와 재채점
점수는 한 번만 계산해 캐시한다. 그 결과 **모든 run 이 동일한 retrieval pass 를
공유**하며, 20회 본실험에서 depth 조건 간 차이가 retrieval 무작위성에서 오는
일이 원천적으로 없다.

중요: 비용 모델은 그럼에도 **배포 시점의 쿼리당 비용**을 청구한다. 캐싱은
구현 최적화로 기록될 뿐 할인이 아니다 (계획서 Subtask 1.2).

캐시는 공통 top-50 풀 하나만 저장한다. d=10 의 점수는 d=50 점수의 부분집합이라
네 조건 모두가 같은 파일에서 나온다 - 조건별 캐시를 따로 만들면 조건 간
후보 풀이 달라질 위험이 생긴다.

Precomputed retrieval cache. One common top-50 pool serves all four depths, so
conditions cannot drift apart in their candidate pool. The cost model still
charges the deployment-time per-query cost; caching is an implementation
optimisation, not a discount.
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


class RetrievalCache:
    """한 클라이언트의 질문별 (top-50 후보, dense 점수, rerank 점수)."""

    def __init__(
        self,
        qids: Sequence[str],
        pids: Sequence[str],
        candidates: "np.ndarray",
        dense_scores: "np.ndarray",
        rerank_scores: "np.ndarray",
        meta: dict = None,
    ):
        self.qids = list(qids)
        self.pids = list(pids)
        self.candidates = np.asarray(candidates, dtype=np.int32)
        self.dense_scores = np.asarray(dense_scores, dtype=np.float32)
        self.rerank_scores = np.asarray(rerank_scores, dtype=np.float32)
        self.meta = meta or {}
        self._row = {qid: i for i, qid in enumerate(self.qids)}

    def __len__(self) -> int:
        return len(self.qids)

    def __contains__(self, qid: str) -> bool:
        return qid in self._row

    @property
    def pool_size(self) -> int:
        return self.candidates.shape[1] if len(self.candidates) else 0

    def get(self, qid: str) -> Tuple[List[str], Dict[str, float]]:
        """dense 순위의 후보 id 목록과 pid -> rerank 점수 매핑."""
        row = self._row[qid]
        candidate_pids = [self.pids[i] for i in self.candidates[row]]
        scores = {pid: float(s) for pid, s in zip(candidate_pids, self.rerank_scores[row])}
        return candidate_pids, scores

    def dense_score_map(self, qid: str) -> Dict[str, float]:
        row = self._row[qid]
        return {
            self.pids[i]: float(s)
            for i, s in zip(self.candidates[row], self.dense_scores[row])
        }

    # --- 직렬화 -------------------------------------------------------------

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            qids=np.array(self.qids, dtype=object),
            pids=np.array(self.pids, dtype=object),
            candidates=self.candidates,
            dense_scores=self.dense_scores,
            rerank_scores=self.rerank_scores,
            meta=np.array([json.dumps(self.meta)], dtype=object),
        )
        return str(p)

    @classmethod
    def load(cls, path: str) -> "RetrievalCache":
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"][0])) if "meta" in data else {}
            return cls(
                qids=list(data["qids"]),
                pids=list(data["pids"]),
                candidates=data["candidates"],
                dense_scores=data["dense_scores"],
                rerank_scores=data["rerank_scores"],
                meta=meta,
            )

    @classmethod
    def build(
        cls,
        qids: Sequence[str],
        hits: Sequence[Sequence[Tuple[str, float]]],
        rerank_scores: Sequence[Dict[str, float]],
        meta: dict = None,
    ) -> "RetrievalCache":
        """retriever 출력 + 재채점 결과 -> 캐시 객체.

        후보 수가 질문마다 다르면(코퍼스가 작을 때) 짧은 쪽을 마지막 후보로
        패딩한다. 패딩된 자리는 선택에 영향을 주지 않도록 점수를 -inf 로 둔다.
        """
        pool = max((len(h) for h in hits), default=0)
        pid_index: Dict[str, int] = {}
        pids: List[str] = []

        candidates = np.zeros((len(qids), pool), dtype=np.int32)
        dense = np.full((len(qids), pool), -np.inf, dtype=np.float32)
        rerank = np.full((len(qids), pool), -np.inf, dtype=np.float32)

        for row, (hit_list, scores) in enumerate(zip(hits, rerank_scores)):
            for col in range(pool):
                pid, dense_score = hit_list[min(col, len(hit_list) - 1)]
                if pid not in pid_index:
                    pid_index[pid] = len(pids)
                    pids.append(pid)
                candidates[row, col] = pid_index[pid]
                if col < len(hit_list):
                    dense[row, col] = dense_score
                    rerank[row, col] = scores.get(pid, -np.inf)
        return cls(qids, pids, candidates, dense, rerank, meta)


def cache_path(cache_dir: str, client_id: int, partition_seed: int) -> str:
    """클라이언트 캐시 경로. **partition_seed 는 필수다.**

    두 분할(1001, 1002)은 클라이언트마다 다른 passage 를 준다. 따라서 후보 풀도
    달라서 캐시가 분할마다 따로 있어야 한다. 경로에 시드가 없던 동안 두 가지가
    동시에 깨졌다:

      1) 시드 1002 의 precompute 가 시드 1001 의 캐시를 **덮어썼다** (66 분 소실)
      2) Week 3 이 분할은 시드별로, 캐시는 시드 없이 읽어서, 클라이언트가
         **자기가 갖지도 않은 passage 의 후보 목록**을 받았다 - 오류 없이

    기본값을 주지 않는 이유가 2) 다. 기본값이 있으면 호출부가 시드를 잊어도
    조용히 돌아간다. 필수로 두면 잊은 곳이 즉시 드러난다.
    """
    return str(
        Path(cache_dir) / "retrieval" / f"seed_{partition_seed}" / f"client_{client_id:02d}.npz"
    )


def load_all(cache_dir: str, num_clients: int, partition_seed: int) -> Dict[int, RetrievalCache]:
    """분할 시드에 맞는 캐시만 읽는다. 시드를 섞으면 클라이언트가 남의 후보를 받는다."""
    return {
        cid: RetrievalCache.load(cache_path(cache_dir, cid, partition_seed))
        for cid in range(num_clients)
    }
