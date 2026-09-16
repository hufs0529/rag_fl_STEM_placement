"""cross-encoder 재채점기 + FLOPs 회계 (Subtask 1.2 / 1.4).

모델은 distillation 규모의 6-layer MiniLM 이다. 이는 의도된 설계로, 측정된
이득이 "제약된 하드웨어에서도 돌릴 만한 부품"에 귀속되게 하기 위함이다.

비용 회계상 중요한 점: 실험에서는 점수를 한 번만 계산해 캐시하지만
(모든 run 이 동일한 retrieval pass 를 공유), 비용 모델은 **배포 시점의
쿼리당 비용**을 청구한다. 캐싱은 구현 최적화로 기록될 뿐 할인이 아니다.

A distilled 6-layer MiniLM cross-encoder, chosen so any measured gain is
attributable to a component small enough to be plausible on constrained
hardware. Scores are computed once and cached because both retrieval components
are frozen and the corpora static, but the cost model nevertheless charges the
deployment-time per-query cost; caching is recorded as an implementation
optimisation, not a discount.
"""

from typing import Dict, List, Sequence

import numpy as np


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 64,
        device: str = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.forward_passes = 0      # 비용 회계용 카운터 / cost-accounting counter

    def score(self, query: str, passages: Sequence[dict]) -> Dict[str, float]:
        """(query, passage) 쌍 점수. 반환은 passage_id -> score."""
        if not passages:
            return {}
        pairs = [[query, _pair_text(p)] for p in passages]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        self.forward_passes += len(pairs)
        return {p["pid"]: float(s) for p, s in zip(passages, np.atleast_1d(scores))}

    def score_batch(
        self, queries: Sequence[str], passage_lists: Sequence[Sequence[dict]]
    ) -> List[Dict[str, float]]:
        """여러 질의를 한 번에 - 후보 풀 전체(top-50)를 한 번 채점해 캐시에 넣는다.

        depth 조건별로 따로 채점하지 않는 이유: d=10 의 점수는 d=50 점수의
        부분집합이므로, 공통 풀을 한 번 채점하면 네 조건 모두를 만족한다.
        """
        flat_pairs, owners = [], []
        for i, (q, plist) in enumerate(zip(queries, passage_lists)):
            for p in plist:
                flat_pairs.append([q, _pair_text(p)])
                owners.append((i, p["pid"]))
        if not flat_pairs:
            return [{} for _ in queries]

        scores = self.model.predict(
            flat_pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        self.forward_passes += len(flat_pairs)

        out: List[Dict[str, float]] = [{} for _ in queries]
        for (i, pid), s in zip(owners, np.atleast_1d(scores)):
            out[i][pid] = float(s)
        return out


def _pair_text(passage: dict) -> str:
    title = (passage.get("title") or "").strip()
    return f"{title}. {passage['text'].strip()}" if title else passage["text"].strip()


def forward_passes_for_depth(n_queries: int, depth: int) -> int:
    """depth d 에서의 재채점 forward pass 수 = n_queries x d.

    d=0 은 0. 이 값이 Subtask 1.4 에서 "절약된 바이트"와 맞바꿔지는 반복 비용의
    단위량이다.
    """
    return 0 if depth <= 0 else n_queries * depth
