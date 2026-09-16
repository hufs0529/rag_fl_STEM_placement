"""처리 변수 구현: reranking depth d 에 따른 passage 선택.

계획서의 핵심 한 줄. 모든 조건은 동일한 top-50 후보 풀에서 출발하고,
프롬프트에는 항상 정확히 3개의 passage 가 들어간다. 조건 간 차이는
"그 3개를 고르는 데 얼마나 계산을 썼는가" 뿐이다.

  d = 0  : dense 상위 3개 그대로 (reranking 없음)
  d = 3  : dense 가 이미 고른 3개를 cross-encoder 점수로 재정렬만 함 (control)
  d = 10 : 상위 10개를 재채점해 상위 3개 선택
  d = 50 : 후보 풀 전체를 재채점해 상위 3개 선택

The treatment axis. Every condition draws from one common top-50 pool and puts
exactly three passages in the prompt; conditions differ only in how much compute
was spent selecting them. d=3 is the control - it reorders what dense already
selected without promoting new passages, separating an ordering effect from a
genuine retrieval-quality effect.
"""

from typing import Dict, List, Sequence


def select_passages(
    dense_ranked_ids: Sequence[str],
    rerank_scores: Dict[str, float],
    depth: int,
    top_k: int = 3,
) -> List[str]:
    """depth d 로 재순위를 매긴 뒤 상위 top_k passage id 를 반환.

    Args:
        dense_ranked_ids: dense retrieval 이 매긴 순서의 후보 id (top-50 pool).
        rerank_scores:    cross-encoder 점수 (사전 계산·캐시됨, Subtask 1.2).
        depth:            재채점할 후보 수. 0 이면 reranking 없음.
        top_k:            프롬프트에 넣을 passage 수 (전 조건 3 으로 고정).

    동점 처리: dense 순위를 tie-break 으로 써서 결정적(deterministic)으로 만든다.
    Ties are broken by dense rank so selection is deterministic across runs.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if depth == 0:
        return list(dense_ranked_ids[:top_k])

    head = list(dense_ranked_ids[:depth])
    dense_rank = {pid: i for i, pid in enumerate(head)}
    missing = [pid for pid in head if pid not in rerank_scores]
    if missing:
        raise KeyError(f"rerank scores missing for {len(missing)} candidates, e.g. {missing[:3]}")

    reordered = sorted(head, key=lambda pid: (-rerank_scores[pid], dense_rank[pid]))
    return reordered[:top_k]


def rank_change_rate(dense_ranked_ids: Sequence[str], selected: Sequence[str]) -> float:
    """treatment 가 실제로 걸렸는지 확인하는 진단 지표 (Subtask 1.3).

    dense 상위 top_k 와 비교해 실제로 자리가 바뀐 비율. d=0 이면 0.0 이어야 하고,
    d=3 control 은 "구성은 같고 순서만 바뀐" 경우를 잡아낸다.

    Fraction of selected slots whose passage differs from the dense top-k at the
    same position. Confirms the treatment took effect; 0.0 by construction at d=0.
    """
    k = len(selected)
    if k == 0:
        return 0.0
    baseline = list(dense_ranked_ids[:k])
    changed = sum(1 for i in range(k) if i >= len(baseline) or baseline[i] != selected[i])
    return changed / k


def promotion_rate(dense_ranked_ids: Sequence[str], selected: Sequence[str]) -> float:
    """dense top-k 에는 없던 passage 가 승격된 비율.

    d=3 control 에서는 정의상 0.0 - 재정렬만 하고 새 passage 를 올리지 않는다.
    이 지표가 ordering effect 와 retrieval-quality effect 를 갈라준다.

    Fraction of selected passages that were NOT in the dense top-k. Zero by
    construction at d=3, which is exactly what separates the ordering effect
    from the retrieval-quality effect.
    """
    k = len(selected)
    if k == 0:
        return 0.0
    baseline = set(dense_ranked_ids[:k])
    return sum(1 for pid in selected if pid not in baseline) / k
