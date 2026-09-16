"""클라이언트 업데이트 발산(client-update divergence) 측정 (Subtask 1.2 (iii), 1.3).

왜 필요한가: 이 실험의 가설은 "retrieval 품질이 좋아지면 클라이언트들이 같은
과제를 배우게 되어 업데이트가 덜 발산하고, 따라서 라운드 수가 준다"이다.
따라서 (a) drift 가 애초에 관측 가능한 K 를 골라야 하고(gate 3),
(b) 본실험에서 depth 가 recall 을 올릴 때 실제로 drift 가 줄었는지 확인해야
한다(Subtask 1.4 의 mechanism check).

Client-update divergence. The hypothesis is that better retrieval makes clients
learn the same task, so their updates diverge less and fewer rounds are needed.
Hence gate 3 must pick a K where drift is observable at all, and the main runs
must check whether depth-driven recall gains coincide with reduced divergence.
"""

from typing import Dict, List, Sequence

import numpy as np


def flatten_update(update: Dict[str, "np.ndarray"], keys: Sequence[str] = None) -> "np.ndarray":
    """state dict -> 1차원 벡터. 키 순서를 고정해 클라이언트 간 비교 가능하게 한다."""
    keys = list(keys) if keys is not None else sorted(update.keys())
    return np.concatenate([np.asarray(update[k], dtype=np.float64).ravel() for k in keys])


def client_deltas(
    global_params: Dict[str, "np.ndarray"],
    client_params: Sequence[Dict[str, "np.ndarray"]],
) -> List["np.ndarray"]:
    """각 클라이언트의 이번 라운드 업데이트 delta_i = theta_i - theta_global."""
    keys = sorted(global_params.keys())
    base = flatten_update(global_params, keys)
    return [flatten_update(p, keys) - base for p in client_params]


def divergence(deltas: Sequence["np.ndarray"], eps: float = 1e-12) -> float:
    """평균 업데이트 대비 정규화된 발산도.

        div = mean_i || d_i - d_bar ||  /  (|| d_bar || + eps)

    0 이면 모든 클라이언트가 같은 방향·크기로 움직인 것(발산 없음).
    1 근처면 개별 업데이트의 흩어짐이 평균 업데이트만큼 크다는 뜻.
    평균 업데이트 크기로 나누므로 학습률과 K 에 대해 대체로 스케일 불변이라
    K 스윕 간 비교가 가능하다.

    Divergence normalised by the mean update, so it is largely scale-invariant
    in learning rate and K and therefore comparable across the gate-3 sweep.
    """
    if len(deltas) < 2:
        return 0.0
    stacked = np.stack(deltas)
    mean = stacked.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(stacked - mean, axis=1)))
    return spread / (float(np.linalg.norm(mean)) + eps)


def mean_pairwise_cosine(deltas: Sequence["np.ndarray"], eps: float = 1e-12) -> float:
    """업데이트 방향의 평균 쌍별 코사인 유사도 (1 = 완전 정렬, 0 = 직교).

    divergence() 가 크기+방향을 함께 보는 반면 이 지표는 방향만 본다.
    둘을 같이 기록하면 "크기 차이 때문인지 방향 차이 때문인지"를 구분할 수 있다.
    """
    if len(deltas) < 2:
        return 1.0
    stacked = np.stack(deltas)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True) + eps
    unit = stacked / norms
    sims = unit @ unit.T
    n = len(deltas)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def absolute_spread(deltas: Sequence["np.ndarray"]) -> float:
    """정규화하지 않은 흩어짐: mean_i || d_i - d_bar ||.

    divergence() 의 **분자**다. 둘을 함께 남겨야 하는 이유는 K 스윕에서 두 양이
    서로 반대로 움직이기 때문이다. K 가 커지면 각 업데이트가 커져 || d_bar || 가
    커지므로, 정규화된 divergence 는 **줄어든다**. 반면 절대 흩어짐은 **늘어난다**.

    "drift 가 관측 가능한 가장 작은 K" 를 고르려면 어느 쪽을 보느냐가 선택을
    바꾼다. 정규화된 값만 남기면 그 판단을 다시 할 때 전체 스윕을 다시 돌려야
    하므로(실측 13 시간) 둘 다 기록한다.
    """
    if len(deltas) < 2:
        return 0.0
    stacked = np.stack(deltas)
    return float(np.mean(np.linalg.norm(stacked - stacked.mean(axis=0), axis=1)))


def summarise(deltas: Sequence["np.ndarray"]) -> Dict[str, float]:
    """라운드마다 로그에 남기는 drift 요약."""
    stacked = np.stack(deltas) if deltas else np.zeros((0, 1))
    return {
        "divergence": divergence(deltas),
        "absolute_spread": absolute_spread(deltas),
        "mean_pairwise_cosine": mean_pairwise_cosine(deltas),
        "mean_update_norm": float(np.mean(np.linalg.norm(stacked, axis=1))) if len(deltas) else 0.0,
        "n_clients": len(deltas),
    }
