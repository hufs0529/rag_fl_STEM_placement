"""FedAvg 집계 (Subtask 1.2).

서버는 클라이언트 어댑터를 평균해 전역 (A, B) 쌍으로 만들고 매 라운드
재배포한다. 전 클라이언트 참여(full participation)이고 파티션이 동일 크기이므로
가중치는 균등하지만, 가중 평균도 지원해 두어 데이터 크기가 달라지는 변형을
나중에 검토할 수 있게 한다.

FedAvg aggregation. Full participation over an equal-size partition means the
weights are uniform, but weighted averaging is supported so an unequal-size
variant can be examined later.
"""

from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np


def fedavg(
    client_params: Sequence[Dict[str, np.ndarray]],
    weights: Sequence[float] = None,
) -> "OrderedDict":
    if not client_params:
        raise ValueError("no client parameters to aggregate")
    keys = list(client_params[0].keys())
    for params in client_params[1:]:
        if list(params.keys()) != keys:
            raise ValueError("client parameter keys differ; aggregation would be unsound")

    if weights is None:
        weights = [1.0] * len(client_params)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    norm = [w / total for w in weights]

    out = OrderedDict()
    for key in keys:
        stacked = np.stack([np.asarray(p[key], dtype=np.float64) for p in client_params])
        out[key] = np.tensordot(np.asarray(norm), stacked, axes=(0, 0)).astype(np.float32)
    return out
