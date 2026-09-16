"""FL 통신 payload 추출/복원과 바이트 회계 (Subtask 1.2 / 1.3).

LoRA 는 frozen base LLM 의 업데이트를 저랭크 곱 AB 로 근사하므로 A, B 만
학습되고 통신된다. 여기서 뽑아내는 텐서 집합이 곧 라운드당 payload 이며,
Subtask 1.4 의 "절약된 바이트" 계산의 단위가 된다.

Parameter extraction/round-trip for FL communication. The tensors pulled out
here are exactly the per-round payload and the unit in which Subtask 1.4
counts saved bytes.
"""

from collections import OrderedDict
from typing import Dict, List


def get_trainable_state_dict(model) -> "OrderedDict":
    """학습되는(=통신되는) 파라미터만, 키 순서를 정렬해 결정적으로.

    **clone() 이 필수다.** detach() 는 저장소를 공유하고, CPU 에서 .cpu() 는
    아무 일도 하지 않으므로, clone 없이는 반환된 텐서가 모델의 **살아있는**
    파라미터를 가리키는 뷰가 된다. 그러면 이 함수로 찍은 "스냅샷"이 이후 학습에
    따라 함께 변하고, 여러 클라이언트의 파라미터를 모아 비교하면 전부 마지막
    상태로 수렴해 divergence 가 정확히 0 으로 나온다 (실측: cos +1.000).

    GPU 에서는 .cpu() 가 복사를 수반해 증상이 숨는다 - CPU 에서만 드러나는
    종류의 결함이라 더 위험하다.
    """
    return OrderedDict(
        (k, v.detach().cpu().clone())
        for k, v in sorted(model.named_parameters(), key=lambda kv: kv[0])
        if v.requires_grad
    )


def set_trainable_state_dict(model, state_dict: Dict) -> None:
    named = dict(model.named_parameters())
    for k, v in state_dict.items():
        named[k].data.copy_(v.to(named[k].device).to(named[k].dtype))


def state_dict_to_numpy(state_dict: Dict) -> "OrderedDict":
    """numpy 로 변환. **반드시 복사한다** - get_trainable_state_dict 와 같은 이유다.

    v.to(dtype=float32) 는 이미 float32 면 같은 텐서를 돌려주고, .numpy() 는
    메모리를 공유한다. 복사하지 않으면 반환된 배열이 모델 파라미터의 뷰가 되어
    나중에 조용히 변한다.
    """
    import numpy as np

    return OrderedDict(
        (k, np.array(v.to(dtype=_float32()).numpy(), copy=True))
        for k, v in state_dict.items()
    )


def numpy_to_state_dict(arrays: Dict) -> "OrderedDict":
    import torch

    return OrderedDict((k, torch.tensor(v)) for k, v in arrays.items())


def state_dict_to_ndarrays(state_dict: Dict) -> List:
    """Flower 가 주고받는 ndarray 리스트 형태 (키 순서는 호출자가 보관).

    state_dict_to_numpy 와 같은 이유로 복사한다 - Flower 가 payload 를 직렬화하기
    전에 로컬 학습이 더 진행되면, 복사하지 않은 배열은 그 변화를 함께 반영한다.
    """
    import numpy as np

    return [np.array(v.to(dtype=_float32()).numpy(), copy=True) for v in state_dict.values()]


def ndarrays_to_state_dict(keys: List[str], arrays: List) -> "OrderedDict":
    import torch

    return OrderedDict((k, torch.tensor(a)) for k, a in zip(keys, arrays))


def _float32():
    import torch

    return torch.float32


def payload_bytes_of(state_dict: Dict, dtype_width: int = 2) -> int:
    """payload 바이트. 기본 2 = fp16 전송 기준 (계획서의 1.84 MB 산정과 동일)."""
    return sum(v.numel() * dtype_width for v in state_dict.values())


def cumulative_bytes(rounds: int, num_clients: int, payload: int) -> int:
    """총 통신량 = R x N x |AB| x 2 (업로드 + 다운로드)."""
    return rounds * num_clients * payload * 2
