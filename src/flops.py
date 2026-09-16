"""transformer forward pass FLOPs 추정 (Subtask 1.4 비용 모델의 입력).

비용 모델은 reranking FLOPs 를 절약된 바이트와 맞바꾼다. 따라서 "한 쌍을
재채점하는 데 몇 FLOP 이 드는가"를 실측하거나 최소한 명시적 모델로 추정해야
하고, 그 가정이 보고서에 그대로 드러나야 한다.

추정식(표준 근사): 인코더 한 번 forward 는 파라미터당 토큰당 약 2 FLOP.
  FLOPs ≈ 2 x params x tokens  +  attention 항 (4 x layers x tokens^2 x hidden)

torch profiler 로 실측이 가능하면 measure_* 함수가 실측치를 돌려주고,
불가능하면 이 해석적 추정을 쓴다. 어느 쪽을 썼는지 결과에 함께 기록한다.

FLOPs estimation for the two-term cost model. Measured where possible, analytic
otherwise, and the result always records which was used.
"""

from typing import Dict, Optional


def transformer_forward_flops(
    n_params: int, n_tokens: int, n_layers: int = None, hidden: int = None
) -> float:
    """forward 1회 FLOPs 근사. 2 x params x tokens + attention 이차항."""
    flops = 2.0 * n_params * n_tokens
    if n_layers and hidden:
        flops += 4.0 * n_layers * (n_tokens ** 2) * hidden
    return flops


def cross_encoder_flops_per_pair(
    n_params: int = 22_700_000,      # ms-marco-MiniLM-L-6-v2
    n_tokens: int = 256,             # 질문 + 100-word passage 의 실제 평균에 가깝다
    n_layers: int = 6,
    hidden: int = 384,
) -> float:
    """(query, passage) 한 쌍을 재채점하는 비용. depth d 는 이 값의 d 배."""
    return transformer_forward_flops(n_params, n_tokens, n_layers, hidden)


def embedder_flops_per_passage(
    n_params: int = 33_400_000,      # bge-small-en-v1.5
    n_tokens: int = 128,
    n_layers: int = 12,
    hidden: int = 384,
) -> float:
    """passage 하나를 임베딩하는 비용 - 일회성 indexing 항의 단위량."""
    return transformer_forward_flops(n_params, n_tokens, n_layers, hidden)


def measure_forward_flops(model, sample_inputs, device: str = None) -> Optional[float]:
    """torch profiler 로 forward FLOPs 실측. 실패하면 None 을 돌려 추정식으로 넘긴다."""
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        model.eval()
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                model(**sample_inputs)
        total = sum(evt.flops for evt in prof.key_averages() if evt.flops)
        return float(total) if total else None
    except Exception:
        return None


def describe(measured: Optional[float], analytic: float, label: str) -> Dict:
    """실측/추정 중 무엇을 썼는지 명시해 결과에 남긴다."""
    return {
        "label": label,
        "flops": float(measured if measured else analytic),
        "source": "measured" if measured else "analytic",
        "analytic_estimate": analytic,
        "measured_estimate": measured,
    }
