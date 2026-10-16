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


# 2 x params x tokens 의 params 는 **연산에 참여하는** 파라미터여야 한다.
# 임베딩 행렬은 gather 로 읽히고 행렬곱에 들어가지 않으므로 FLOPs 를 거의 내지
# 않는데, 작은 모델에서는 그것이 파라미터의 절반을 차지한다. 전체 수를 넣으면
# 비용을 2 배 넘게 과대계상하고, 그 값이 환율의 **분모**이므로 결론을 직접 왜곡한다.
#
#   ms-marco-MiniLM-L-6-v2   전체 22,713,601 | 임베딩 11,917,824 (52%) | 연산 10,795,777
#   bge-small-en-v1.5        전체 33,360,000 | 임베딩 11,917,824 (36%) | 연산 21,442,176
#
# (두 모델의 임베딩 수가 같은 것은 같은 30,522 단어 WordPiece 사전과 hidden 384
#  를 쓰기 때문이다.)

CE_TOKENS = 256          # 질문 + 100-word passage 의 실제 평균에 가깝다
EMBED_TOKENS = 128


def cross_encoder_flops_per_pair(
    n_params: int = 10_795_777,      # ms-marco-MiniLM-L-6-v2, 임베딩 제외
    n_tokens: int = CE_TOKENS,
    n_layers: int = 6,
    hidden: int = 384,
) -> float:
    """(query, passage) 한 쌍을 재채점하는 비용. depth d 는 이 값의 d 배."""
    return transformer_forward_flops(n_params, n_tokens, n_layers, hidden)


def embedder_flops_per_passage(
    n_params: int = 21_442_176,      # bge-small-en-v1.5, 임베딩 제외
    n_tokens: int = EMBED_TOKENS,
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


def describe(
    measured: Optional[float], analytic: float, label: str,
    n_tokens_measured: Optional[int] = None, n_tokens_assumed: Optional[int] = None,
) -> Dict:
    """실측/추정 중 무엇을 썼는지, 그리고 둘이 얼마나 벌어졌는지 남긴다.

    두 값이 크게 벌어지면 **조용히 한쪽을 쓰면 안 된다.** 이 값은 환율의 분모라서
    배수 오차가 결론을 그대로 뒤집는다. 실제로 한 번 그렇게 벌어졌다: 실측은
    66 토큰 샘플에서 재고 추정식은 256 토큰을 가정해 3.9 배, 추정식이 임베딩
    파라미터까지 세어 2.1 배, 합쳐 8.7 배였다.
    """
    out = {
        "label": label,
        "flops": float(measured if measured else analytic),
        "source": "measured" if measured else "analytic",
        "analytic_estimate": analytic,
        "measured_estimate": measured,
        "n_tokens_assumed": n_tokens_assumed,
        "n_tokens_measured": n_tokens_measured,
    }
    if measured and analytic:
        ratio = max(measured, analytic) / min(measured, analytic)
        out["measured_over_analytic"] = measured / analytic
        out["agree_within_2x"] = ratio < 2.0
        if ratio >= 2.0:
            out["warning"] = (
                f"측정과 추정이 {ratio:.1f} 배 벌어졌다. 이 값은 환율의 분모이므로 "
                "토큰 수와 파라미터 집계를 확인할 것"
            )
    return out
