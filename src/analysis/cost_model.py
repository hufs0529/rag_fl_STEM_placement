"""2항 비용 모델과 환율·손익분기 (Subtask 1.4).

이 실험의 결론은 "reranking 이 좋다/나쁘다"가 아니라 **운영 구간**이다.
그러려면 로컬 계산과 네트워크 전송을 같은 단위로 환산해야 한다. 여기서는 초(second)를
공통 단위로 쓴다:

    절약된 전송 시간 = 절약된 바이트 x 8 / 대역폭(bps)
    추가된 계산 시간 = (일회성 indexing + 쿼리당 reranking) FLOPs / 장치 처리량(FLOP/s)

2항 모델:
    C_index  = |corpus| x FLOPs_embed(passage)          <- 일회성
    C_rerank = Q x d x FLOPs_ce(query, passage)          <- 쿼리당 반복

reranking 은 대역폭이 **낮을수록** 유리하다(전송이 비싸므로). 쿼리 수 Q 가
커질수록 불리하다(반복 비용이 쌓이므로). 그래서 결론은 (대역폭, 쿼리 수) 평면
위의 손익분기 곡선이 된다.

Two-term cost model with an explicit exchange rate. Local compute and network
transfer are converted into the same unit (seconds), and the verdict is a
break-even contour over bandwidth x query volume rather than a single verdict.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

BITS_PER_BYTE = 8


def total_bytes(rounds: float, num_clients: int, payload_bytes: int) -> float:
    """R x N x |AB| x 2 (업로드 + 다운로드)."""
    return rounds * num_clients * payload_bytes * 2


def saved_bytes(
    baseline_rounds: Optional[float],
    treated_rounds: Optional[float],
    num_clients: int,
    payload_bytes: int,
) -> Optional[float]:
    """목표 정확도 도달에 필요한 라운드 차이를 바이트로 환산.

    어느 한쪽이 tau 에 도달하지 못했으면 None - 외삽하지 않는다.
    """
    if baseline_rounds is None or treated_rounds is None:
        return None
    return total_bytes(baseline_rounds - treated_rounds, num_clients, payload_bytes)


def indexing_flops(corpus_size: int, flops_per_passage: float) -> float:
    """일회성 항. depth 와 무관하므로 처리 조건 간 차이를 만들지 않는다.

    (그래서 처리 축으로 encoder 용량이 아니라 depth 를 고른 것이다 - depth 는
    반복 비용이라 라운드당 통신과 같은 단위로 비교된다.)
    """
    return float(corpus_size) * float(flops_per_passage)


def reranking_flops(query_volume: float, depth: int, flops_per_pair: float) -> float:
    """반복 항. depth 에 정비례한다.

    설정에서 온 값이 문자열일 수 있어(YAML 의 부호 없는 지수 표기) float 로
    강제한다 - 조용히 문자열 곱셈이 되는 것보다 낫다.
    """
    return float(query_volume) * max(0, int(depth)) * float(flops_per_pair)


def compute_seconds(flops: float, device_throughput: float) -> float:
    return float(flops) / float(device_throughput)


def transfer_seconds(byte_count: float, bandwidth_mbps: float) -> float:
    return float(byte_count) * BITS_PER_BYTE / (float(bandwidth_mbps) * 1e6)


def exchange_rate(saved: Optional[float], extra_flops: float) -> Optional[float]:
    """환율: 추가 FLOP 1 당 절약되는 바이트 수.

    대역폭·장치와 무관한 순수 물량비라, 비용 가정을 바꿔도 다시 계산할 필요가
    없는 유일한 수치다. 손익분기는 여기에 대역폭과 처리량을 곱해 얻는다.
    """
    if saved is None or extra_flops <= 0:
        return None
    return saved / extra_flops


def net_benefit_seconds(
    saved: Optional[float],
    extra_flops: float,
    bandwidth_mbps: float,
    device_throughput: float,
) -> Optional[float]:
    """양수면 reranking 이 이득. 절약된 전송 시간 - 추가된 계산 시간."""
    if saved is None:
        return None
    return transfer_seconds(saved, bandwidth_mbps) - compute_seconds(extra_flops, device_throughput)


def break_even_bandwidth_mbps(
    saved: Optional[float], extra_flops: float, device_throughput: float
) -> Optional[float]:
    """손익분기 대역폭. 이보다 **느린** 링크에서는 reranking 이 이득이다.

    saved x 8 / B = flops / T  ->  B* = saved x 8 x T / flops
    """
    if saved is None or extra_flops <= 0:
        return None
    if saved <= 0:
        return 0.0
    return saved * BITS_PER_BYTE * device_throughput / extra_flops / 1e6


def depth_cost(
    depth: int,
    query_volume: float,
    corpus_size: int,
    flops_per_pair: float,
    flops_per_passage: float,
    amortise_indexing: bool = True,
) -> Dict[str, float]:
    """한 depth 조건의 총 계산 비용을 2항으로 분해.

    amortise_indexing=False 는 "이 실험을 위해 인덱스를 새로 만든다"는 가정이고,
    True 는 "RAG 를 이미 배포 중이라 인덱스는 이미 있다"는 가정이다. 후자가
    현실적이지만 전자가 보수적이므로 둘 다 보고한다.
    """
    index = 0.0 if amortise_indexing else indexing_flops(corpus_size, flops_per_passage)
    rerank = reranking_flops(query_volume, depth, flops_per_pair)
    return {
        "indexing_flops": index,
        "reranking_flops": rerank,
        "total_flops": index + rerank,
        "amortise_indexing": amortise_indexing,
    }


def break_even_surface(
    saved: Optional[float],
    depth: int,
    bandwidths_mbps: Sequence[float],
    query_volumes: Sequence[float],
    corpus_size: int,
    flops_per_pair: float,
    flops_per_passage: float,
    device_throughput: float,
    amortise_indexing: bool = True,
) -> Dict:
    """(대역폭 x 쿼리 수) 격자 위의 순이득과 손익분기 곡선.

    격자의 각 칸은 "그 조건에서 reranking 을 켜면 총 시간이 줄어드는가"이며,
    contour 는 그 경계다. 결론을 단일 판정이 아니라 구간으로 보고하기 위한 형식.
    """
    grid = np.zeros((len(query_volumes), len(bandwidths_mbps)))
    contour: List[Dict] = []

    for i, queries in enumerate(query_volumes):
        cost = depth_cost(depth, queries, corpus_size, flops_per_pair,
                          flops_per_passage, amortise_indexing)
        for j, bandwidth in enumerate(bandwidths_mbps):
            net = net_benefit_seconds(saved, cost["total_flops"], bandwidth, device_throughput)
            grid[i, j] = np.nan if net is None else net
        contour.append({
            "query_volume": float(queries),
            "break_even_bandwidth_mbps": break_even_bandwidth_mbps(
                saved, cost["total_flops"], device_throughput
            ),
            "total_flops": cost["total_flops"],
        })

    return {
        "depth": depth,
        "saved_bytes": saved,
        "bandwidths_mbps": list(map(float, bandwidths_mbps)),
        "query_volumes": list(map(float, query_volumes)),
        "net_benefit_seconds": grid.tolist(),
        "break_even_contour": contour,
        "amortise_indexing": amortise_indexing,
        "reading": (
            "positive net benefit = reranking pays for itself; reranking wins on links "
            "SLOWER than the break-even bandwidth and loses as query volume grows"
        ),
    }
