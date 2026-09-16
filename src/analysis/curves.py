"""학습곡선 보간과 목표 정확도 도달 라운드 (Subtask 1.4).

핵심 측정: "목표 정확도 tau 에 도달하는 라운드"를 곡선의 선형 보간으로 읽는다.
고정 예산 + early stopping 없음으로 실험을 돌렸기 때문에, 결론이 정지 규칙에
의존하지 않는다.

tau 는 하나로 고정하지 않는다. 고정하면 결론이 tau 선택에 의존하므로,
baseline 이 도달한 구간 전체에 대해 스윕하고 절감량을 **곡선으로** 보고한다.
어떤 조건이 tau 에 끝내 도달하지 못한 구간은 외삽하지 않고 "도달하지 못함"으로
보고한다.

Rounds-to-target by linear interpolation of the fixed-budget curves, swept over
tau rather than read at a single chosen tau. Regions where a condition never
reaches tau are reported as such rather than extrapolated.
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def rounds_to_target(curve: Sequence[dict], tau: float, metric: str = "f1") -> Optional[float]:
    """tau 에 처음 도달하는 라운드 (선형 보간). 끝내 도달 못 하면 None.

    곡선은 평가된 라운드에서만 표본이 있으므로, tau 를 넘어선 첫 지점과 그
    직전 지점 사이를 선형 보간한다. 라운드 수가 이 규모에서 너무 거친 자라는
    문제를 이 보간이 해결한다.
    """
    points = sorted(
        ((p["round"], p[metric]) for p in curve if p.get(metric) is not None),
        key=lambda rv: rv[0],
    )
    if not points:
        return None

    previous = None
    for round_number, value in points:
        if value >= tau:
            if previous is None:
                return float(round_number)
            prev_round, prev_value = previous
            if value == prev_value:
                return float(round_number)
            fraction = (tau - prev_value) / (value - prev_value)
            return float(prev_round + fraction * (round_number - prev_round))
        previous = (round_number, value)
    return None


def attainable_range(curves: Sequence[Sequence[dict]], metric: str = "f1") -> tuple:
    """baseline 곡선들이 실제로 도달한 정확도 구간 - tau 스윕의 범위.

    이 구간 밖의 tau 는 baseline 이 도달하지 못하므로 "절감"을 정의할 수 없다.
    """
    starts, peaks = [], []
    for curve in curves:
        values = [p[metric] for p in curve if p.get(metric) is not None]
        if values:
            starts.append(values[0])
            peaks.append(max(values))
    if not peaks:
        return (0.0, 0.0)
    return (max(starts), min(peaks))


def tau_grid(curves: Sequence[Sequence[dict]], points: int = 25, metric: str = "f1") -> List[float]:
    low, high = attainable_range(curves, metric)
    if high <= low:
        return []
    return list(np.linspace(low, high, points))


def mean_and_stderr(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """시드 요약: 평균과 표준오차. 유의성 검정이 아니라 효과 크기와 구간을 보고한다.

    계획서: "Seeds are summarised as mean with standard error, reporting effect
    sizes and intervals rather than significance tests." 시드 5개로 검정을 하면
    검정력이 없고, 없는 검정력으로 얻은 p 값은 결론을 오도한다.
    """
    present = [v for v in values if v is not None and not math.isnan(v)]
    n = len(present)
    if n == 0:
        return {"mean": None, "stderr": None, "n": 0, "missing": len(values)}
    mean = sum(present) / n
    if n == 1:
        return {"mean": mean, "stderr": None, "n": 1, "missing": len(values) - n}
    variance = sum((v - mean) ** 2 for v in present) / (n - 1)
    return {
        "mean": mean,
        "stderr": math.sqrt(variance / n),
        "n": n,
        "missing": len(values) - n,
    }


def rounds_by_depth(
    curves_by_depth: Dict[int, List[List[dict]]], tau: float, metric: str = "f1"
) -> Dict[int, Dict]:
    """depth 별로 시드마다 rounds-to-tau 를 구하고 평균·표준오차로 요약.

    시드별로 보간한 뒤 평균을 낸다(곡선을 먼저 평균내지 않는다). 곡선을 먼저
    평균내면 시드 간 분산이 사라져 구간을 보고할 수 없다.
    """
    out = {}
    for depth, curves in sorted(curves_by_depth.items()):
        per_seed = [rounds_to_target(c, tau, metric) for c in curves]
        out[depth] = {**mean_and_stderr(per_seed), "per_seed": per_seed}
    return out


def curve_mean(curves: Sequence[Sequence[dict]], metric: str = "f1") -> List[dict]:
    """같은 라운드 스케줄 위에서 시드 평균 곡선 (그림용).

    모든 run 이 동일한 스케줄로 평가되었음은 Week 4 의 무결성 점검이 보장한다.
    """
    by_round: Dict[int, List[float]] = {}
    for curve in curves:
        for point in curve:
            if point.get(metric) is not None:
                by_round.setdefault(point["round"], []).append(point[metric])
    out = []
    for round_number in sorted(by_round):
        stats = mean_and_stderr(by_round[round_number])
        out.append({"round": round_number, "mean": stats["mean"],
                    "stderr": stats["stderr"], "n": stats["n"]})
    return out


def final_scores(curves: Sequence[Sequence[dict]], metric: str = "f1") -> Dict:
    """고정 예산이 끝난 시점의 점수 - "그 예산이 무엇을 사주는가"의 직접 답."""
    finals = []
    for curve in curves:
        values = [p[metric] for p in curve if p.get(metric) is not None]
        if values:
            finals.append(values[-1])
    return mean_and_stderr(finals)
