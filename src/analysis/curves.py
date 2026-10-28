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
    """baseline 곡선들이 실제로 쓸 수 있는 tau 구간. 두 경계 모두 **유도된다.**

    하한 = max(라운드 1 의 값).  라운드 0 의 값이 아니다.
      곡선은 라운드 0 에서 1 사이에 거의 수직으로 오른다(실측 0.5 -> 17~18,
      최종값의 60%). 그 구간 안의 tau 는 보간이 "라운드 0.54 에 도달" 같은 값을
      내는데, 라운드는 서버가 어댑터를 모으는 최소 단위이므로 실행할 수 없는
      값이다. 네 시드 모두 라운드 1 을 넘겨야 하므로 **max** 를 쓴다.
      (라운드 0 을 하한으로 쓰면 25 점 격자의 15 점이 이 구간에 빠진다.)

    상한 = min(최고값).
      baseline 이 도달하지 못하는 tau 에서는 절감(= 라운드(d0) - 라운드(d))의
      왼쪽 항이 없다. 그리고 탈락하는 시드는 무작위가 아니다 - 최고값이 낮은
      시드부터 빠지므로, 남는 것은 **원래 잘한 시드**다. 실측에서 tau=29 는
      시드 2, 4(최고 29.92, 30.66)만 남기고 1, 3(28.44, 28.14)을 버린다.
    """
    firsts, peaks = [], []
    for curve in curves:
        points = sorted(
            ((p["round"], p[metric]) for p in curve if p.get(metric) is not None),
            key=lambda rv: rv[0],
        )
        after_first = [v for r, v in points if r >= 1]
        if after_first:
            firsts.append(after_first[0])
            peaks.append(max(after_first))
    if not peaks:
        return (0.0, 0.0)
    return (max(firsts), min(peaks))


def tau_grid(curves: Sequence[Sequence[dict]], points: int = None, metric: str = "f1") -> List[float]:
    """tau 격자를 **데이터가 정한다.** baseline 곡선의 평가점 값 자체를 쓴다.

    균등 격자를 쓰면 간격을 사람이 정해야 하고("왜 25 점인가"), 곡선이 급한
    구간에 점이 몰린다. 그리고 보간은 평가점 사이에서 직선이므로, 한 구간 안의
    여러 tau 는 같은 직선 위에서 읽혀 **독립 정보가 아니다** - 곡선의 상승
    구간 수가 쓸 수 있는 tau 의 상한이다(이 실험에서 16 개).

    baseline 이 실제로 낸 값을 tau 로 쓰면 구간마다 정확히 하나씩 떨어지고,
    격자를 선언할 필요가 없어진다. points 는 호환성을 위해 받지만 무시한다.
    """
    low, high = attainable_range(curves, metric)
    if high <= low:
        return []
    values = {
        round(float(p[metric]), 4)
        for curve in curves for p in curve
        if p.get(metric) is not None and p.get("round", 0) >= 1
        and low <= p[metric] <= high
    }
    return sorted(values)


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


def matched_budget_table(
    curves_by_depth: Dict[int, List[List[dict]]],
    baseline_depth: int = 0,
    metric: str = "f1",
) -> List[dict]:
    """**세로 읽기**: 같은 통신 예산에서의 metric 차이. tau 를 고르지 않는다.

    가로 읽기(tau 를 고정하고 라운드를 읽는 것)는 tau 격자를 사람이 정해야 하고,
    평가 노이즈가 **기울기로 나뉘어** 증폭된다. 실측 기울기는 r30->r40 에서
    0.0065 F1/라운드이고 r50->r59 에서는 음수다. 노이즈 +-0.3 F1 이 거기서는
    수십 라운드가 되고, 부호까지 뒤집힌다.

    세로 읽기는 그 문제가 없다:
      - 평가 스케줄이 예산을 이미 고정한다. 고를 격자가 없다.
      - 노이즈가 측정된 축(metric)에 그대로 남는다 - 효과와 같은 단위로 비교된다.
      - 곡선이 비단조여도 같은 라운드끼리 빼므로 영향이 없다.

    연구 질문은 바이트 단위이므로 가로 읽기도 필요하지만, 1 차 결과는 이쪽이다.
    두 읽기가 겹치는 구간에서 부호가 일치하는 것이 단일 tau 를 정당화하는 것보다
    강한 근거가 된다.
    """
    baseline = curves_by_depth.get(baseline_depth) or []
    if not baseline:
        return []
    rounds = sorted({p["round"] for p in baseline[0] if p.get(metric) is not None})
    out = []
    for rnd in rounds:
        if rnd < 1:
            continue
        row = {"round": rnd}
        base_at = [_at_round(c, rnd, metric) for c in baseline]
        row["baseline_mean"] = mean_and_stderr(base_at)["mean"]
        row["cumulative_bytes"] = next(
            (p.get("cumulative_bytes") for p in baseline[0] if p["round"] == rnd), None
        )
        row["depths"] = {}
        for depth, curves in sorted(curves_by_depth.items()):
            if depth == baseline_depth or len(curves) != len(baseline):
                continue
            deltas = [
                None if (a is None or b is None) else a - b
                for a, b in zip((_at_round(c, rnd, metric) for c in curves), base_at)
            ]
            summary = mean_and_stderr(deltas)
            se = summary.get("stderr")
            summary["separated_from_zero"] = bool(
                se and summary["mean"] is not None and abs(summary["mean"]) > 2 * se
            )
            row["depths"][str(depth)] = summary
        out.append(row)
    return out


def _at_round(curve: Sequence[dict], rnd: int, metric: str) -> Optional[float]:
    for p in curve:
        if p.get("round") == rnd:
            return p.get(metric)
    return None


def local_slope(curves: Sequence[Sequence[dict]], r0: int, r1: int, metric: str = "f1"):
    """[r0, r1] 구간의 평균 기울기 (metric / 라운드). 세로 차이를 바이트로 바꿀 때 쓴다.

    **단일 값으로 보고하면 안 된다.** 실측에서 추정 구간에 따라 3 배 이상
    달라지고 부호도 뒤집힌다(r50->r59 는 음수). 호출부는 여러 구간에서 재서
    범위로 보고해야 한다.
    """
    a = mean_and_stderr([_at_round(c, r0, metric) for c in curves])["mean"]
    b = mean_and_stderr([_at_round(c, r1, metric) for c in curves])["mean"]
    if a is None or b is None or r1 == r0:
        return None
    return (b - a) / (r1 - r0)


# --- 분해능 (Week 5 보강) ---------------------------------------------------
#
# 예전에는 "절약이 자기 표준오차보다 크면 관측" 이라는 1 시그마 규칙이었고,
# 보고서 본문은 다시 "표준오차의 2 배" 로 적었다. 둘 다 n=4 에서는 틀렸다.
# 시드 4 개면 자유도 3 이고, 양측 95 % 임계값은 1.96 도 2 도 아니라 **3.182** 다.
# 느슨한 쪽으로 1.6 배 틀렸고, 그래서 d=10 의 +0.60 +- 0.28 이 "유의" 로 읽혔다
# (올바른 검정에서는 p = 0.127).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042}


def t_critical_95(df: int) -> float:
    """양측 95 % t 임계값. df 가 표에 없으면 보수적으로 아래쪽 df 값을 쓴다."""
    if df < 1:
        return float("inf")
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k < df]
    return _T95[max(smaller)] if smaller else 1.960


def permutation_floor_p(n: int) -> Optional[float]:
    """부호 치환검정으로 **낼 수 있는 가장 작은** 양측 p.

    대응 설계에서 치환은 n 개 차이의 부호를 뒤집는 것뿐이므로 배열이 2^n 가지다.
    n=4 면 16 가지이고 최소 양측 p 는 2/16 = 0.125 - 데이터가 아무리 깨끗해도
    **p < 0.05 가 구조적으로 불가능**하다. 이 한 줄이 within_noise 판정을
    변명이 아니라 설계 사실로 만든다. n=6 부터 0.031 로 가능해진다.
    """
    if n < 1:
        return None
    return 2.0 / (2 ** n)


def paired_saving(
    baseline_per_seed: Sequence[Optional[float]],
    depth_per_seed: Sequence[Optional[float]],
) -> Dict[str, Optional[float]]:
    """시드로 짝지은 라운드 절약 (baseline - depth).

    **왜 대응(paired)인가.** 시드마다 난수 분할과 초기화가 달라 라운드 수가
    통째로 올라가거나 내려간다. 두 조건을 같은 시드에서 빼면 그 성분이 상쇄된다.
    실측으로 비대응 차이의 표준오차는 0.879 라운드인데 대응은 0.16~0.47 이다 -
    같은 데이터에서 분해능이 2~5 배 좋아진다.

    두 수열은 같은 시드 순서여야 한다 (enumerate_runs 가 seed 외부 / depth
    내부로 돌기 때문에 group_by_depth 의 결과가 시드 순서다).
    """
    pairs = [
        (b, d) for b, d in zip(baseline_per_seed, depth_per_seed)
        if b is not None and d is not None
        and not math.isnan(b) and not math.isnan(d)
    ]
    n = len(pairs)
    if n == 0:
        return {"mean": None, "stderr": None, "n": 0,
                "t_critical": None, "threshold": None, "resolved": False}
    diffs = [b - d for b, d in pairs]
    stats = mean_and_stderr(diffs)
    stderr = stats["stderr"]
    t_crit = t_critical_95(n - 1) if n > 1 else None
    threshold = None if stderr is None or t_crit is None else t_crit * stderr
    return {
        "mean": stats["mean"],
        "stderr": stderr,
        "n": n,
        "t_critical": t_crit,
        "threshold": threshold,
        "resolved": bool(
            threshold is not None and stats["mean"] is not None
            and abs(stats["mean"]) > threshold
        ),
    }
