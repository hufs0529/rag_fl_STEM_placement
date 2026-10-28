"""기전 점검 (Subtask 1.4 secondary check).

주장한 인과 경로는 이렇다:

    reranking depth ↑  →  gold recall ↑  →  클라이언트가 같은 과제를 학습
                       →  update divergence ↓  →  필요한 라운드 수 ↓

따라서 라운드가 줄었는데 divergence 가 줄지 않았다면, 이득은 맞되 **다른
기전**으로 생긴 것이다(예: 더 좋은 컨텍스트가 단순히 손실을 낮춤). 계획서는
이 경우를 "다른 기전을 시사한다"고 보고하도록 요구한다. 여기서는 그 판정을
자동으로 내리지 않고, 세 지표(recall 증가, divergence 감소, 라운드 절감)를
나란히 놓고 정합 여부를 표시한다.

Mechanism check: do depth-driven recall gains coincide with reduced update
divergence? A gain without one would indicate a different mechanism.
"""

from statistics import median
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.analysis.curves import t_critical_95


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean_divergence(records: Sequence[dict], first_n_rounds: Optional[int] = None) -> Optional[float]:
    """run 의 평균 client-update divergence.

    first_n_rounds 를 주면 초반 라운드만 본다. drift 가 처리 효과를 만든다면
    그 효과는 어댑터가 아직 움직이는 초반에 나타나야 하기 때문이다.
    """
    values = [
        r["drift_divergence"] for r in records
        if r.get("event") == "round" and r.get("drift_divergence") is not None
        and r.get("round", 0) > 0
        and (first_n_rounds is None or r["round"] <= first_n_rounds)
    ]
    return float(np.mean(values)) if values else None


def sweep_summary(sweep: Sequence[Dict], baseline_depth: int = 0) -> Dict[int, Dict]:
    """tau **전구간**에서 depth 별 절약을 요약한다.

    왜 한 점이 아닌가: representative tau 하나를 고르면 결론이 그 선택의 산물이
    된다. 실측으로 tau=25.02 에서는 d=25 만 양수(+0.10)였지만, 63 점 전체에서는
    중앙값이 -0.113 이고 양수인 tau 는 24/63 뿐이다. 한 점은 오히려 그 depth 를
    실제보다 좋게 보이게 했다. 환율 표(5 절)도 같은 이유로 중앙값을 쓴다 - 예전에는
    그 표만 대표 tau 한 점이어서 d=3 과 d=25 의 부호가 3.3 절 표와 **반대로** 나왔다.

    왜 구간을 자르지 않는가: tau 상단에서는 기준선 곡선이 평평해져(tau 25.02 에서
    9.0 라운드였던 것이 28.14 에서 38.7 라운드) 작은 F1 차이가 큰 라운드 차이로
    번역되고, 절약이 ±11 까지 튄다. 그 구간을 **잘라내면** 어디서 자를지가 또
    임의 선택이 된다. 그래서 전부 남기고, 크기에 휘둘리지 않는 통계를 쓴다:
    중앙값과 "몇 개의 tau 에서 그런가" 라는 개수.
    """
    out: Dict[int, Dict] = {}
    depths = sorted({int(d) for e in sweep for d in e.get("depths", {})})
    for depth in depths:
        saved, resolved, positive = [], 0, 0
        for entry in sweep:
            cell = entry.get("depths", {}).get(str(depth))
            if not cell or cell.get("rounds_saved") is None:
                continue
            value = cell["rounds_saved"]
            # 분해능은 **차이 자신의** 95 % 문턱이다. 예전에는 depth 의 표준오차
            # 하나와 1 시그마로 비교했는데 둘 다 틀렸다: (1) 비교해야 할 것은
            # 차이의 표준오차이고, 시드로 짝지으면 2~5 배 좁다, (2) n=4 의 양측
            # 95 % 임계값은 1 도 2 도 아니라 3.182 다. 느슨한 쪽으로 틀려서
            # d=10 의 +0.60 +- 0.28 이 "유의" 로 읽혔다 (올바른 검정에서 p=0.127).
            threshold = cell.get("rounds_saved_threshold")
            if threshold is None:                   # 대응 통계가 없는 옛 산출물
                stderr = cell.get("rounds_saved_stderr") or cell.get("rounds_stderr")
                t_crit = t_critical_95((cell.get("seeds_reaching_tau") or 1) - 1)
                threshold = None if stderr is None else t_crit * stderr
            saved.append(value)
            if value > 0:
                positive += 1
                if threshold is not None and value > threshold:
                    resolved += 1
        n = len(saved)
        out[depth] = {
            "n_tau": n,
            "tau_with_saving": positive,
            "tau_with_resolved_saving": resolved,
            "fraction_with_saving": (positive / n) if n else None,
            "fraction_resolved": (resolved / n) if n else None,
            "median_rounds_saved": median(saved) if saved else None,
            "min_rounds_saved": min(saved) if saved else None,
            "max_rounds_saved": max(saved) if saved else None,
        }
    return out


def mechanism_table(
    recall_by_depth: Dict[int, float],
    divergence_by_depth: Dict[int, float],
    rounds_saved_by_depth: Dict[int, Optional[float]],
    rounds_saved_stderr_by_depth: Optional[Dict[int, Optional[float]]] = None,
    sweep_by_depth: Optional[Dict[int, Dict]] = None,
) -> Dict:
    """세 지표를 기준선(d=0) 대비 변화량으로 나란히 놓는다.

    rounds_saved 에는 불확실성을 함께 싣는다. 절약량만 보면 0.1 라운드와 2 라운드가
    같은 무게로 읽히는데, 이 측정의 표준오차는 0.2~0.9 라운드다.
    """
    depths = sorted(set(recall_by_depth) | set(divergence_by_depth))
    baseline_depth = min(depths) if depths else 0
    rows = []
    for depth in depths:
        recall = recall_by_depth.get(depth)
        divergence = divergence_by_depth.get(depth)
        base_recall = recall_by_depth.get(baseline_depth)
        base_divergence = divergence_by_depth.get(baseline_depth)
        rows.append({
            "depth": depth,
            "gold_recall_at_3": recall,
            "recall_gain": None if recall is None or base_recall is None else recall - base_recall,
            "divergence": divergence,
            "divergence_reduction": (
                None if divergence is None or base_divergence is None
                else base_divergence - divergence
            ),
            "rounds_saved": rounds_saved_by_depth.get(depth),
            "rounds_saved_stderr": (rounds_saved_stderr_by_depth or {}).get(depth),
            **{k: v for k, v in ((sweep_by_depth or {}).get(depth) or {}).items()},
        })
    return {"baseline_depth": baseline_depth, "rows": rows}


def assess_mechanism(table: Dict, tolerance: float = 1e-9) -> Dict:
    """정합성 판정 - 자동 결론이 아니라 보고서에 실을 관찰.

    반환의 verdict 는 네 가지 중 하나:
      consistent      : recall 이 오르고 divergence 가 줄고 라운드가 절약됨
      other_mechanism : 라운드는 절약되었으나 divergence 는 줄지 않음
      within_noise    : 양수 절약이 있으나 자기 불확실성 안에 들어 있음
      no_effect       : 양수 절약 자체가 없음

    **절약은 자기 표준오차를 넘어야 관측으로 친다.** 예전에는 `any(s > 1e-9)` 였고,
    그래서 한 조건의 +0.0988 라운드(표준오차 0.515, 0.2 시그마)가 나머지 넷이 전부
    음수인데도 판정을 no_effect 에서 consistent 로 뒤집었다. 보고서에 "가설과
    정합" 이라고 실릴 뻔한 자리다.
    """
    rows = [r for r in table["rows"] if r["depth"] != table["baseline_depth"]]
    saved = [r["rounds_saved"] for r in rows]
    gains = [r["recall_gain"] for r in rows]
    reductions = [r["divergence_reduction"] for r in rows]

    def resolved(row):
        """절약이 관측되었다고 할 수 있는가.

        tau 전구간 요약이 있으면 **과반의 tau 에서** 절약이 자기 표준오차를
        넘어야 한다. 한 점만 보면 그 점의 선택이 결론을 정한다 - 실측에서
        tau=25.02 한 점의 +0.0988(표준오차의 0.2 배)이 판정을 뒤집었다.
        요약이 없으면 예전처럼 그 행의 값과 오차만 본다.
        """
        share = row.get("fraction_resolved")
        if share is not None:
            return share > 0.5
        s, se = row["rounds_saved"], row.get("rounds_saved_stderr")
        if s is None or s <= tolerance:
            return False
        return True if se is None else s > se

    def any_positive(row):
        """어느 target 에서든 양수 절약이 있었는가.

        중앙값만 보면 '일부 tau 에서는 아꼈으나 전반적으로는 아니다' 가
        '절약이 전혀 없었다' 로 내려앉는다. 둘은 다른 관찰이다.
        """
        if (row.get("tau_with_saving") or 0) > 0:
            return True
        return row["rounds_saved"] is not None and row["rounds_saved"] > tolerance

    positive = [r for r in rows if any_positive(r)]
    observed = [r for r in rows if resolved(r)]
    any_saving = bool(observed)
    any_reduction = any(d is not None and d > tolerance for d in reductions)

    if not any_saving and positive:
        best = max(positive, key=lambda r: (r.get("fraction_resolved") or 0,
                                            r["rounds_saved"] or 0))
        se = best.get("rounds_saved_stderr")
        verdict = "within_noise"
        share = best.get("fraction_resolved")
        n_tau = best.get("n_tau")
        if share is not None and n_tau:
            note = (f"the largest median saving, {best['rounds_saved']:.3f} rounds at "
                    f"d={best['depth']}, clears its own standard error at only "
                    f"{best.get('tau_with_resolved_saving')} of {n_tau} targets; no depth "
                    f"resolves a saving at a majority of targets")
        else:
            note = (f"the largest rounds-to-target saving, {best['rounds_saved']:.3f} rounds at "
                    f"d={best['depth']}, is inside its own uncertainty"
                    + (f" (s.e. {se:.3f})" if se is not None else "")
                    + "; no saving is resolved by this measurement")
    elif not any_saving:
        verdict = "no_effect"
        note = "no rounds-to-target saving was observed, so there is no mechanism to explain"
    elif any_reduction:
        verdict = "consistent"
        note = ("recall gains coincide with reduced client-update divergence, consistent with "
                "the proposed path: better retrieval -> clients learn the same task -> less drift")
    else:
        verdict = "other_mechanism"
        note = ("rounds were saved without a reduction in client-update divergence: the gain is "
                "real but arrives by a different mechanism than the one proposed")

    return {
        "verdict": verdict,
        "note": note,
        "depths_with_resolved_saving": [r["depth"] for r in observed],
        "corr_recall_vs_divergence_reduction": pearson(gains, reductions),
        "corr_recall_vs_rounds_saved": pearson(gains, saved),
        "corr_divergence_reduction_vs_rounds_saved": pearson(reductions, saved),
    }
