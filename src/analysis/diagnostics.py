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

from typing import Dict, List, Optional, Sequence

import numpy as np


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


def mechanism_table(
    recall_by_depth: Dict[int, float],
    divergence_by_depth: Dict[int, float],
    rounds_saved_by_depth: Dict[int, Optional[float]],
    rounds_saved_stderr_by_depth: Optional[Dict[int, Optional[float]]] = None,
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
        """절약이 자기 불확실성보다 큰가. 표준오차를 모르면 부호만 본다."""
        s, se = row["rounds_saved"], row.get("rounds_saved_stderr")
        if s is None or s <= tolerance:
            return False
        return True if se is None else s > se

    positive = [r for r in rows if r["rounds_saved"] is not None
                and r["rounds_saved"] > tolerance]
    observed = [r for r in rows if resolved(r)]
    any_saving = bool(observed)
    any_reduction = any(d is not None and d > tolerance for d in reductions)

    if not any_saving and positive:
        best = max(positive, key=lambda r: r["rounds_saved"])
        se = best.get("rounds_saved_stderr")
        verdict = "within_noise"
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
