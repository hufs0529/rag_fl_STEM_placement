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
) -> Dict:
    """세 지표를 기준선(d=0) 대비 변화량으로 나란히 놓는다."""
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
        })
    return {"baseline_depth": baseline_depth, "rows": rows}


def assess_mechanism(table: Dict, tolerance: float = 1e-9) -> Dict:
    """정합성 판정 - 자동 결론이 아니라 보고서에 실을 관찰.

    반환의 verdict 는 세 가지 중 하나:
      consistent      : recall 이 오르고 divergence 가 줄고 라운드가 절약됨
      other_mechanism : 라운드는 절약되었으나 divergence 는 줄지 않음
      no_effect       : 라운드 절감 자체가 관측되지 않음
    """
    rows = [r for r in table["rows"] if r["depth"] != table["baseline_depth"]]
    saved = [r["rounds_saved"] for r in rows]
    gains = [r["recall_gain"] for r in rows]
    reductions = [r["divergence_reduction"] for r in rows]

    any_saving = any(s is not None and s > tolerance for s in saved)
    any_reduction = any(d is not None and d > tolerance for d in reductions)

    if not any_saving:
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
        "corr_recall_vs_divergence_reduction": pearson(gains, reductions),
        "corr_recall_vs_rounds_saved": pearson(gains, saved),
        "corr_divergence_reduction_vs_rounds_saved": pearson(reductions, saved),
    }
