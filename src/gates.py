"""Week 1 파일럿 게이트 3종의 판정 로직 (Subtask 1.2).

게이트는 "본실험 전에 남은 자유 파라미터를 고정"하기 위한 값싼 사전 점검이다.
각 게이트는 통과 조건과 **실패 시 행동**이 미리 정해져 있어야 한다.

  (i)  capability headroom  - 학습 불필요. gold vs random 컨텍스트의 F1 격차가
       어떤 retrieval 개선이든 가져다 줄 수 있는 상한을 준다. 격차 < ~5 F1 이면
       더 큰 체크포인트로 올린다.
  (ii) depth monotonicity   - retrieval 만 필요. gold recall@3 이 d 에 대해
       단조 증가해야 한다. 아니면 처리 축 자체가 존재하지 않는 것이므로 depth 를 수정.
  (iii) local-step calibration - K 를 스윕해 client-update divergence 를 재고,
       drift 가 측정 가능한 최소 K 를 고른다. drift 가 없는 곳에서는
       drift 를 줄이는 처리를 관측할 수 없기 때문이다. S 는 클라이언트 데이터
       1~2 epoch 으로 두고 R = S / K 가 따라 나온다.

Decision logic for the three Week-1 pilot gates. Each has a pass condition and
a predefined failure action, and together they fix the remaining free
parameters before any full run.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence


@dataclass
class GateResult:
    name: str
    passed: bool
    action: str
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] gate {self.name}: {self.action}"


# --- gate (i) capability headroom ------------------------------------------

def headroom_decision(
    f1_gold: float,
    f1_random: float,
    min_gap: float = 5.0,
    fallback_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
) -> GateResult:
    """gold 컨텍스트와 random 컨텍스트의 F1 격차가 처리 효과의 상한이다.

    격차가 좁으면 retrieval 을 아무리 잘해도 옮길 수 있는 정확도가 없으므로,
    실험을 돌리기 전에 더 큰 체크포인트로 올린다.
    """
    gap = f1_gold - f1_random
    passed = gap >= min_gap
    action = (
        f"proceed with the planned checkpoint (headroom {gap:.2f} F1 >= {min_gap})"
        if passed
        else f"headroom {gap:.2f} F1 < {min_gap}: escalate to {fallback_model}"
    )
    return GateResult(
        name="capability_headroom",
        passed=passed,
        action=action,
        detail={"f1_gold": f1_gold, "f1_random": f1_random, "gap": gap, "min_gap": min_gap},
    )


# --- gate (ii) depth monotonicity ------------------------------------------

def monotonicity_decision(
    recall_by_depth: Dict[int, float],
    tolerance: float = 0.0,
    headroom: Optional[dict] = None,
    min_recoverable: float = 0.10,
) -> GateResult:
    """gold recall@3 이 depth 에 대해 단조 증가하는지.

    단조성이 깨지면 "depth 를 늘리면 검색 품질이 오른다"는 처리 축이 존재하지
    않는 것이므로, 조건 d 집합을 다시 잡는다. 또한 전혀 증가하지 않으면
    (평평하면) 처리가 걸리지 않은 것이라 역시 실패로 본다.
    """
    depths = sorted(recall_by_depth)
    values = [recall_by_depth[d] for d in depths]
    violations = [
        {"from_depth": depths[i], "to_depth": depths[i + 1],
         "drop": values[i] - values[i + 1]}
        for i in range(len(values) - 1)
        if values[i + 1] < values[i] - tolerance
    ]
    total_gain = values[-1] - values[0] if values else 0.0
    passed = not violations and total_gain > tolerance
    if violations:
        action = "recall@3 is not monotone in depth: revise the depth set"
    elif total_gain <= tolerance:
        action = "recall@3 is flat in depth: the treatment axis does not exist; revise depths"
    else:
        action = f"treatment axis confirmed (recall@3 gain {total_gain:.4f} from d={depths[0]} to d={depths[-1]})"

    # 단조성만으로는 부족하다. 처리가 **일할 수 있는 질문의 비율**(recoverable)이
    # 너무 작으면, 곡선이 오르더라도 효과가 희석되어 5 시드로는 관측되지 않는다.
    # 어느 손잡이를 돌려야 하는지도 이 분해가 알려준다.
    if passed and headroom is not None:
        recoverable = headroom.get("recoverable", 0.0)
        if recoverable < min_recoverable:
            passed = False
            if headroom.get("out_of_pool", 0.0) > headroom.get("already_at_top_k", 0.0):
                knob = ("most gold sits OUTSIDE the candidate pool: raise "
                        "retrieval.candidate_pool (more depth cannot reach it)")
            else:
                knob = ("most gold is ALREADY in the dense top-k: the corpus is too easy, "
                        "raise data.hard_negative_questions")
            action = (f"only {recoverable:.1%} of questions are recoverable by reranking "
                      f"(need >= {min_recoverable:.0%}); {knob}")

    return GateResult(
        name="depth_monotonicity",
        passed=passed,
        action=action,
        detail={
            "recall_by_depth": {str(d): recall_by_depth[d] for d in depths},
            "total_gain": total_gain,
            "violations": violations,
            "headroom": headroom,
            "min_recoverable": min_recoverable,
        },
    )


# --- gate (iii) local-step calibration -------------------------------------

def local_step_decision(
    divergence_by_k: Dict[int, float],
    min_divergence: float = 0.02,
    total_local_steps: Optional[int] = None,
    chosen_K: Optional[int] = None,
    spread_by_k: Optional[Dict[int, float]] = None,
    min_rounds: int = 20,
) -> GateResult:
    """K 를 **검증**한다. 선언된 K 가 없으면 고르려 시도하되, 기준이 판별력을
    잃은 경우에는 조용히 최솟값을 돌려주지 않고 실패한다.

    원래 규칙은 "drift 가 기준을 넘는 **최소** K" 였다. 실측에서 이 규칙이
    퇴화했다: 다섯 K 의 divergence 가 0.33~0.51 로 기준 0.02 의 17~25 배였고,
    따라서 규칙은 데이터와 무관하게 **항상 후보 중 최솟값**을 반환했다. 게이트가
    아무것도 고르지 않은 셈인데 출력은 통과였다 - 가장 나쁜 조합이다.

    더 깊은 원인은 정규화다. divergence = 흩어짐 / ||평균 업데이트|| 인데, K 가
    커지면 분모가 분자보다 빨리 커져 **정규화된 값이 감소**한다. 반면 흩어짐
    자체는 단조 증가한다. 한 기준으로 "가장 작은 K" 를 묻는 질문이 성립하지 않는다.

    그래서 선택을 측정으로 위장하지 않고 **선언 + 검증**으로 분리했다.
    chosen_K 를 주면 그 K 에서
        (a) drift 가 기준을 넘는가,
        (b) 흩어짐이 최소 K 대비 유의하게 큰가,
        (c) R = S/K 가 보간에 필요한 min_rounds 이상인가
    를 확인하고 근거를 함께 기록한다. 통과 여부는 측정이 결정하고, 값은 사람이
    고른다 - 어느 쪽인지 산출물에 남는다.
    """
    ks = sorted(divergence_by_k)
    eligible = [k for k in ks if divergence_by_k[k] >= min_divergence]
    discriminates = 0 < len(eligible) < len(ks)
    detail = {
        "divergence_by_k": {str(k): divergence_by_k[k] for k in ks},
        "min_divergence": min_divergence,
        "eligible_K": eligible,
        # 기준이 후보를 실제로 걸러냈는가. False 면 "최소 K" 규칙은 의미가 없다.
        "criterion_discriminates": discriminates,
        "min_rounds": min_rounds,
    }
    if spread_by_k:
        detail["absolute_spread_by_k"] = {str(k): spread_by_k[k] for k in sorted(spread_by_k)}

    if not eligible:
        detail["chosen_K"] = None
        return GateResult(
            name="local_step_calibration", passed=False,
            action=(f"no K in {ks} reaches divergence {min_divergence}: "
                    "extend the K sweep upward or tighten the partition before proceeding"),
            detail=detail,
        )

    if chosen_K is None:
        if not discriminates:
            detail["chosen_K"] = None
            return GateResult(
                name="local_step_calibration", passed=False,
                action=(f"every K in {ks} clears divergence {min_divergence} "
                        f"(measured {min(divergence_by_k.values()):.4f}-"
                        f"{max(divergence_by_k.values()):.4f}), so 'smallest K above the "
                        "threshold' returns the smallest candidate whatever the data say. "
                        "Declare gates.local_steps.chosen_K from the recorded curves and "
                        "re-run so the choice is verified rather than fabricated"),
                detail=detail,
            )
        chosen_K = eligible[0]
        detail["selection"] = "smallest K above the threshold"
    else:
        detail["selection"] = "declared, then verified against the measured curves"

    detail["chosen_K"] = chosen_K
    failures = []
    if chosen_K not in divergence_by_k:
        failures.append(f"K={chosen_K} is not in the measured sweep {ks}")
    else:
        detail["divergence_at_chosen_K"] = divergence_by_k[chosen_K]
        if divergence_by_k[chosen_K] < min_divergence:
            failures.append(
                f"divergence at K={chosen_K} is {divergence_by_k[chosen_K]:.4f}, "
                f"below {min_divergence}")
        if spread_by_k and chosen_K in spread_by_k:
            ratio = spread_by_k[chosen_K] / max(1e-12, spread_by_k[ks[0]])
            detail["spread_at_chosen_K"] = spread_by_k[chosen_K]
            # 흩어짐이 최소 K 와 비슷하면, 그 K 의 drift 는 업데이트가 작아서 생긴
            # 상대적 잡음과 구별되지 않는다.
            detail["spread_ratio_vs_smallest_K"] = ratio
            if ratio < 2.0:
                failures.append(
                    f"absolute spread at K={chosen_K} is only {ratio:.2f}x that at "
                    f"K={ks[0]}; drift there is not separable from small-update noise")

    rounds = None
    if total_local_steps:
        rounds = rounds_from_budget(total_local_steps, chosen_K)
        detail.update({"total_local_steps": total_local_steps, "rounds": rounds})
        if rounds < min_rounds:
            failures.append(
                f"R = S/K = {rounds} is below {min_rounds}; rounds-to-target could not be "
                "interpolated at better than "
                f"{100.0 / max(1, rounds):.0f}% of the budget")

    if failures:
        detail["failures"] = failures
        return GateResult(name="local_step_calibration", passed=False,
                          action="; ".join(failures), detail=detail)

    if rounds is not None:
        action = (f"K = {chosen_K} ({detail['selection']}), S = {total_local_steps}, "
                  f"therefore R = S/K = {rounds}")
    else:
        action = f"K = {chosen_K} ({detail['selection']}); set S to fix R = S/K"
    return GateResult(name="local_step_calibration", passed=True, action=action, detail=detail)


def rounds_from_budget(total_local_steps: int, local_steps_per_round: int) -> int:
    """R = S / K. 라운드 수는 자유 파라미터가 아니라 이 나눗셈의 결과다.

    The number of rounds is not a free parameter: it is total local steps
    divided by local steps per round.
    """
    if local_steps_per_round <= 0:
        raise ValueError("local_steps_per_round must be positive")
    return max(1, total_local_steps // local_steps_per_round)


def total_steps_from_passes(
    examples_per_client: int, effective_batch: int, passes: float
) -> int:
    """S = 클라이언트 데이터 passes 회 통과에 필요한 local step 수.

    계획서: "Total local steps S is set to 1-2 passes over each client's data".

    **effective_batch 는 한 step 이 소비하는 예제 수**이며, 누적을 쓰면
    `train.batch_size x train.grad_accumulation` 이다. 미니배치 크기를 그대로
    넣으면 S 가 누적 횟수만큼 과대평가된다 (batch 2 / 누적 4 에서 4 배). S 는
    R = S / K 를 통해 라운드 수를 정하므로, 이 혼동은 통신 예산을 직접 틀리게 만든다.
    """
    per_step = max(1, effective_batch)
    steps_per_pass = max(1, examples_per_client // per_step)
    return max(1, int(round(steps_per_pass * passes)))


def summarise_gates(results: Sequence[GateResult]) -> dict:
    """세 게이트를 합쳐 go/no-go 한 줄로."""
    go = all(r.passed for r in results)
    return {
        "go": go,
        "verdict": "GO - proceed to the full experiment" if go
        else "NO-GO - apply the stated failure action(s) before proceeding",
        "gates": [r.to_dict() for r in results],
    }
