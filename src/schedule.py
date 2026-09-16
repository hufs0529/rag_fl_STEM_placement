"""곡선 평가 라운드 스케줄 (Subtask 1.3).

계획서: "Curves are evaluated on a fixed round schedule, dense early and sparse
late." 초반을 촘촘히 보는 이유는 LoRA 어댑터가 몇 라운드 만에 수렴하므로
처리 효과가 드러나는 구간이 앞쪽이기 때문이고, 후반을 성기게 보는 이유는
평가가 비싸기 때문이다(고정 1,000 질문 x 탐욕 디코딩).

스케줄은 **모든 조건에서 동일**해야 한다. 조건마다 평가 지점이 다르면
Subtask 1.4 의 보간(interpolation)이 조건 간 비교 가능성을 잃는다.

The fixed evaluation schedule, dense early and sparse late, identical across
every condition so the Week-5 interpolation stays comparable.
"""

from typing import List


def eval_rounds(total_rounds: int, dense_until: int = 10, sparse_every: int = 5) -> List[int]:
    """평가할 라운드 번호 목록 (1-indexed, 마지막 라운드는 항상 포함).

    round 0 은 학습 전 기저값으로 따로 기록한다 (baseline_round_0).
    """
    if total_rounds < 1:
        return []
    rounds = list(range(1, min(dense_until, total_rounds) + 1))
    rounds += [
        r for r in range(dense_until + sparse_every, total_rounds + 1, sparse_every)
        if r > dense_until
    ]
    if total_rounds not in rounds:
        rounds.append(total_rounds)
    return sorted(set(rounds))


def is_eval_round(round_number: int, total_rounds: int, dense_until: int, sparse_every: int) -> bool:
    return round_number in eval_rounds(total_rounds, dense_until, sparse_every)


def schedule_from_config(cfg, total_rounds: int) -> List[int]:
    return eval_rounds(
        total_rounds,
        cfg.eval.round_schedule_dense_until,
        cfg.eval.round_schedule_sparse_every,
    )
