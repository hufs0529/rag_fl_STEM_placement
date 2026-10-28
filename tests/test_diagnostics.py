import pytest

from src.analysis.diagnostics import (
    assess_mechanism, mean_divergence, mechanism_table, sweep_summary,
)


def _table(divergence_at_50):
    return mechanism_table(
        recall_by_depth={0: 0.40, 10: 0.48, 50: 0.53},
        divergence_by_depth={0: 0.090, 10: 0.078, 50: divergence_at_50},
        rounds_saved_by_depth={0: 0.0, 10: 5.0, 50: 9.0},
    )


def test_changes_are_reported_relative_to_the_baseline_depth():
    rows = {r["depth"]: r for r in _table(0.061)["rows"]}
    assert rows[50]["recall_gain"] == 0.13
    assert abs(rows[50]["divergence_reduction"] - 0.029) < 1e-9
    assert rows[0]["recall_gain"] == 0.0


def test_a_saving_with_less_drift_is_consistent_with_the_proposed_path():
    assert assess_mechanism(_table(0.061))["verdict"] == "consistent"


def test_a_saving_without_less_drift_is_flagged_as_another_mechanism():
    # 계획서: "a gain without one would indicate a different mechanism"
    table = mechanism_table(
        recall_by_depth={0: 0.40, 50: 0.53},
        divergence_by_depth={0: 0.090, 50: 0.095},
        rounds_saved_by_depth={0: 0.0, 50: 9.0},
    )
    assert assess_mechanism(table)["verdict"] == "other_mechanism"


def test_a_saving_smaller_than_its_own_uncertainty_is_not_a_saving():
    """실측에서 한 조건의 +0.0988 라운드(표준오차 0.515)가, 나머지가 전부 음수인데도
    판정을 no_effect 에서 consistent 로 뒤집었다. 보고서에 "가설과 정합" 으로
    실릴 뻔했다."""
    table = mechanism_table(
        recall_by_depth={0: 0.40, 25: 0.71, 50: 0.72},
        divergence_by_depth={0: 0.090, 25: 0.083, 50: 0.080},
        rounds_saved_by_depth={0: 0.0, 25: 0.0988, 50: -0.1465},
        rounds_saved_stderr_by_depth={0: 0.175, 25: 0.515, 50: 0.858},
    )
    out = assess_mechanism(table)
    assert out["verdict"] == "within_noise"
    assert out["depths_with_resolved_saving"] == []
    assert "0.099" in out["note"] and "0.515" in out["note"]


def test_a_saving_larger_than_its_uncertainty_still_counts():
    table = mechanism_table(
        recall_by_depth={0: 0.40, 50: 0.53},
        divergence_by_depth={0: 0.090, 50: 0.061},
        rounds_saved_by_depth={0: 0.0, 50: 9.0},
        rounds_saved_stderr_by_depth={0: 0.2, 50: 0.5},
    )
    out = assess_mechanism(table)
    assert out["verdict"] == "consistent"
    assert out["depths_with_resolved_saving"] == [50]


def test_without_a_stderr_the_sign_alone_decides():
    """표준오차를 안 넘기는 호출(예전 서명)도 그대로 동작해야 한다."""
    table = mechanism_table(
        recall_by_depth={0: 0.40, 50: 0.53},
        divergence_by_depth={0: 0.090, 50: 0.061},
        rounds_saved_by_depth={0: 0.0, 50: 0.3},
    )
    assert assess_mechanism(table)["verdict"] == "consistent"


def test_no_saving_means_there_is_no_mechanism_to_explain():
    table = mechanism_table(
        recall_by_depth={0: 0.40, 50: 0.53},
        divergence_by_depth={0: 0.090, 50: 0.061},
        rounds_saved_by_depth={0: 0.0, 50: 0.0},
    )
    assert assess_mechanism(table)["verdict"] == "no_effect"


def test_divergence_average_ignores_round_zero_and_unevaluated_fields():
    records = [
        {"event": "meta"},
        {"event": "round", "round": 0},
        {"event": "round", "round": 1, "drift_divergence": 0.10},
        {"event": "round", "round": 2, "drift_divergence": 0.20},
    ]
    assert mean_divergence(records) == pytest.approx(0.15)


def test_early_rounds_can_be_isolated():
    # drift 가 효과를 만든다면 어댑터가 아직 움직이는 초반에 나타나야 한다
    records = [
        {"event": "round", "round": r, "drift_divergence": d}
        for r, d in [(1, 0.2), (2, 0.2), (30, 0.01)]
    ]
    assert mean_divergence(records, first_n_rounds=10) == 0.2


def _sweep(saved_by_depth_per_tau, stderr=0.5, seeds=4):
    """tau 격자를 흉내낸다: {depth: [tau 마다의 절약]}.

    stderr 는 **차이 자신의** 표준오차이고, 분해능 문턱은 t(0.975, seeds-1) 를
    곱한 값이다. seeds=4 면 3.182 x 0.5 = 1.591 이므로, 아래 테스트들의 3.0 은
    문턱을 넘고 2.0 도 넘지만 1.0 은 넘지 못한다. 예전 픽스처는 1 시그마로 비교해
    0.6 같은 값도 "분해됨" 으로 셌다.
    """
    n = len(next(iter(saved_by_depth_per_tau.values())))
    return [
        {"tau": 20.0 + i,
         "depths": {str(d): {"rounds_saved": vals[i],
                             "rounds_stderr": stderr,
                             "rounds_saved_stderr": stderr,
                             "seeds_reaching_tau": seeds}
                    for d, vals in saved_by_depth_per_tau.items()}}
        for i in range(n)
    ]


def test_the_sweep_summary_counts_targets_rather_than_trusting_one():
    """tau 한 점은 그 점의 선택이 결론을 정한다. 실측에서 tau=25.02 에서는 d=25 가
    유일한 양수(+0.10)였지만 63 점 전체 중앙값은 -0.113 이었다."""
    out = sweep_summary(_sweep({25: [2.0, -1.0, -1.0, -1.0]}))
    s = out[25]
    assert s["n_tau"] == 4
    assert s["tau_with_saving"] == 1
    assert s["tau_with_resolved_saving"] == 1
    assert s["median_rounds_saved"] == -1.0
    assert (s["min_rounds_saved"], s["max_rounds_saved"]) == (-1.0, 2.0)


def test_a_saving_at_a_minority_of_targets_is_within_noise():
    sweep = _sweep({0: [0.0] * 5, 50: [3.0, 3.0, -1.0, -1.0, -1.0]})
    summary = sweep_summary(sweep)
    table = mechanism_table(
        recall_by_depth={0: 0.40, 50: 0.53},
        divergence_by_depth={0: 0.090, 50: 0.061},
        rounds_saved_by_depth={0: 0.0, 50: summary[50]["median_rounds_saved"]},
        sweep_by_depth=summary,
    )
    out = assess_mechanism(table)
    assert out["verdict"] == "within_noise"
    assert "2 of 5" in out["note"]


def test_a_saving_at_a_majority_of_targets_counts():
    sweep = _sweep({0: [0.0] * 5, 50: [3.0, 3.0, 3.0, 3.0, -1.0]})
    summary = sweep_summary(sweep)
    table = mechanism_table(
        recall_by_depth={0: 0.40, 50: 0.53},
        divergence_by_depth={0: 0.090, 50: 0.061},
        rounds_saved_by_depth={0: 0.0, 50: summary[50]["median_rounds_saved"]},
        sweep_by_depth=summary,
    )
    assert assess_mechanism(table)["verdict"] == "consistent"
