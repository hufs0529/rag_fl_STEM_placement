import pytest

from src.analysis.diagnostics import assess_mechanism, mean_divergence, mechanism_table


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
