import pytest

from src.analysis.curves import (
    attainable_range, curve_mean, final_scores, mean_and_stderr,
    rounds_by_depth, rounds_to_target, tau_grid,
)

CURVE = [{"round": r, "f1": v} for r, v in [(1, 10.0), (2, 20.0), (4, 30.0), (8, 32.0)]]


def test_target_is_read_by_linear_interpolation_not_by_the_nearest_round():
    # 라운드 수는 이 규모에서 너무 거친 자다 - 보간이 그 문제를 푼다
    assert rounds_to_target(CURVE, 25.0) == 3.0
    assert rounds_to_target(CURVE, 15.0) == 1.5


def test_target_already_met_at_the_first_evaluation_returns_that_round():
    assert rounds_to_target(CURVE, 5.0) == 1.0


def test_a_target_never_reached_is_none_rather_than_extrapolated():
    # 외삽하면 도달하지 못한 조건에 유리한 숫자를 지어내게 된다
    assert rounds_to_target(CURVE, 99.0) is None


def test_tau_is_swept_over_the_range_the_baseline_attains():
    low, high = attainable_range([CURVE])
    assert (low, high) == (10.0, 32.0)
    grid = tau_grid([CURVE], points=5)
    assert len(grid) == 5 and grid[0] == 10.0 and grid[-1] == 32.0


def test_seeds_are_summarised_as_mean_and_standard_error():
    stats = mean_and_stderr([10.0, 12.0, 14.0])
    assert stats["mean"] == 12.0
    assert stats["stderr"] == pytest.approx(1.1547, abs=1e-4)
    assert stats["n"] == 3


def test_seeds_that_never_reach_tau_are_counted_not_dropped_silently():
    stats = mean_and_stderr([10.0, None, 14.0])
    assert stats["n"] == 2
    assert stats["missing"] == 1


def test_a_single_seed_reports_no_standard_error():
    assert mean_and_stderr([7.0])["stderr"] is None


def test_rounds_are_interpolated_per_seed_before_averaging():
    # 곡선을 먼저 평균내면 시드 간 분산이 사라져 구간을 보고할 수 없다
    fast = [{"round": r, "f1": v} for r, v in [(1, 10.0), (2, 30.0)]]
    slow = [{"round": r, "f1": v} for r, v in [(1, 10.0), (2, 15.0), (4, 30.0)]]
    out = rounds_by_depth({10: [fast, slow]}, tau=20.0)
    assert out[10]["per_seed"] == pytest.approx([1.5, 2.0 + 2 / 3])
    assert out[10]["mean"] == pytest.approx((1.5 + 2.0 + 2 / 3) / 2)
    assert out[10]["stderr"] is not None


def test_mean_curve_keeps_the_round_schedule():
    out = curve_mean([CURVE, CURVE])
    assert [p["round"] for p in out] == [1, 2, 4, 8]
    assert out[0]["mean"] == 10.0


def test_final_score_is_what_the_fixed_budget_bought():
    assert final_scores([CURVE])["mean"] == 32.0
