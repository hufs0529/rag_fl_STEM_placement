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


def test_the_lower_bound_is_round_1_not_round_0():
    """곡선은 라운드 0->1 에 거의 수직으로 오른다(실측 0.5 -> 18). 라운드 0 을
    하한으로 쓰면 그 구간의 tau 가 "라운드 0.54 에 도달" 을 내는데, 라운드는
    서버가 어댑터를 모으는 최소 단위이므로 실행할 수 없는 값이다."""
    steep = [{"round": 0, "f1": 0.5}] + CURVE
    assert attainable_range([steep])[0] == 10.0        # 라운드 1 의 값
    assert rounds_to_target(steep, 5.0) < 1.0          # 하한 아래는 라운드 1 미만


def test_the_lower_bound_takes_the_max_so_no_seed_lands_inside_round_1():
    a = [{"round": 1, "f1": 10.0}, {"round": 2, "f1": 30.0}]
    b = [{"round": 1, "f1": 14.0}, {"round": 2, "f1": 30.0}]
    assert attainable_range([a, b])[0] == 14.0


def test_the_upper_bound_takes_the_min_so_no_seed_drops_out():
    """상한 위에서는 최고값이 낮은 시드부터 탈락하고, 남는 것은 원래 잘한
    시드다 - 평균이 그만큼 낙관적으로 치우친다."""
    weak = [{"round": 1, "f1": 10.0}, {"round": 2, "f1": 20.0}]
    strong = [{"round": 1, "f1": 10.0}, {"round": 2, "f1": 40.0}]
    assert attainable_range([weak, strong])[1] == 20.0
    assert rounds_to_target(weak, 30.0) is None


def test_the_tau_grid_comes_from_the_curves_not_from_a_declared_spacing():
    """격자 간격을 사람이 정하면 "왜 그 간격인가" 에 답할 수 없고, 보간은
    평가점 사이에서 직선이므로 한 구간 안의 여러 tau 는 독립 정보가 아니다."""
    grid = tau_grid([CURVE])
    assert grid == [10.0, 20.0, 30.0, 32.0]            # 곡선이 실제로 낸 값
    assert tau_grid([CURVE], points=5) == grid         # points 는 무시된다


def test_the_tau_grid_excludes_round_0_values():
    steep = [{"round": 0, "f1": 0.5}] + CURVE
    assert 0.5 not in tau_grid([steep])


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
