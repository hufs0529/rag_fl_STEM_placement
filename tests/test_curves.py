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


# --- 분해능 (Week 5 보강) ---------------------------------------------------

def test_the_critical_value_for_four_seeds_is_not_two():
    """시드 4 개는 자유도 3 이고 양측 95 % 임계값이 3.182 다.

    보고서가 "표준오차의 2 배" 를 쓰고 있었는데, 그건 느슨한 쪽으로 1.6 배
    틀린 값이다. 그 규칙에서는 d=10 의 +0.60 +- 0.28 이 "유의" 로 읽혔지만
    올바른 검정에서는 p = 0.127 이다.
    """
    from src.analysis.curves import t_critical_95
    assert t_critical_95(3) == pytest.approx(3.182, abs=1e-3)
    assert t_critical_95(3) > 2.0
    # df 가 커지면 1.96 으로 수렴한다 - 큰 표본에서만 2 배 규칙이 맞다
    assert t_critical_95(30) < t_critical_95(10) < t_critical_95(3)
    assert t_critical_95(0) == float("inf")      # 자유도가 없으면 판정 불가


def test_four_seeds_cannot_reach_p_below_five_percent():
    """설계의 구조적 한계. 이 한 줄이 within_noise 를 변명이 아닌 사실로 만든다.

    대응 치환검정은 n 개 차이의 부호를 뒤집는 것뿐이라 배열이 2^n 가지다.
    n=4 면 16 가지이고 최소 양측 p = 0.125 - 데이터가 아무리 깨끗해도
    p<0.05 가 **구조적으로 불가능**하다.
    """
    from src.analysis.curves import permutation_floor_p
    assert permutation_floor_p(4) == pytest.approx(0.125)
    assert permutation_floor_p(4) > 0.05          # 4 개로는 불가능
    assert permutation_floor_p(6) == pytest.approx(0.03125)
    assert permutation_floor_p(6) < 0.05          # 6 개부터 가능
    assert permutation_floor_p(0) is None


def test_pairing_by_seed_is_tighter_than_comparing_the_two_means():
    """시드 효과가 상쇄되므로 대응이 비대응보다 좁다.

    실측으로 비대응 차이의 표준오차는 0.879 라운드인데 대응은 0.16~0.47 이다.
    같은 데이터에서 분해능이 2~5 배 좋아진다 - 공짜로 얻는 검정력이므로
    대응을 쓰지 않을 이유가 없다.
    """
    import math
    from src.analysis.curves import mean_and_stderr, paired_saving
    # 시드마다 라운드 수가 통째로 올라가지만 차이는 늘 +0.5 다
    base = [10.0, 20.0, 30.0, 40.0]
    depth = [9.5, 19.5, 29.5, 39.5]
    pair = paired_saving(base, depth)
    assert pair["mean"] == pytest.approx(0.5)
    assert pair["stderr"] == pytest.approx(0.0)       # 시드 성분이 완전히 상쇄
    # 비대응으로 보면 산포가 그대로 남아 아무것도 분해하지 못한다
    unpaired = math.hypot(mean_and_stderr(base)["stderr"], mean_and_stderr(depth)["stderr"])
    assert unpaired > 5.0


def test_a_saving_must_clear_the_t_threshold_not_just_its_standard_error():
    from src.analysis.curves import paired_saving
    # 차이 0.65 +- 0.27 -> 문턱 3.182 x 0.27 = 0.85. 1 시그마 규칙이면 통과했다.
    base, depth = [10.0, 10.0, 10.0, 10.0], [9.9, 8.8, 9.7, 9.0]
    pair = paired_saving(base, depth)
    assert pair["mean"] == pytest.approx(0.65, abs=0.01)
    assert pair["mean"] > pair["stderr"]             # 옛 규칙으로는 "분해됨"
    assert pair["threshold"] == pytest.approx(pair["t_critical"] * pair["stderr"])
    assert pair["resolved"] is False                 # 올바른 규칙으로는 아니다


def test_pairing_drops_seeds_that_only_one_arm_reached():
    from src.analysis.curves import paired_saving
    pair = paired_saving([10.0, None, 30.0], [9.0, 19.0, None])
    assert pair["n"] == 1                            # 첫 시드만 양쪽에 있다
    assert pair["stderr"] is None                    # n=1 이면 산포를 못 잰다
    assert pair["resolved"] is False
