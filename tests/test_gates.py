import pytest

from src.gates import (
    headroom_decision, local_step_decision, monotonicity_decision,
    rounds_from_budget, summarise_gates, total_steps_from_passes,
)


def test_headroom_fails_when_the_gap_is_too_narrow():
    result = headroom_decision(f1_gold=12.0, f1_random=9.5, min_gap=5.0)
    assert not result.passed
    assert "escalate" in result.action


def test_monotonicity_passes_on_a_rising_curve():
    assert monotonicity_decision({0: 0.41, 3: 0.41, 10: 0.48, 50: 0.53}).passed


def test_monotonicity_fails_on_a_dip():
    result = monotonicity_decision({0: 0.41, 3: 0.39, 10: 0.48})
    assert not result.passed
    assert result.detail["violations"][0]["from_depth"] == 0


def test_flat_recall_is_a_failure_not_a_pass():
    # 평평하면 처리 축이 존재하지 않는 것이므로 통과시켜서는 안 된다
    result = monotonicity_decision({0: 0.4, 3: 0.4, 10: 0.4, 50: 0.4})
    assert not result.passed
    assert "does not exist" in result.action


def test_local_step_gate_picks_the_smallest_K_above_threshold():
    result = local_step_decision({4: 0.004, 8: 0.011, 16: 0.031, 32: 0.07}, 0.02)
    assert result.passed
    assert result.detail["chosen_K"] == 16


def test_local_step_gate_fails_when_no_K_shows_drift():
    result = local_step_decision({4: 0.001, 8: 0.002}, 0.02)
    assert not result.passed
    assert result.detail["chosen_K"] is None


def test_rounds_follow_from_the_budget_not_from_convention():
    assert rounds_from_budget(960, 16) == 60
    assert rounds_from_budget(100, 64) == 1          # 최소 1 라운드는 보장


def test_total_steps_come_from_passes_over_client_data():
    assert total_steps_from_passes(examples_per_client=800, effective_batch=8, passes=1.5) == 150


def test_total_steps_count_the_effective_batch_not_the_minibatch():
    """누적을 쓰면 한 step 이 batch_size x grad_accumulation 개를 소비한다.

    미니배치(2)를 넣으면 S 가 4 배로 부풀고, S 는 R = S / K 로 라운드 수를 정하므로
    통신 예산이 직접 틀어진다. 실측으로 프로브의 S=188 이 "1.5 패스"로 기록됐으나
    실제로는 6 패스였다.
    """
    # 클라이언트 250 예제, 유효 배치 8 (= 2 x 4), 1.5 패스
    assert total_steps_from_passes(250, effective_batch=8, passes=1.5) == 46
    # 미니배치 2 를 잘못 넣으면 4 배가 된다
    assert total_steps_from_passes(250, effective_batch=2, passes=1.5) == 188


def test_steps_per_pass_times_effective_batch_recovers_the_dataset():
    """S x 유효배치 ≈ 예제 수 x 패스 수 여야 한다."""
    for n, eb, passes in ((800, 8, 1.0), (10_090, 8, 1.5), (250, 8, 2.0)):
        S = total_steps_from_passes(n, eb, passes)
        assert abs(S * eb - n * passes) / (n * passes) < 0.02


def test_summary_is_go_only_when_every_gate_passes():
    good = headroom_decision(30.0, 10.0)
    bad = monotonicity_decision({0: 0.5, 10: 0.4})
    assert summarise_gates([good])["go"]
    assert not summarise_gates([good, bad])["go"]


# --- 처리가 일할 수 있는 범위까지 보는 게이트 (Subtask 1.2) ------------------

HEALTHY = {"already_at_top_k": 0.30, "recoverable": 0.35, "out_of_pool": 0.35, "ceiling": 0.65}
RISING = {0: 0.30, 3: 0.30, 10: 0.42, 50: 0.52}


def test_a_rising_curve_with_a_healthy_working_range_passes():
    assert monotonicity_decision(RISING, headroom=HEALTHY).passed


def test_a_rising_curve_with_almost_no_working_range_fails():
    """곡선이 올라도 recoverable 이 작으면 효과가 희석돼 5 시드로는 관측되지 않는다."""
    narrow = {"already_at_top_k": 0.60, "recoverable": 0.02, "out_of_pool": 0.38, "ceiling": 0.62}
    result = monotonicity_decision(RISING, headroom=narrow)
    assert not result.passed
    assert "recoverable" in result.action


def test_the_gate_says_which_knob_to_turn_when_the_corpus_is_too_easy():
    easy = {"already_at_top_k": 0.80, "recoverable": 0.03, "out_of_pool": 0.17, "ceiling": 0.83}
    assert "hard_negative" in monotonicity_decision(RISING, headroom=easy).action


def test_the_gate_says_which_knob_to_turn_when_gold_is_outside_the_pool():
    # 풀 밖에 있는 정답지는 depth 를 올려도 못 닿는다 - 풀을 키워야 한다
    missing = {"already_at_top_k": 0.05, "recoverable": 0.04, "out_of_pool": 0.91, "ceiling": 0.09}
    assert "candidate_pool" in monotonicity_decision(RISING, headroom=missing).action


def test_headroom_is_optional_so_the_monotonicity_check_still_stands_alone():
    assert monotonicity_decision(RISING).passed


# --- local_step_decision: 선언 + 검증 -------------------------------------

DIV_R1 = {4: 0.5097, 8: 0.3895, 16: 0.3376, 32: 0.3343, 64: 0.4334}
SPREAD_R1 = {4: 0.17622, 8: 0.30589, 16: 0.52833, 32: 0.91703, 64: 1.64043}


def test_a_threshold_every_K_clears_is_refused_not_silently_resolved():
    """실측 재현: 모든 K 가 기준의 17~25 배를 넘으면 '최소 K' 규칙은 의미가 없다.

    예전 구현은 이 상황에서 조용히 K=4 를 돌려주고 통과시켰다.
    """
    r = local_step_decision(DIV_R1, min_divergence=0.02, total_local_steps=1892)
    assert r.passed is False
    assert r.detail["criterion_discriminates"] is False
    assert "whatever the data say" in r.action


def test_a_discriminating_threshold_still_picks_the_smallest():
    div = {4: 0.01, 8: 0.05, 16: 0.09}
    r = local_step_decision(div, min_divergence=0.02, total_local_steps=400, min_rounds=10)
    assert r.passed and r.detail["chosen_K"] == 8
    assert r.detail["criterion_discriminates"] is True


def test_a_declared_K_is_verified_and_its_evidence_recorded():
    r = local_step_decision(DIV_R1, 0.02, total_local_steps=1892,
                            chosen_K=32, spread_by_k=SPREAD_R1, min_rounds=20)
    assert r.passed
    assert r.detail["chosen_K"] == 32
    assert r.detail["rounds"] == 59                      # 1892 // 32
    assert r.detail["divergence_at_chosen_K"] == pytest.approx(0.3343)
    assert r.detail["spread_ratio_vs_smallest_K"] == pytest.approx(0.91703 / 0.17622, rel=1e-6)
    assert r.detail["selection"].startswith("declared")


def test_a_declared_K_whose_drift_is_noise_is_rejected():
    """흩어짐이 최소 K 의 2 배도 안 되면, 그 drift 는 작은 업데이트가 만든 잡음이다."""
    r = local_step_decision(DIV_R1, 0.02, total_local_steps=1892,
                            chosen_K=8, spread_by_k=SPREAD_R1, min_rounds=20)
    assert r.passed is False
    assert "not separable from small-update noise" in r.action


def test_a_declared_K_that_leaves_too_few_rounds_is_rejected():
    """R 이 보간 해상도를 못 내면 떨어뜨린다 - 프로브 규모 S 를 넣는 실수를 잡는다."""
    r = local_step_decision(DIV_R1, 0.02, total_local_steps=188,   # 프로브 규모
                            chosen_K=32, spread_by_k=SPREAD_R1, min_rounds=20)
    assert r.passed is False
    assert "below 20" in r.action


def test_a_declared_K_outside_the_sweep_is_rejected():
    r = local_step_decision(DIV_R1, 0.02, total_local_steps=1892,
                            chosen_K=128, spread_by_k=SPREAD_R1)
    assert r.passed is False
    assert "not in the measured sweep" in r.action


def test_no_K_reaching_the_threshold_still_fails_loudly():
    r = local_step_decision({4: 0.001, 8: 0.002}, min_divergence=0.02)
    assert r.passed is False and r.detail["chosen_K"] is None
    assert "extend the K sweep upward" in r.action
