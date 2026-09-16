import numpy as np
import pytest

from src.divergence import client_deltas, divergence, mean_pairwise_cosine, summarise


def test_identical_updates_have_zero_divergence():
    deltas = [np.ones(16) for _ in range(8)]
    assert divergence(deltas) == 0.0
    assert mean_pairwise_cosine(deltas) > 0.999


def test_opposing_updates_diverge():
    deltas = [np.ones(16), -np.ones(16), np.ones(16), -np.ones(16)]
    assert divergence(deltas) > 1.0
    assert mean_pairwise_cosine(deltas) < 0.0


def test_divergence_is_scale_invariant_so_K_values_are_comparable():
    rng = np.random.default_rng(0)
    deltas = [rng.normal(size=32) for _ in range(8)]
    assert divergence(deltas) == pytest.approx(divergence([10 * d for d in deltas]))


def test_client_deltas_subtract_the_global_parameters():
    global_params = {"a": np.zeros(4), "b": np.ones(2)}
    clients = [{"a": np.ones(4), "b": np.ones(2)}, {"a": np.zeros(4), "b": np.zeros(2)}]
    deltas = client_deltas(global_params, clients)
    assert np.allclose(deltas[0], [1, 1, 1, 1, 0, 0])
    assert np.allclose(deltas[1], [0, 0, 0, 0, -1, -1])


def test_single_client_reports_no_divergence():
    assert summarise([np.ones(4)])["divergence"] == 0.0


def test_absolute_spread_is_the_numerator_of_divergence():
    import numpy as np
    from src.divergence import absolute_spread, divergence
    deltas = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    mean_norm = float(np.linalg.norm(np.stack(deltas).mean(axis=0)))
    assert absolute_spread(deltas) == pytest.approx(divergence(deltas) * mean_norm, rel=1e-9)


def test_the_two_measures_move_oppositely_as_updates_grow():
    """K 가 커지는 상황을 모사: 업데이트가 커지면서 흩어짐도 비례해 커진다.

    절대 흩어짐은 늘고 정규화된 divergence 는 그대로거나 줄어든다 - 게이트 (iii) 의
    선택 규칙이 어느 쪽을 보는지에 따라 답이 달라지는 이유다.
    """
    import numpy as np
    from src.divergence import absolute_spread, divergence
    small = [np.array([1.0, 0.0]), np.array([0.9, 0.2])]
    # 업데이트 크기 10 배, 흩어짐은 10 배보다 덜 증가
    large = [np.array([10.0, 0.0]), np.array([9.5, 1.0])]
    assert absolute_spread(large) > absolute_spread(small)
    assert divergence(large) <= divergence(small) * 1.01


def test_absolute_spread_needs_two_clients():
    import numpy as np
    from src.divergence import absolute_spread
    assert absolute_spread([]) == 0.0
    assert absolute_spread([np.array([1.0, 2.0])]) == 0.0
