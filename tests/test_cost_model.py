import pytest

from src.analysis.cost_model import (
    break_even_bandwidth_mbps, break_even_surface, depth_cost, exchange_rate,
    indexing_flops, net_benefit_seconds, reranking_flops, saved_bytes,
    total_bytes, transfer_seconds,
)

PAYLOAD = 1_840_000
CLIENTS = 8


def test_total_bytes_follow_the_plan_formula():
    assert total_bytes(60, CLIENTS, PAYLOAD) == 60 * 8 * 1_840_000 * 2


def test_saved_bytes_convert_a_round_difference():
    saved = saved_bytes(40.0, 28.0, CLIENTS, PAYLOAD)
    assert saved == 12 * CLIENTS * PAYLOAD * 2


def test_no_saving_is_computed_when_a_condition_never_reached_tau():
    assert saved_bytes(40.0, None, CLIENTS, PAYLOAD) is None
    assert saved_bytes(None, 28.0, CLIENTS, PAYLOAD) is None


def test_a_slower_condition_yields_a_negative_saving_rather_than_zero():
    # 처리가 손해였다면 그 사실이 숫자에 남아야 한다
    assert saved_bytes(28.0, 40.0, CLIENTS, PAYLOAD) < 0


def test_indexing_is_one_off_and_reranking_is_per_query():
    assert indexing_flops(200_000, 1e9) == 2e14
    assert reranking_flops(1e5, 50, 1e10) == 1e5 * 50 * 1e10
    # depth 0 은 반복 비용이 없다
    assert reranking_flops(1e5, 0, 1e10) == 0


def test_the_two_terms_are_reported_separately():
    cost = depth_cost(10, 1e4, 200_000, 1e10, 1e9, amortise_indexing=False)
    assert cost["indexing_flops"] == 2e14
    assert cost["reranking_flops"] == 1e4 * 10 * 1e10
    assert cost["total_flops"] == cost["indexing_flops"] + cost["reranking_flops"]


def test_amortising_an_existing_index_drops_the_one_off_term():
    # RAG 를 이미 배포 중이면 인덱스는 이미 있다 - 두 가정을 모두 보고한다
    amortised = depth_cost(10, 1e4, 200_000, 1e10, 1e9, amortise_indexing=True)
    assert amortised["indexing_flops"] == 0.0


def test_exchange_rate_is_bytes_saved_per_extra_flop():
    assert exchange_rate(1e9, 1e12) == pytest.approx(1e-3)
    assert exchange_rate(None, 1e12) is None
    assert exchange_rate(1e9, 0) is None


def test_reranking_pays_on_slow_links_and_loses_on_fast_ones():
    saved, flops, throughput = 1e9, 1e12, 1e12
    slow = net_benefit_seconds(saved, flops, bandwidth_mbps=1, device_throughput=throughput)
    fast = net_benefit_seconds(saved, flops, bandwidth_mbps=10_000, device_throughput=throughput)
    assert slow > 0 > fast


def test_break_even_bandwidth_is_where_the_two_costs_meet():
    saved, flops, throughput = 1e9, 1e12, 1e12
    b = break_even_bandwidth_mbps(saved, flops, throughput)
    assert net_benefit_seconds(saved, flops, b, throughput) == pytest.approx(0.0, abs=1e-9)


def test_break_even_bandwidth_falls_as_query_volume_grows():
    # 쿼리가 많을수록 반복 비용이 쌓여 reranking 이 불리해진다
    contour = break_even_surface(
        1e9, 50, [1, 10], [1e3, 1e6], 200_000, 1e10, 1e9, 1e12
    )["break_even_contour"]
    assert contour[0]["break_even_bandwidth_mbps"] > contour[1]["break_even_bandwidth_mbps"]


def test_the_surface_reports_a_regime_not_a_single_verdict():
    surface = break_even_surface(1e9, 50, [1, 10, 100], [1e3, 1e5], 200_000, 1e10, 1e9, 1e12)
    grid = surface["net_benefit_seconds"]
    assert len(grid) == 2 and len(grid[0]) == 3
    # 같은 depth 라도 대역폭에 따라 부호가 바뀐다
    assert grid[0][0] > grid[0][-1]


def test_transfer_time_uses_bits_per_second():
    assert transfer_seconds(1e6, bandwidth_mbps=8) == pytest.approx(1.0)
