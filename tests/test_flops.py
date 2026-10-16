from src.flops import (
    CE_TOKENS, EMBED_TOKENS, cross_encoder_flops_per_pair, describe,
    embedder_flops_per_passage, transformer_forward_flops,
)


def test_forward_flops_scale_with_parameters_and_tokens():
    base = transformer_forward_flops(1_000_000, 100)
    assert transformer_forward_flops(2_000_000, 100) == 2 * base
    assert transformer_forward_flops(1_000_000, 200) == 2 * base


def test_attention_term_is_quadratic_in_sequence_length():
    short = transformer_forward_flops(1_000_000, 100, n_layers=6, hidden=384)
    long = transformer_forward_flops(1_000_000, 200, n_layers=6, hidden=384)
    assert long > 2 * short          # 선형항만이면 정확히 2배였을 것


def test_the_distilled_reranker_is_cheaper_per_pair_than_a_full_encoder():
    assert cross_encoder_flops_per_pair() < transformer_forward_flops(110_000_000, 256, 12, 768)


def test_describe_records_whether_the_number_was_measured():
    analytic = embedder_flops_per_passage()
    assert describe(None, analytic, "bge")["source"] == "analytic"
    measured = describe(1.23e9, analytic, "bge")
    assert measured["source"] == "measured"
    assert measured["flops"] == 1.23e9
    # 추정치도 함께 남겨 보고서에서 둘을 대조할 수 있게 한다
    assert measured["analytic_estimate"] == analytic


def test_the_cost_per_pair_excludes_the_embedding_matrix():
    """2 x params x tokens 의 params 는 **행렬곱에 들어가는** 파라미터여야 한다.

    MiniLM-L6 은 22.7M 중 11.9M(52%)이 임베딩이고, 임베딩은 gather 로 읽혀
    FLOPs 를 거의 내지 않는다. 전체 수를 넣으면 2.1 배 과대계상되고, 이 값은
    환율의 **분모**라서 결론이 그만큼 유리하게 왜곡된다.
    """
    per_pair = cross_encoder_flops_per_pair()
    all_params = transformer_forward_flops(22_713_601, CE_TOKENS, 6, 384)
    compute_params = transformer_forward_flops(10_795_777, CE_TOKENS, 6, 384)
    assert per_pair == compute_params
    assert per_pair < all_params / 1.9


def test_describe_flags_a_large_gap_between_measured_and_analytic():
    """두 값이 배수로 벌어지면 조용히 한쪽을 쓰지 않고 경고를 남긴다."""
    close = describe(5.4e9, 6.1e9, "ce", n_tokens_measured=256, n_tokens_assumed=256)
    assert close["agree_within_2x"] and "warning" not in close

    far = describe(1.4e9, 1.2e10, "ce", n_tokens_measured=66, n_tokens_assumed=256)
    assert not far["agree_within_2x"]
    assert "warning" in far and "배" in far["warning"]
    assert far["measured_over_analytic"] < 0.2
    assert far["n_tokens_measured"] == 66 and far["n_tokens_assumed"] == 256


def test_describe_still_falls_back_to_analytic_without_a_measurement():
    out = describe(None, 6.1e9, "ce", n_tokens_assumed=256)
    assert out["source"] == "analytic" and out["flops"] == 6.1e9
    assert "agree_within_2x" not in out
