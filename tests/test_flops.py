from src.flops import (
    cross_encoder_flops_per_pair, describe, embedder_flops_per_passage,
    transformer_forward_flops,
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
