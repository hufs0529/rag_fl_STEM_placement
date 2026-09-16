import pytest

from src.metrics import (
    aggregate, answer_in_passage, exact_match, gold_recall_at_k,
    normalize_answer, score_prediction, token_f1,
)


def test_normalisation_strips_articles_punctuation_and_case():
    assert normalize_answer("The Beatles!") == "beatles"
    assert normalize_answer("  a   Dog's  life. ") == "dogs life"


def test_f1_is_partial_credit_where_em_is_not():
    pred, gold = "william shakespeare", "shakespeare"
    assert exact_match(pred, gold) == 0.0
    assert 0.0 < token_f1(pred, gold) < 1.0


def test_score_takes_the_max_over_multiple_golds():
    scored = score_prediction("Paris", ["Lyon", "paris, france", "Paris"])
    assert scored["em"] == 1.0
    assert scored["f1"] == 1.0


def test_aggregate_is_on_the_0_100_scale():
    out = aggregate(["paris", "wrong"], [["Paris"], ["Berlin"]])
    assert out["em"] == 50.0
    assert out["n"] == 2


def test_answer_containment_is_normalised_not_literal():
    passage = "The capital, Paris, is on the Seine."
    assert answer_in_passage(passage, ["paris"])
    assert answer_in_passage(passage, ["the Paris!"])
    assert not answer_in_passage(passage, ["berlin"])


def test_empty_answer_never_matches():
    # 빈 정답이 모든 passage 에 매칭되면 gold 라벨이 오염된다
    assert not answer_in_passage("anything at all", ["", "   "])


def test_gold_recall_at_k_only_looks_at_the_first_k():
    selected = ["p0", "p1", "p2", "gold"]
    assert gold_recall_at_k(selected, ["gold"], k=3) == 0.0
    assert gold_recall_at_k(selected, ["gold"], k=4) == 1.0


def test_recall_counts_gold_ids_not_answer_strings():
    """코퍼스의 "아무 책"도 정답 문자열을 담고 있다(위키의 99.997%가 그렇다).

    문자열로 다시 세면 그런 책을 가져와도 성공이 되어 네 조건 모두의 recall 이
    부풀고, 조건 간 차이가 그만큼 깎인다.
    """
    from src.metrics import gold_recall_by_text
    # 우연히 정답 단어를 담은 "아무 책"을 가져온 상황
    assert gold_recall_by_text(["an unrelated page mentioning paris"], ["Paris"], 3) == 1.0
    assert gold_recall_at_k(["random_pid"], ["gold_pid"], 3) == 0.0


# --- 처리가 작동할 수 있는 범위 --------------------------------------------

from src.metrics import headroom_summary, retrieval_bucket

def test_gold_already_in_the_top_k_leaves_nothing_for_depth_to_do():
    assert retrieval_bucket(["g", "p1", "p2", "p3"], ["g"], 3) == "already"


def test_gold_below_the_top_k_but_inside_the_pool_is_the_working_range():
    assert retrieval_bucket(["p0", "p1", "p2", "g", "p4"], ["g"], 3) == "recoverable"


def test_gold_outside_the_pool_is_unreachable_at_any_depth():
    assert retrieval_bucket(["p0", "p1", "p2"], ["g"], 3) == "out_of_pool"


def test_a_question_with_several_gold_passages_needs_only_one():
    assert retrieval_bucket(["p0", "p1", "g2", "p3"], ["g1", "g2", "g3"], 3) == "already"


def test_summary_separates_the_working_range_from_the_ceiling():
    out = headroom_summary(["already", "already", "recoverable", "out_of_pool"])
    assert out["already_at_top_k"] == 0.5
    assert out["recoverable"] == 0.25
    assert out["out_of_pool"] == 0.25
    # 상한이 높아도(0.75) 처리가 일할 폭은 0.25 뿐 - 둘을 구분해야 한다
    assert out["ceiling"] == 0.75


def test_paired_delta_counts_only_the_questions_that_flipped():
    from src.metrics import paired_recall_delta
    # 210 개는 양쪽 다 맞음, 4 개는 처리만 맞음, 86 개는 양쪽 다 틀림
    base = [1.0] * 210 + [0.0] * 90
    treat = [1.0] * 210 + [1.0] * 4 + [0.0] * 86
    r = paired_recall_delta(base, treat)
    assert (r["gained"], r["lost"], r["discordant"]) == (4, 0, 4)
    assert r["delta"] == pytest.approx(4 / 300)
    assert r["std_error"] == pytest.approx(2 / 300)       # sqrt(4)/300
    assert r["z"] == pytest.approx(2.0)


def test_paired_delta_on_identical_vectors_has_no_uncertainty():
    from src.metrics import paired_recall_delta
    r = paired_recall_delta([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert r["delta"] == 0.0 and r["std_error"] == 0.0
    assert r["z"] is None                                  # 0/0 을 만들지 않는다
    assert r["separated_from_zero"] is False


def test_paired_delta_nets_losses_against_gains():
    from src.metrics import paired_recall_delta
    r = paired_recall_delta([0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert (r["gained"], r["lost"]) == (1, 1)
    assert r["delta"] == 0.0
    assert r["std_error"] > 0.0                            # 불확실성은 0 이 아니다
    assert r["separated_from_zero"] is False


def test_paired_delta_rejects_mismatched_lengths():
    from src.metrics import paired_recall_delta
    with pytest.raises(ValueError):
        paired_recall_delta([1.0], [1.0, 0.0])
