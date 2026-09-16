import pytest

from src.prompting import build_prompt, build_training_text, tokenize_with_answer_mask


def test_prompt_contains_every_selected_passage_and_the_question(passages):
    prompt = build_prompt("what is the capital of france?", passages)
    for p in passages:
        assert p["text"] in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_shape_is_identical_across_conditions(passages):
    # 조건 간 차이는 passage 내용뿐이어야 한다 - 템플릿 자체가 달라지면 안 된다
    a = build_prompt("q", passages)
    b = build_prompt("q", list(reversed(passages)))
    assert a.count("Context:") == b.count("Context:") == 1
    assert len(a.splitlines()) == len(b.splitlines())


def test_loss_is_computed_only_on_answer_tokens(tokenizer, passages):
    text = build_training_text("who?", passages, "Paris")
    enc = tokenize_with_answer_mask(tokenizer, text["prompt"], text["target"], max_length=512)
    masked = sum(1 for label in enc["labels"] if label == -100)
    assert masked == enc["n_prompt_tokens"]
    assert enc["labels"][-1] == tokenizer.eos_token_id
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"])


def test_overlong_prompts_are_truncated_from_the_left_never_the_answer(tokenizer, passages):
    text = build_training_text("who?", passages * 20, "Paris")
    enc = tokenize_with_answer_mask(tokenizer, text["prompt"], text["target"], max_length=32)
    assert len(enc["input_ids"]) == 32
    # 정답 토큰은 전부 살아 있어야 한다
    assert enc["labels"][-enc["n_answer_tokens"]:] == enc["input_ids"][-enc["n_answer_tokens"]:]


def test_max_length_smaller_than_the_answer_is_an_error(tokenizer, passages):
    text = build_training_text("who?", passages, "a rather long answer span here")
    with pytest.raises(ValueError):
        tokenize_with_answer_mask(tokenizer, text["prompt"], text["target"], max_length=2)


# --- gate (i) 전용 형식 시연 (few-shot) ------------------------------------

from src.prompting import FEWSHOT_DEMOS, format_fewshot


def test_training_prompts_are_unchanged_by_default(passages):
    # 시연은 게이트 전용이다 - 학습·평가 경로는 기본값 0 으로 영향받지 않는다
    assert build_prompt("q", passages) == build_prompt("q", passages, n_shots=0)
    assert format_fewshot(0) == ""


def test_demonstrations_show_the_short_answer_format(passages):
    prompt = build_prompt("who wrote hamlet", passages, n_shots=2)
    # 시연이 "짧은 답"을 보여주고
    assert "Answer: Paris" in prompt
    assert "Answer: 20 July 1969" in prompt
    # 진짜 질문은 맨 뒤에 답 없이 남는다
    assert prompt.rstrip().endswith("Answer:")
    assert prompt.count("Question:") == 3


def test_demonstrations_carry_their_own_context(passages):
    # "주어진 글에서 뽑아 답하라"는 동작까지 보여줘야 맥락 활용만 분리해 잴 수 있다
    prompt = build_prompt("q", passages, n_shots=1)
    assert "Paris is the capital" in prompt
    assert prompt.count("Context:") == 2


def test_more_shots_than_available_is_capped_not_an_error(passages):
    prompt = build_prompt("q", passages, n_shots=99)
    assert prompt.count("Question:") == len(FEWSHOT_DEMOS) + 1
