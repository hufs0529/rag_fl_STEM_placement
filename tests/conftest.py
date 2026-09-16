"""테스트 공용 픽스처. transformers 없이도 토크나이즈 경로를 검증하기 위한 가짜 토크나이저."""

import pytest


class FakeTokenizer:
    """공백 단위로 자르는 최소 토크나이저 - id 는 단어 해시.

    실제 토크나이저를 쓰지 않는 이유: 라벨 마스킹과 좌측 절단 로직은 어휘와
    무관한 순수 인덱스 연산이므로, 무거운 의존성 없이 검증할 수 있어야 한다.
    """

    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [abs(hash(tok)) % 30000 + 10 for tok in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def passages():
    return [
        {"pid": "p0", "title": "Paris", "text": "Paris is the capital of France."},
        {"pid": "p1", "title": "France", "text": "France is in western Europe."},
        {"pid": "p2", "title": "Seine", "text": "The Seine flows through Paris."},
    ]
