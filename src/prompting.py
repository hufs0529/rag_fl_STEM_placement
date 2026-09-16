"""RAG 프롬프트 조립 + answer-token loss masking (Subtask 1.2 / 1.3).

retrieval 은 local step 안에 있다: 질문을 임베딩 -> 클라이언트 i 의 사설 인덱스에서
후보 검색 -> 재채점 -> 선택된 passage 를 프롬프트 앞에 붙임 -> 그 다음 loss 계산.
따라서 reranking 은 모델이 아니라 "학습 입력"을 바꾸고, 효과는 gradient 품질을
통해 전파된다.

프롬프트 템플릿은 모든 조건에서 동일하다. 조건 간 차이는 passage 3개의 내용뿐이다.

Prompt assembly and answer-token loss masking. Retrieval sits inside the local
step, so reranking alters the training inputs rather than the model, and its
effect propagates through gradient quality. The template is identical across
every condition; only the content of the three passages differs.
"""

from typing import Dict, List, Sequence

SYSTEM_INSTRUCTION = (
    "Answer the question using only the context below. "
    "Reply with the shortest span that answers it."
)

PROMPT_TEMPLATE = (
    "{system}\n\n"
    "{context_block}"
    "Question: {question}\n"
    "Answer:"
)

CONTEXT_ENTRY = "[{i}] {title}: {text}\n"

# gate (i) 전용 few-shot 시연.
#
# 왜 필요한가: 학습 전 135M 모델은 "짧은 추출형 답"이라는 **출력 형식**을 모른다.
# 실측상 빈 문자열을 뱉거나 컨텍스트 형식을 이어쓴다
#   ''                                   -> F1 0
#   '[2] "International Regulations..."'  -> F1 0에 가까움
# 그 상태로 재면 게이트가 "맥락을 쓸 수 있는가"가 아니라 "형식을 아는가"를 재게 되어,
# 모델을 키워도 해결되지 않는 실패로 오진한다. 본실험은 answer-token loss 로 형식을
# 가르치므로, 게이트에서는 시연으로 형식만 알려주고 **맥락 활용 능력만 분리해 잰다**.
#
# 시연은 손으로 쓴 짧은 것이다 - 데이터셋 질문을 쓰면 평가 질문과 겹칠 수 있고,
# 길면 컨텍스트 예산을 먹는다. 컨텍스트를 포함시키는 이유는 "주어진 글에서 뽑아
# 답하라"는 동작까지 함께 보여주기 위함이다.
FEWSHOT_DEMOS = [
    {
        "title": "Paris",
        "text": "Paris is the capital and most populous city of France.",
        "question": "what is the capital of france",
        "answer": "Paris",
    },
    {
        "title": "Apollo 11",
        "text": "Apollo 11 landed the first humans on the Moon on 20 July 1969.",
        "question": "when did apollo 11 land on the moon",
        "answer": "20 July 1969",
    },
]


def format_fewshot(n_shots: int) -> str:
    """시연 n개를 프롬프트 앞에 붙일 문자열로. 0이면 빈 문자열(기본 동작)."""
    if n_shots <= 0:
        return ""
    blocks = []
    for demo in FEWSHOT_DEMOS[:n_shots]:
        blocks.append(
            "Context:\n"
            + CONTEXT_ENTRY.format(i=1, title=demo["title"], text=demo["text"])
            + f"\nQuestion: {demo['question']}\nAnswer: {demo['answer']}\n"
        )
    return "\n".join(blocks) + "\n"


def format_context(passages: Sequence[dict]) -> str:
    """선택된 passage 를 프롬프트용 블록으로. 개수는 전 조건 3개로 고정."""
    if not passages:
        return "Context: (none)\n\n"
    body = "".join(
        CONTEXT_ENTRY.format(i=i + 1, title=p.get("title", "").strip(), text=p["text"].strip())
        for i, p in enumerate(passages)
    )
    return "Context:\n" + body + "\n"


def build_prompt(question: str, passages: Sequence[dict], n_shots: int = 0) -> str:
    """프롬프트 조립. n_shots 는 gate (i) 전용이며 기본 0 - 학습·평가 경로는 영향 없음."""
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTRUCTION,
        context_block=format_fewshot(n_shots) + format_context(passages),
        question=question.strip(),
    )


def build_training_text(question: str, passages: Sequence[dict], answer: str) -> Dict[str, str]:
    """(prompt, target) 쌍. target 앞의 공백은 토크나이저 경계를 안정화한다."""
    return {"prompt": build_prompt(question, passages), "target": " " + answer.strip()}


def tokenize_with_answer_mask(
    tokenizer,
    prompt: str,
    target: str,
    max_length: int = 1024,
) -> Dict[str, List[int]]:
    """prompt 구간은 라벨을 -100 으로 마스킹하고 answer token 에서만 loss 를 계산한다.

    컨텍스트가 길어 max_length 를 넘길 때는 **prompt 의 앞쪽(컨텍스트)** 을
    자른다. 정답 토큰을 자르면 그 예제는 학습 신호가 사라지므로, 잘림이
    처리 조건 간 체계적 차이를 만들지 않도록 항상 왼쪽에서 자른다.

    Mask the prompt span in labels (-100) and compute loss only on answer
    tokens. When the sequence exceeds max_length the prompt is truncated from
    the LEFT, never the answer: dropping answer tokens would remove the training
    signal, and left-truncation keeps truncation from differing systematically
    between treatment conditions.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        target_ids = target_ids + [eos]

    budget = max_length - len(target_ids)
    if budget <= 0:
        raise ValueError("max_length is too small to hold the answer tokens")
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_answer_tokens": len(target_ids),
    }


def collate_with_padding(batch: Sequence[Dict[str, List[int]]], pad_token_id: int):
    """오른쪽 패딩. 패딩 위치의 라벨은 -100 으로 두어 loss 에서 제외한다."""
    import torch

    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention.append(x["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
