"""탐욕적(greedy) 배치 생성 - 평가와 gate (i) 에서 공유.

계획서 Subtask 1.3: greedy decoding 으로 샘플링 분산을 제거한다. 조건 간
비교에서 디코딩 무작위성이 섞이면 작은 효과가 묻히기 때문이다.

Greedy decoding shared by evaluation and gate (i), removing sampling variance
between conditions.
"""

from typing import List, Sequence


def greedy_generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 32,
    batch_size: int = 16,
    max_length: int = 1024,
    device: str = None,
) -> List[str]:
    """프롬프트 배치를 탐욕 디코딩. 생성된 부분만 잘라서 돌려준다.

    왼쪽 패딩을 쓰는 이유: causal LM 의 배치 생성에서 오른쪽 패딩은 생성 시작
    위치를 어긋나게 만든다. 생성 동안만 padding_side 를 바꾸고 복구한다.
    """
    import torch

    device = device or next(model.parameters()).device
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: List[str] = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = list(prompts[start : start + batch_size])
                enc = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length - max_new_tokens,
                ).to(device)
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_tokens = generated[:, enc["input_ids"].shape[1] :]
                outputs.extend(
                    tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                )
    finally:
        tokenizer.padding_side = original_side
    return [postprocess_answer(o) for o in outputs]


def postprocess_answer(text: str) -> str:
    """첫 줄만 취하고 공백 정리 - 짧은 추출형 답만 채점 대상이다."""
    return text.strip().split("\n")[0].strip()
