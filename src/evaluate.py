"""중앙 집중 평가 (Subtask 1.3).

고정된 1,000 질문 held-out 세트를 **전역 모델**에 대해 탐욕 디코딩으로 채점한다.
조건 간 비교에서 제거해야 하는 분산이 두 가지 있기 때문이다:
  - 샘플링 분산 -> greedy decoding
  - 평가셋 분산 -> 모든 run 에서 동일한 고정 세트

가장 중요한 설계: **평가 컨텍스트는 모든 조건에서 d=0 으로 고정**한다.
그래야 측정되는 것이 "reranking 으로 모델이 무엇을 배웠는가"이지
"추론 시점에 더 좋은 컨텍스트를 넣어주면 얼마나 잘하는가"가 아니게 된다.
후자를 재면 처리 효과가 부풀려지고, 이 실험의 주장(라운드 수 절감)과 무관해진다.

Central evaluation. Evaluation context is held at d=0 for all conditions,
isolating what the model learned from reranking rather than what better
inference-time context supplies.
"""

from typing import Dict, List, Sequence

from src.generate import greedy_generate
from src.metrics import aggregate, gold_recall_at_k
from src.prompting import build_prompt
from src.selection import select_passages


def build_eval_prompts(
    questions: Sequence[dict],
    caches: Dict[int, "object"],
    store,
    eval_depth: int = 0,
    top_k: int = 3,
) -> List[dict]:
    """평가용 프롬프트. eval_depth 는 설정상 0 으로 고정되어 있다.

    질문의 소속 클라이언트 캐시에서 후보를 가져온다: 평가 질문도 어딘가의
    사설 인덱스에서 검색되어야 하며, 중앙 인덱스를 쓰면 배포 상황과 어긋난다.
    """
    prompts = []
    for q in questions:
        cache = caches[q["client_id"]]
        candidates, scores = cache.get(q["qid"])
        selected = select_passages(candidates, scores, eval_depth, top_k)
        prompts.append({
            "qid": q["qid"],
            "prompt": build_prompt(q["question"], store.many(selected)),
            "answers": q["answers"],
            "selected_ids": selected,
            # recall 진단은 gold id 로 대조한다 (문자열 재매칭은 코퍼스의
            # "아무 책"까지 성공으로 세어 모든 조건의 수치를 부풀린다)
            "gold_passage_ids": list(q.get("gold_passage_ids", ())),
        })
    return prompts


def evaluate_model(
    model,
    tokenizer,
    eval_prompts: Sequence[dict],
    store,
    max_new_tokens: int = 32,
    max_length: int = 1024,
    batch_size: int = 16,
    top_k: int = 3,
) -> Dict[str, float]:
    """전역 모델을 고정 평가셋에서 채점. F1 이 primary, EM 은 병기."""
    predictions = greedy_generate(
        model,
        tokenizer,
        [p["prompt"] for p in eval_prompts],
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )
    scores = aggregate(predictions, [p["answers"] for p in eval_prompts])

    # 진단: 평가셋에서의 gold recall@3 (평가 컨텍스트는 d=0 고정이므로
    # 조건 간 동일해야 한다 - 달라지면 평가 경로에 처리가 새어든 것이다)
    recall = sum(
        gold_recall_at_k(p["selected_ids"], p.get("gold_passage_ids", ()), top_k)
        for p in eval_prompts
    ) / max(1, len(eval_prompts))

    return {**scores, "eval_gold_recall_at_3": recall}


def sample_predictions(
    model, tokenizer, eval_prompts: Sequence[dict], n: int = 5, **kwargs
) -> List[dict]:
    """보고서에 실을 정성 예시 몇 개."""
    subset = list(eval_prompts)[:n]
    predictions = greedy_generate(model, tokenizer, [p["prompt"] for p in subset], **kwargs)
    return [
        {"qid": p["qid"], "answers": p["answers"], "prediction": pred}
        for p, pred in zip(subset, predictions)
    ]
