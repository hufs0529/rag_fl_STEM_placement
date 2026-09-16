"""200k 검색 코퍼스 구성 (Subtask 1.1).

구성 요소 세 가지:
  1) gold passage      - 질문당 gold_per_question 개까지
  2) BM25 hard negative - 질문과 어휘적으로 가깝지만 **정답 문자열을 포함하지 않는** passage
  3) random remainder   - 스캔이 reservoir 로 모아둔 **정답 미포함** passage 에서 무작위

세 성분이 corpus_size 예산을 나눠 갖는다. gold 를 질문당 무제한으로 넣으면
(6만 질문 x 최대 20개) 예산을 혼자 다 먹고 remainder 가 사라져, 코퍼스 전체가
"누군가의 정답을 담은 passage"가 된다 - 검색 난이도가 의도와 달라진다.
그래서 gold 수에 상한을 두고, 그래도 넘치면 **조용히 넘기지 않고 에러를 낸다**.

hard negative 에서 정답 보유 passage 를 배제하는 것이 핵심이다. 정답을 담은
distractor 를 negative 로 넣으면 recall 판정이 false-positive 를 내고
(우리는 정답 문자열 포함으로 gold 를 판정하므로) 처리 효과 측정이 오염된다.

Excluding answer-bearing distractors prevents false-positive recall: because
gold is identified by answer-string containment, a "negative" that contains the
answer string would be counted as a successful retrieval.

passage 는 이미 100 단어로 고정되어 있으므로 chunking 단계는 없다.
"""

import random
from typing import Dict, Iterable, List, Optional, Sequence

from src.metrics import answer_in_passage


def collect_gold_passages(
    questions: Sequence[dict], store, max_per_question: int = None
) -> List[str]:
    """표본 질문의 gold passage id (중복 제거, 코퍼스에 실재하는 것만)."""
    out, seen = [], set()
    for q in questions:
        golds = q["gold_passage_ids"]
        if max_per_question:
            golds = golds[:max_per_question]
        for pid in golds:
            if pid in store and pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def bm25_hard_negatives(
    questions: Sequence[dict],
    store,
    candidate_pids: Sequence[str],
    per_question: int = 5,
    overfetch: int = 4,
    pool_size: Optional[int] = None,
    seed: int = 42,
    progress_every: int = 500,
    log=print,
) -> List[str]:
    """BM25 로 어휘적으로 가까운 passage 를 뽑되 정답 보유 passage 는 버린다.

    overfetch 배수만큼 넉넉히 뽑는 이유: 상위권이 대부분 정답 보유 passage 로
    채워지는 질문이 있어, 필터링 후에도 per_question 개를 확보하기 위함이다.

    **비용 주의:** rank_bm25 는 순수 파이썬이고 get_scores() 가 질의 1개마다
    코퍼스 전체를 훑는다. 따라서 비용이 (질문 수 x 후보 수) 로 곱해져, 상한이
    없으면 6만 질문 x 수백만 후보 = 사실상 끝나지 않는다. pool_size 로 BM25
    인덱스 크기를, 호출부에서 질문 수를 각각 제한한다. 둘 다 키우면 negative 가
    어려워지지만 시간도 같은 비율로 는다.

    Pure-python BM25 scores every document per query, so the cost multiplies as
    (questions x candidates). Both are capped, and the trade-off is explicit:
    larger caps give harder negatives at proportionally more time.
    """
    from rank_bm25 import BM25Okapi

    corpus_ids = list(candidate_pids)
    if pool_size and len(corpus_ids) > pool_size:
        corpus_ids = random.Random(seed).sample(corpus_ids, pool_size)
    if not corpus_ids or not questions:
        return []

    log(f"  BM25 over {len(corpus_ids):,} candidates for {len(questions):,} questions "
        f"(cost scales as their product)")
    tokenised = [store.get(pid)["text"].lower().split() for pid in corpus_ids]
    bm25 = BM25Okapi(tokenised)

    negatives, seen = [], set()
    for processed, q in enumerate(questions, 1):
        if progress_every and processed % progress_every == 0:
            log(f"    BM25 {processed:,}/{len(questions):,} questions | "
                f"{len(negatives):,} negatives")
        scores = bm25.get_scores(q["question"].lower().split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        taken = 0
        for idx in order[: per_question * overfetch]:
            pid = corpus_ids[idx]
            if pid in seen:
                continue
            passage = store.get(pid)
            if answer_in_passage(passage["text"], q["answers"]):
                continue                      # 정답 보유 distractor 는 negative 가 아니다
            seen.add(pid)
            negatives.append(pid)
            taken += 1
            if taken >= per_question:
                break
    return negatives


def fill_random_remainder(
    chosen: Sequence[str], pool: Sequence[str], target_size: int, seed: int = 42
) -> List[str]:
    """목표 코퍼스 크기까지 무작위 passage 로 채운다.

    pool 은 스캔이 reservoir 로 모아둔 **정답 미포함** passage 여야 한다.
    gold 더미에서 채우면 remainder 가 remainder 가 아니게 된다.
    """
    chosen_set = set(chosen)
    remainder = [pid for pid in pool if pid not in chosen_set]
    rng = random.Random(seed)
    rng.shuffle(remainder)
    needed = max(0, target_size - len(chosen_set))
    return list(chosen) + remainder[:needed]


def build_corpus(
    questions: Sequence[dict],
    store,
    target_size: int = 200_000,
    hard_negatives_per_question: int = 5,
    gold_per_question: Optional[int] = 2,
    hard_negative_questions: Optional[int] = None,
    bm25_pool_size: Optional[int] = None,
    random_pool: Sequence[str] = None,
    seed: int = 42,
    log=print,
) -> Dict[str, object]:
    """gold + hard negative + random 으로 목표 크기의 코퍼스를 만든다.

    세 성분이 target_size 예산을 나눠 갖는다:
      - gold         : 질문당 gold_per_question 개까지 (전부 포함하면 예산 초과)
      - hard negative: 앞 hard_negative_questions 개 질문에 대해서만 (BM25 비용)
      - remainder    : random_pool (정답 미포함 표본) 에서 나머지를 채움

    반환에는 구성 비율과 부족분을 함께 담는다. 코퍼스 구성은 recall 난이도를
    직접 결정하므로 보고서에 그대로 실려야 한다.
    """
    gold = collect_gold_passages(questions, store, max_per_question=gold_per_question)
    cap = f" (up to {gold_per_question} per question)" if gold_per_question else ""
    log(f"  gold passages: {len(gold):,}{cap}")

    if len(gold) > target_size:
        raise ValueError(
            f"gold passages ({len(gold):,}) from {len(questions):,} questions already "
            f"exceed the corpus budget ({target_size:,}). Three knobs, in the order "
            "usually wanted:\n"
            "  data.question_limit            - fewer questions (this is what a "
            "reduced-scale run needs; shrinking corpus_size alone leaves the question "
            "count at full size)\n"
            "  data.gold_per_question_in_corpus - fewer gold per question\n"
            "  data.corpus_size              - a larger budget\n"
            "Filling the whole corpus with gold would make every passage answer-bearing "
            "and change retrieval difficulty, so this is refused rather than truncated."
        )

    negative_questions = list(questions)
    if hard_negative_questions:
        negative_questions = negative_questions[:hard_negative_questions]

    gold_set = set(gold)
    pool = list(random_pool) if random_pool is not None else store.ids()
    negative_candidates = [pid for pid in pool if pid not in gold_set]
    negatives = bm25_hard_negatives(
        negative_questions, store, negative_candidates, hard_negatives_per_question,
        pool_size=bm25_pool_size, seed=seed, log=log,
    )
    log(f"  BM25 hard negatives (answer-free): {len(negatives):,} "
        f"from {len(negative_questions):,} questions")

    # gold 단독뿐 아니라 **gold + negative 합계**가 예산을 넘는지도 본다.
    # 넘으면 fill_random_remainder 가 needed=0 을 돌려주어 remainder 가 통째로
    # 사라지고(코퍼스가 전부 answer-bearing 이 됨) 크기도 조용히 초과한다.
    # 바로 그 실패를 막으려고 remainder 를 따로 모았으므로, 여기서 멈춘다.
    if len(gold) + len(negatives) > target_size:
        raise ValueError(
            f"gold ({len(gold):,}) + hard negatives ({len(negatives):,}) = "
            f"{len(gold) + len(negatives):,} exceeds the corpus budget ({target_size:,}), "
            "which would leave no random remainder at all. Lower "
            f"data.gold_per_question_in_corpus (currently {gold_per_question}) or "
            f"data.hard_negative_questions (currently {len(negative_questions):,}), "
            "or raise data.corpus_size."
        )

    selected = fill_random_remainder(gold + negatives, pool, target_size, seed)
    shortfall = max(0, target_size - len(selected))
    log(f"  after random remainder: {len(selected):,}"
        + (f"  (SHORT by {shortfall:,})" if shortfall else ""))
    if shortfall:
        log("  WARNING: the random pool could not fill the corpus. Raise "
            "data.random_pool_size or scan more passages; the corpus is now more "
            "answer-bearing than intended.")

    return {
        "passage_ids": selected,
        "composition": {
            "gold": len(gold),
            "hard_negatives": len(negatives),
            "random_remainder": max(0, len(selected) - len(gold) - len(negatives)),
            "total": len(selected),
            "target_size": target_size,
            "shortfall": shortfall,
            "gold_per_question": gold_per_question,
            "hard_negative_questions": len(negative_questions),
        },
    }


def verify_no_answer_bearing_negatives(
    questions: Sequence[dict], negative_pids: Iterable[str], store
) -> List[dict]:
    """검증용: negative 집합에 정답 보유 passage 가 섞이지 않았는지 확인.

    조용히 통과하면 recall 이 부풀려지므로, 위반 목록을 그대로 돌려준다.
    """
    negative_set = set(negative_pids)
    violations = []
    for q in questions:
        for pid in negative_set.intersection(q["gold_passage_ids"]):
            violations.append({"qid": q["qid"], "pid": pid})
    return violations
