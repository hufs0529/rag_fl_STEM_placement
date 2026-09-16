"""게이트 (ii) 의 프로브 코퍼스 조립 - Week 2 코퍼스의 축소판.

이 로직이 두 번 틀렸기 때문에 스크립트에서 떼어내 테스트 가능하게 만들었다.

**첫 번째 실수**: 질문당 gold 를 전부(평균 13.4 권) 넣고 채움재도 gold 더미에서
가져왔다. 5,000 권 중 2,689 권이 표본 질문의 정답지(54%)가 되어 dense 가 이미
91% 를 상위 3 개 안에 넣었고, recoverable 이 7.5% 로 주저앉았다. 아무도 학습하지
않을 코퍼스의 난이도를 잰 것이다.

**두 번째 실수**: 질문당 1 권으로 고치고 나머지를 임의 위키 문서로 채우니, 이번엔
코퍼스의 8% 만 "누군가의 정답지"가 됐다. Week 2 는 40.4% 다 (80,720 / 200,000).
임의 문서는 이기기 쉽고 **다른 질문의 정답지는 어렵다** - 답이 들어있는 문서라
질의와 어휘가 겹친다. 그래서 난이도가 또 Week 2 와 달랐다.

지금 구성은 세 성분이다:
    표본 질문의 gold      질문당 1 권 (data.gold_per_question_in_corpus)
  + 다른 질문의 gold      gold 비중이 Week 2 와 같아질 만큼
  + 정답 미포함 채움재    나머지 (data/random_passages.jsonl)

불변식 하나가 정확성을 떠받친다: **표본 질문의 gold 는 단 한 권도 채움재로
들어가면 안 된다.** 질문당 1 권만 정답으로 유지하므로, 나머지 gold 가 라벨 없이
코퍼스에 앉으면 모델이 사실상 맞는 책을 받았는데 recall 은 틀렸다고 센다.

BM25 hard negative 는 넣지 않는다. 넣으면 gold 가 더 아래로 밀려 recoverable 이
**늘어나므로**, 빼고 재는 쪽이 보수적이다.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple


def week2_gold_share(n_questions_total: int, corpus_size: int, gold_per_question: int = 1) -> float:
    """Week 2 코퍼스에서 "누군가의 정답지"가 차지하는 비중.

    설정에 상수로 박지 않고 실제 질문 수에서 유도한다 - 스캔 결과가 달라지면
    이 값도 함께 달라져야 한다.
    """
    if corpus_size <= 0:
        return 0.0
    return min(1.0, gold_per_question * max(0, n_questions_total) / float(corpus_size))


def select_question_gold(
    questions: Sequence[dict],
    contains: Callable[[str], bool],
    gold_per_question: int = 1,
) -> Tuple[List[dict], List[str], set]:
    """표본 질문에서 질문당 gold 를 최대 gold_per_question 권 고른다.

    반환: (유지된 질문, gold id 목록, 금지 id 집합)

    금지 집합은 **잘리지 않은** 원본 gold 목록 전체다. 유지된 1 권만이 아니다.
    """
    kept: List[dict] = []
    gold_ids: List[str] = []
    seen = set()
    forbidden = set()
    for q in questions:
        forbidden.update(q.get("gold_passage_ids", ()))
        owned = [pid for pid in q.get("gold_passage_ids", ()) if contains(pid)][:gold_per_question]
        if not owned:
            continue                        # 코퍼스에 넣을 gold 가 없으면 질문도 버린다
        kept.append({**q, "gold_passage_ids": owned})
        for pid in owned:
            if pid not in seen:
                seen.add(pid)
                gold_ids.append(pid)
    return kept, gold_ids, forbidden


def plan_probe_corpus(
    questions: Sequence[dict],
    contains: Callable[[str], bool],
    all_gold_ids: Sequence[str],
    filler_ids: Sequence[str],
    size: int,
    gold_share: float,
    gold_per_question: int = 1,
    shuffle: Optional[Callable[[list], None]] = None,
) -> Dict[str, object]:
    """프로브 코퍼스의 구성을 정한다. 임베딩이나 I/O 는 하지 않는다.

    questions   - 표본 질문
    contains    - pid 가 gold 저장소에 있는지 판정
    all_gold_ids- gold 저장소의 전체 id (다른 질문의 gold 를 여기서 뽑는다)
    filler_ids  - 정답 미포함 passage 의 id
    size        - 목표 코퍼스 크기
    gold_share  - "누군가의 정답지" 목표 비중 (Week 2 에서 유도)
    """
    kept, gold_ids, forbidden = select_question_gold(questions, contains, gold_per_question)

    target_gold = int(round(max(0.0, min(1.0, gold_share)) * max(0, size)))
    taken = set(gold_ids)
    gold_filler: List[str] = []
    if target_gold > len(gold_ids):
        # 표본 질문의 gold 는 한 권도 들어갈 수 없다 - 라벨 없는 정답이 되기 때문.
        available = [pid for pid in all_gold_ids if pid not in taken and pid not in forbidden]
        if shuffle is not None:
            shuffle(available)
        gold_filler = available[: target_gold - len(gold_ids)]

    remaining = max(0, size - len(gold_ids) - len(gold_filler))
    filler = list(filler_ids)
    if shuffle is not None:
        shuffle(filler)
    filler = filler[:remaining]

    total = len(gold_ids) + len(gold_filler) + len(filler)
    return {
        "questions": kept,
        "question_gold": gold_ids,
        "other_question_gold": gold_filler,
        "answer_free_filler": filler,
        "size": total,
        "gold_share": (len(gold_ids) + len(gold_filler)) / total if total else 0.0,
        "gold_share_target": gold_share,
        "shortfall": max(0, size - total),
    }
