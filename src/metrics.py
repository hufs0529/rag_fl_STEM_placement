"""NQ 공식 정규화 + F1/EM, 그리고 gold 판정용 정답 문자열 포함 검사.

- normalize_answer()   : NQ/SQuAD 공식 정규화 (소문자, 관사/문장부호/공백 제거)
- token_f1(), exact_match()
- answer_in_passage()  : Subtask 1.1 의 gold 판정 기준
                         (normalised answer-string containment, DPR top-k accuracy [8])
- gold_recall_at_k()   : gate (ii) 및 Subtask 1.3 진단 지표

F1 이 primary 인 이유: 연속형이라 학습곡선 보간(interpolation)이 안정적이다.
EM 은 이산형이라 이 규모에서 작은 효과를 잘 못 잡는다 (계획서 Subtask 1.3).

Official NQ normalisation plus F1/EM, and the answer-string containment used
for gold labelling.

F1 is primary because it is continuous and interpolates stably; EM is discrete
and resolves small effects poorly at this scale (plan, Subtask 1.3).
"""

import re
import string
from collections import Counter
from typing import Iterable, List, Sequence


def normalize_answer(s: str) -> str:
    """NQ/SQuAD 공식 정규화 / the official NQ normalisation."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        # 양쪽 다 비면 1.0, 한쪽만 비면 0.0 (SQuAD 관례)
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _max_over_golds(fn, prediction: str, golds: Sequence[str]) -> float:
    return max((fn(prediction, g) for g in golds), default=0.0)


def score_prediction(prediction: str, golds: Sequence[str]) -> dict:
    """정답 리스트에 대해 최대값을 취한다 (NQ 는 정답이 여러 개일 수 있음)."""
    return {
        "f1": _max_over_golds(token_f1, prediction, golds),
        "em": _max_over_golds(exact_match, prediction, golds),
    }


def aggregate(predictions: Sequence[str], golds_list: Sequence[Sequence[str]]) -> dict:
    """데이터셋 단위 집계 - 100점 스케일 (F1/EM 관례).

    Dataset-level aggregate on the conventional 0-100 scale.
    """
    if len(predictions) != len(golds_list):
        raise ValueError("predictions and golds_list must be the same length")
    if not predictions:
        return {"f1": 0.0, "em": 0.0, "n": 0}
    scores = [score_prediction(p, g) for p, g in zip(predictions, golds_list)]
    n = len(scores)
    return {
        "f1": 100.0 * sum(s["f1"] for s in scores) / n,
        "em": 100.0 * sum(s["em"] for s in scores) / n,
        "n": n,
    }


# --- gold labelling (Subtask 1.1) ------------------------------------------

def answer_in_passage(passage_text: str, answers: Iterable[str]) -> bool:
    """정규화된 정답 문자열이 정규화된 passage 에 포함되는가.

    NQ-open 과 psgs_w100 은 서로 독립된 리소스라 gold 라벨이 없다. DPR [8] 의
    top-k accuracy 기준을 그대로 써서 gold 를 판정한다. 이는 근사이며
    (정답 문자열을 포함해도 supporting evidence 가 아닐 수 있음) 최종 보고서의
    한계 절에 명시된다.

    Whether a normalised answer string occurs in the normalised passage. The two
    resources are independent, so this DPR-style criterion stands in for gold
    labels; the approximation is stated as a limitation in the final report.
    """
    norm_passage = normalize_answer(passage_text)
    for answer in answers:
        norm_answer = normalize_answer(answer)
        if norm_answer and norm_answer in norm_passage:
            return True
    return False


def gold_recall_at_k(
    selected_pids: Sequence[str], gold_pids: Iterable[str], k: int = 3
) -> float:
    """선택된 상위 k passage 중 **이 질문의 gold id** 가 하나라도 있으면 1.0.

    왜 문자열 재매칭이 아니라 id 대조인가 (중요):

    gold 판정은 2단계다 - 정답 문자열 포함 **그리고** 질문 내용어 겹침. 그런데
    검색 성공을 다시 문자열 포함만으로 세면 **기준이 어긋난다.** 실측상 위키
    passage 의 99.997% 가 어떤 질문의 정답 문자열을 담고 있어서, 코퍼스의
    "그냥 아무 책"을 가져와도 성공으로 세진다. 측정해보니 3권을 뽑았을 때
    우연히 성공으로 세질 확률이 전체 1.9%, 한 토큰 정답 질문은 6.6% 였다 -
    네 조건 전부에 깔리는 바닥이라 조건 간 차이를 그만큼 깎아먹는다.

    gold id 목록은 이미 2단계를 통과한 것만 담고 있으므로, id 로 대조하면
    판정 기준이 정확히 일치하고 계산도 더 싸다.

    Recall by gold-id membership, not by re-matching answer strings. Gold is a
    two-stage criterion; re-scoring with stage one alone counts random corpus
    passages as successful retrievals (measured: 1.9% of triples overall, 6.6%
    for single-token answers), inflating every condition and compressing the
    difference the experiment is trying to measure.
    """
    gold = set(gold_pids)
    return float(any(pid in gold for pid in selected_pids[:k]))


def gold_recall_by_text(
    selected_passages: Sequence[str], answers: Sequence[str], k: int = 3
) -> float:
    """문자열 포함만으로 보는 느슨한 recall - **진단 비교용으로만** 쓴다.

    gold id 가 없는 상황(예: 코퍼스 밖 임시 실험)에서만 의미가 있다. 본 실험의
    recall 은 gold_recall_at_k 를 쓴다.
    """
    return float(any(answer_in_passage(p, answers) for p in selected_passages[:k]))


# --- 처리가 작동할 수 있는 범위 (Subtask 1.2 진단) --------------------------

def retrieval_bucket(
    pool_pids: Sequence[str], gold_pids: Iterable[str], top_k: int = 3
) -> str:
    """한 질문을 세 통 중 하나에 넣는다. depth 가 **무엇을 할 수 있는지**가 여기서 갈린다.

      already      - 정답지가 이미 dense 상위 top_k 안에 있음
                     -> 모든 depth 가 같은 passage 를 고른다. 처리 효과 0.
      recoverable  - 정답지가 풀 안에는 있지만 top_k 밖
                     -> **깊게 재채점해야 건질 수 있다. 처리가 일할 수 있는 유일한 구간.**
      out_of_pool  - 정답지가 후보 풀에 아예 없음
                     -> d=50 도 못 찾는다. 처리 효과 0.

    상한(ceiling = already + recoverable)만 보면 안 되는 이유: 상한이 높아도
    대부분이 already 면 처리가 건드릴 게 없다. 실제로 중요한 양은 recoverable 이다.

    Which of three buckets a question falls into. Only `recoverable` is the range
    in which reranking depth can change anything; the ceiling conflates it with
    questions that were already easy.
    """
    gold = set(gold_pids)
    if any(pid in gold for pid in pool_pids[:top_k]):
        return "already"
    return "recoverable" if any(pid in gold for pid in pool_pids) else "out_of_pool"


def headroom_summary(buckets: Sequence[str]) -> dict:
    """세 통의 비율과, 처리가 일할 수 있는 폭(recoverable)을 돌려준다."""
    n = len(buckets) or 1
    counts = {b: buckets.count(b) for b in ("already", "recoverable", "out_of_pool")}
    return {
        "n": len(buckets),
        "already_at_top_k": counts["already"] / n,
        "recoverable": counts["recoverable"] / n,
        "out_of_pool": counts["out_of_pool"] / n,
        # 상한 = 풀 안에 정답지가 있는 비율
        "ceiling": (counts["already"] + counts["recoverable"]) / n,
        "counts": counts,
    }


def paired_recall_delta(baseline: Sequence[float], treatment: Sequence[float]) -> dict:
    """같은 질문 집합에서 두 조건의 recall 차이와 그 불확실성 (쌍체 비교).

    d=0 과 d=50 은 **같은 질문**을 보므로 두 평균을 독립 표본처럼 다루면 안 된다.
    두 조건이 똑같이 맞힌 질문과 똑같이 틀린 질문은 차이에 아무 정보도 주지 않는다.
    정보는 판정이 뒤바뀐 질문에만 있다 (McNemar 의 구조):

        gained = 처리는 맞혔고 기준은 틀린 질문 수
        lost   = 기준은 맞혔고 처리는 틀린 질문 수
        delta  = (gained - lost) / n
        s.e.   = sqrt(gained + lost) / n

    이 지표가 필요한 이유는 실측에서 드러났다. 질문 300 개로 돌렸을 때 상승폭이
    0.0133 이었는데, 그 실체는 **질문 4 개**였다. 비율만 보면 커 보이고 뒤바뀐
    질문 수를 보면 작다. 게이트가 단조성만 확인하고 통과시키면, 0 과 구별되지
    않는 효과를 처리 축의 존재 증거로 오해하게 된다.

    Paired (McNemar-style) difference between two recall vectors measured on the
    same questions. Only discordant questions carry information about the
    difference, so the standard error is sqrt(gained + lost) / n.
    """
    if len(baseline) != len(treatment):
        raise ValueError("baseline and treatment must be the same length")
    n = len(baseline)
    if n == 0:
        return {"n": 0, "gained": 0, "lost": 0, "delta": 0.0,
                "std_error": 0.0, "z": None, "separated_from_zero": False}
    gained = sum(1 for b, t in zip(baseline, treatment) if t > b)
    lost = sum(1 for b, t in zip(baseline, treatment) if t < b)
    discordant = gained + lost
    delta = (gained - lost) / n
    std_error = (discordant ** 0.5) / n
    # 뒤바뀐 질문이 하나도 없으면 차이가 정확히 0 이고 불확실성도 0 이다.
    # z 를 0/0 으로 만들지 않도록 None 을 돌려준다.
    z = (gained - lost) / (discordant ** 0.5) if discordant else None
    return {
        "n": n,
        "gained": gained,
        "lost": lost,
        "discordant": discordant,
        "delta": delta,
        "std_error": std_error,
        "z": z,
        # |z| >= 2 를 "0 과 구별된다"의 기준으로 쓴다 (약 95%).
        "separated_from_zero": bool(z is not None and abs(z) >= 2.0),
    }
