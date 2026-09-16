"""NQ-open 로딩과 gold 판정용 정답 인덱스 (Subtask 1.1).

NQ-open(87,925 train / 3,610 validation)과 DPR psgs_w100 은 서로 독립된
리소스이므로 gold passage 라벨이 없다. 따라서 정규화된 정답 문자열 포함으로
gold 를 판정하고, 일치하는 passage 가 없는 질문은 버린다.

**정답 문자열 포함만으로는 코퍼스 전역 라벨러로 쓸 수 없다.** DPR[8] 의 기준은
"그 질문에 대해 **검색된 후보** 안에 정답이 있는가"이지 "코퍼스 어딘가에 정답
문자열이 있는가"가 아니다. NQ 정답의 69% 는 한 토큰("1950", "North", "25")이라,
코퍼스 전역에 적용하면 사실상 모든 passage 가 누군가의 gold 가 된다 - 실측으로
20,000 passage 중 19,628 개(98%)가 매칭되었다.

그래서 2단계 판정을 쓴다:
  1) 정규화된 정답 문자열이 passage 에 포함되고,
  2) 질문의 **변별력 있는 내용어**가 passage(title+text)와 min_question_overlap
     개 이상 겹친다.

실측 기준 overlap>=2 가 거짓 양성의 96.8% 를 제거한다. 대가로 "어휘적으로
답할 수 있는 질문" 쪽으로 질문 집합이 치우치며, 이는 보고서의 한계에 명시한다.

Gold labelling needs two stages. DPR's answer-string criterion applies to the
candidates retrieved FOR THAT QUESTION, not to the corpus at large; used
corpus-wide it makes 98% of passages gold for somebody, because 69% of NQ
answers are a single token. Requiring distinctive question terms to co-occur
removes 96.8% of those false positives, at the cost of biasing the retained
question set toward lexically-answerable questions.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.metrics import normalize_answer

# 질문에서 변별력이 없는 토큰. 의문사/기능어를 빼야 "어떤 passage 든 겹치는"
# 현상을 막을 수 있다.
QUESTION_STOPWORDS = frozenset(
    "what when where who whom which whose how why is are was were be been being "
    "the a an of in on at to for from by with and or do does did can could will "
    "would there that this it its as not no any many much first last name named "
    "called year years time new most".split()
)


def content_terms(text: str, stopwords=QUESTION_STOPWORDS, min_length: int = 3) -> set:
    """변별력 있는 내용어 집합. 짧은 토큰과 기능어는 버린다."""
    return {
        token for token in normalize_answer(text).split()
        if token not in stopwords and len(token) >= min_length
    }


def question_id(question: str) -> str:
    """질문 문자열의 안정적 해시 - 시드/실행 간 동일 id 를 보장한다.

    A stable hash of the question text so ids match across runs and seeds.
    """
    return hashlib.sha1(question.strip().lower().encode("utf-8")).hexdigest()[:16]


# datasets>=3 은 bare alias ("nq_open") 를 더 이상 해석하지 못한다
# (HfUriError: Repository id must be ...). 정식 repo id 를 쓴다.
NQ_OPEN_REPO = "google-research-datasets/nq_open"


def load_nq_open(split: str = "train", limit: int = None) -> List[dict]:
    """HuggingFace NQ-open 로딩 -> [{qid, question, answers}].

    87,925 train / 3,610 validation (계획서 Subtask 1.1 과 일치).
    """
    from datasets import load_dataset

    ds = load_dataset(NQ_OPEN_REPO, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        {
            "qid": question_id(row["question"]),
            "question": row["question"],
            "answers": list(row["answer"]),
        }
        for row in ds
    ]


# --- 정답 문자열 인덱스 / answer-string index -------------------------------

class AnswerIndex:
    """정규화된 정답 문자열 -> 질문 id 역인덱스.

    21M passage 를 한 번 훑으면서 "이 passage 가 어떤 질문의 정답을 담고 있나"를
    상수 시간에 판정하기 위한 구조. passage 의 모든 n-gram 을 만드는 대신
    정답의 첫 토큰 집합으로 선행 필터링하여 실제 비교 횟수를 크게 줄인다.

    Inverted index from normalised answer string to question ids. Rather than
    generating every n-gram of every passage, candidate start positions are
    prefiltered by the set of answer first-tokens, which is what makes a single
    pass over 21M passages tractable.
    """

    def __init__(
        self,
        questions: Sequence[dict],
        max_answer_tokens: int = 8,
        min_question_overlap: int = 2,
    ):
        self.max_answer_tokens = max_answer_tokens
        # 0 이면 1단계(정답 문자열 포함)만 쓴다 - 진단·비교용이며 본 실행의
        # 기본값은 2 다. 0 으로 두면 코퍼스의 대부분이 gold 가 된다.
        self.min_question_overlap = min_question_overlap
        self.answer_to_qids: Dict[str, List[str]] = defaultdict(list)
        self.lengths_by_first_token: Dict[str, set] = defaultdict(set)
        self.question_terms: Dict[str, set] = {}
        self.skipped_answers = 0

        for q in questions:
            self.question_terms[q["qid"]] = content_terms(q["question"])
            for answer in q["answers"]:
                norm = normalize_answer(answer)
                if not norm:
                    continue
                tokens = norm.split()
                if len(tokens) > max_answer_tokens:
                    # 지나치게 긴 정답은 문자열 포함 판정이 사실상 실패한다
                    self.skipped_answers += 1
                    continue
                self.answer_to_qids[norm].append(q["qid"])
                self.lengths_by_first_token[tokens[0]].add(len(tokens))

    def __len__(self) -> int:
        return len(self.answer_to_qids)

    def match(self, passage_text: str, title: str = "") -> List[str]:
        """이 passage 의 gold 인 질문 id (2단계 판정 통과분).

        Question ids for which this passage qualifies as gold.
        """
        return self.match_detailed(passage_text, title)[0]

    def match_detailed(self, passage_text: str, title: str = "") -> tuple:
        """(통과한 qid 목록, 1단계만 통과한 수) - 거짓 양성 제거량을 기록하기 위함.

        passage 를 한 번만 정규화해 두 단계가 같은 토큰 목록을 공유한다.
        """
        tokens = normalize_answer(passage_text).split()
        raw: List[str] = []
        seen = set()
        for i, tok in enumerate(tokens):
            lengths = self.lengths_by_first_token.get(tok)
            if not lengths:
                continue
            for n in lengths:
                if i + n > len(tokens):
                    continue
                candidate = " ".join(tokens[i : i + n])
                for qid in self.answer_to_qids.get(candidate, ()):
                    if qid not in seen:
                        seen.add(qid)
                        raw.append(qid)

        if not raw or self.min_question_overlap <= 0:
            return raw, len(raw)

        passage_terms = {
            token for token in tokens
            if token not in QUESTION_STOPWORDS and len(token) >= 3
        }
        if title:
            passage_terms |= content_terms(title)
        kept = [
            qid for qid in raw
            if len(self.question_terms.get(qid, ()) & passage_terms) >= self.min_question_overlap
        ]
        return kept, len(raw)


# --- held-out 평가셋 / fixed held-out evaluation set -------------------------

def build_eval_set(questions: Sequence[dict], size: int = 1000, seed: int = 42) -> List[dict]:
    """모든 run 에서 동일한 고정 평가셋 (Subtask 1.3).

    시드에 의존하지 않도록 qid 해시 순으로 정렬해 뽑는다. 조건 간 평가셋
    분산을 제거하는 것이 목적이므로 run seed 와는 무관해야 한다.

    인자는 **학습에 배분된 질문**이다 (NQ 의 validation split 이 아니다). 여기서
    뽑힌 질문은 split_train_eval 이 학습 예제에서 제외하므로 누출이 없다.
    인자 이름이 전에 `validation` 이어서 출처를 오해하게 만들었다.

    A fixed held-out set identical across every run, drawn from the questions
    allocated to clients and excluded from training by split_train_eval.
    Selection is by qid-hash order, deliberately independent of the run seed.
    """
    ordered = sorted(questions, key=lambda q: hashlib.sha1((str(seed) + q["qid"]).encode()).hexdigest())
    return ordered[:size]


def save_questions(questions: Iterable[dict], path: str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for q in questions:
            fh.write(json.dumps(q) + "\n")
            n += 1
    return n


def load_questions(path: str) -> List[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_usable(
    questions: Sequence[dict], gold_by_qid: Dict[str, List[str]]
) -> Tuple[List[dict], List[dict]]:
    """gold passage 가 하나라도 있는 질문 / 없어서 버리는 질문으로 나눈다."""
    usable, discarded = [], []
    for q in questions:
        golds = gold_by_qid.get(q["qid"], [])
        if golds:
            usable.append({**q, "gold_passage_ids": golds})
        else:
            discarded.append(q)
    return usable, discarded
