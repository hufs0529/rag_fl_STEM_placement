"""학습 예제 조립: 질문 + 선택된 passage -> 토크나이즈된 배치 (Subtask 1.2 / 1.3).

retrieval 은 local step 안에 있지만, 임베더와 재채점기가 모두 동결이고 코퍼스가
정적이므로 후보 리스트와 재채점 점수는 한 번만 계산해 캐시한다(Week 2).
따라서 학습 시점에는 "캐시된 dense 순위 + 캐시된 점수 -> depth d 로 선택"만
수행하면 되고, 모든 run 이 동일한 retrieval pass 를 공유한다.

Assembling training examples. Retrieval sits inside the local step, but because
both retrieval components are frozen and the corpora static, candidate lists and
rerank scores are precomputed once and cached, so all runs share one retrieval
pass and training-time work reduces to selection at depth d.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.prompting import build_training_text, collate_with_padding, tokenize_with_answer_mask
from src.selection import promotion_rate, rank_change_rate, select_passages


class PassageStore:
    """pid -> {pid, title, text}. 200k 규모라 메모리에 그대로 둔다."""

    def __init__(self, passages: Iterable[dict] = ()):
        self._by_id: Dict[str, dict] = {p["pid"]: p for p in passages}

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, pid: str) -> bool:
        return pid in self._by_id

    def get(self, pid: str) -> dict:
        return self._by_id[pid]

    def many(self, pids: Sequence[str]) -> List[dict]:
        return [self._by_id[pid] for pid in pids]

    def ids(self) -> List[str]:
        return list(self._by_id)

    def add(self, passage: dict) -> None:
        self._by_id[passage["pid"]] = passage

    @classmethod
    def from_jsonl(cls, path: str) -> "PassageStore":
        store = cls()
        with Path(path).open() as fh:
            for line in fh:
                if line.strip():
                    store.add(json.loads(line))
        return store

    @classmethod
    def from_jsonl_many(
        cls,
        paths: Sequence[str],
        keep: Optional[Iterable[str]] = None,
    ) -> "PassageStore":
        """여러 파일에서 읽는다. keep 이 주어지면 그 pid 만 메모리에 남긴다.

        게이트 ② 의 프로브 코퍼스는 **두 파일**에서 나온다: 표본 질문과 다른
        질문의 gold 는 gold_passages.jsonl, 채움재는 random_passages.jsonl.
        그래서 그 캐시를 소비하는 쪽이 gold 만 읽으면 채움재 후보에서
        KeyError 가 난다 (실제로 게이트 ③ 에서 발생).

        두 파일을 통째로 올리면 93 만 권이고 딕셔너리 오버헤드까지 3 GB 가 넘는다.
        캐시에 실제로 등장하는 pid 만 남기면 약 11 만 권으로 끝난다 - 7.6 GB 머신에서
        모델까지 함께 올려야 하므로 이 차이가 중요하다.
        """
        wanted = None if keep is None else set(keep)
        store = cls()
        for path in paths:
            file_path = Path(path)
            if not file_path.exists():
                continue
            with file_path.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    passage = json.loads(line)
                    if wanted is None or passage["pid"] in wanted:
                        store.add(passage)
                    if wanted is not None and len(store) == len(wanted):
                        break                   # 필요한 것을 다 찾았으면 더 읽지 않는다
        return store

    def missing(self, pids: Iterable[str]) -> List[str]:
        """store 에 없는 pid 를 돌려준다. 조회 전에 확인해 KeyError 대신 설명을 낸다."""
        return [pid for pid in dict.fromkeys(pids) if pid not in self._by_id]

    def to_jsonl(self, path: str) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for passage in self._by_id.values():
                fh.write(json.dumps(passage) + "\n")
        return len(self._by_id)


def build_example(
    question: dict,
    candidate_ids: Sequence[str],
    rerank_scores: Dict[str, float],
    store: PassageStore,
    depth: int,
    top_k: int = 3,
) -> dict:
    """한 질문 -> depth d 로 고른 3 passage 가 붙은 (prompt, target) 예제.

    진단 지표(선택된 passage id, rank-change, promotion)를 함께 실어 두어
    "처리가 실제로 걸렸는지"를 학습 로그에서 바로 확인할 수 있게 한다.
    """
    selected_ids = select_passages(candidate_ids, rerank_scores, depth, top_k)
    passages = store.many(selected_ids)
    text = build_training_text(question["question"], passages, question["answers"][0])
    return {
        "qid": question["qid"],
        "question": question["question"],
        "answers": question["answers"],
        "selected_ids": selected_ids,
        "prompt": text["prompt"],
        "target": text["target"],
        "rank_change_rate": rank_change_rate(candidate_ids, selected_ids),
        "promotion_rate": promotion_rate(candidate_ids, selected_ids),
    }


class RagDataset:
    """torch Dataset - 토크나이즈는 지연 수행(lazy)해 메모리를 아낀다."""

    def __init__(self, examples: Sequence[dict], tokenizer, max_length: int = 1024):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = tokenize_with_answer_mask(
            self.tokenizer, ex["prompt"], ex["target"], self.max_length
        )
        return {
            "input_ids": enc["input_ids"],
            "labels": enc["labels"],
            "attention_mask": enc["attention_mask"],
        }


def build_dataloader(
    examples: Sequence[dict],
    tokenizer,
    batch_size: int,
    max_length: int = 1024,
    seed: int = 0,
    shuffle: bool = True,
):
    import torch
    from torch.utils.data import DataLoader

    dataset = RagDataset(examples, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        collate_fn=lambda batch: collate_with_padding(batch, tokenizer.pad_token_id),
    )


def selection_diagnostics(examples: Sequence[dict]) -> Dict[str, float]:
    """처리가 걸렸는지 확인하는 집계 (Subtask 1.3 에서 라운드마다 기록)."""
    if not examples:
        return {"rank_change_rate": 0.0, "promotion_rate": 0.0, "n": 0}
    n = len(examples)
    return {
        "rank_change_rate": sum(e["rank_change_rate"] for e in examples) / n,
        "promotion_rate": sum(e["promotion_rate"] for e in examples) / n,
        "n": n,
    }
