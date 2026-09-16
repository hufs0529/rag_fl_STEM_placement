#!/usr/bin/env python3
"""Week 2: 200k 검색 코퍼스 구성 (Subtask 1.1).

gold passage + BM25 hard negative(정답 문자열 미포함) + 무작위 나머지.

두 가지를 지킨다:
  1) 정답 보유 distractor 를 negative 에서 배제 -> recall 의 false positive 방지
  2) remainder 는 스캔이 따로 모아둔 **정답 미포함 표본**에서만 채움
     -> 코퍼스 전체가 "누군가의 정답"이 되어 난이도가 달라지는 것을 방지

BM25 는 순수 파이썬이라 비용이 (질문 수 x 후보 수) 로 곱해진다. 상한은 설정의
data.hard_negative_questions / data.bm25_pool_size 에 있고, --max-questions 로
더 줄일 수 있다.

Usage:
  python scripts/build_corpus.py
  python scripts/build_corpus.py --dev
  python scripts/build_corpus.py --max-questions 2000      # BM25 질문 수만 줄임
  python scripts/build_corpus.py --question-limit 400      # 코퍼스 질문 수 자체를 줄임
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.corpus_build import build_corpus, verify_no_answer_bearing_negatives
from src.data import PassageStore
from src.logging_utils import banner, save_json
from src.nq_data import load_questions, save_questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--max-questions", type=int, default=None,
                        help="BM25 hard negative 에 쓸 질문 수 상한 "
                             "(기본: data.hard_negative_questions)")
    parser.add_argument("--question-limit", type=int, default=None,
                        help="코퍼스에 넣을 질문 수 상한 (기본: data.question_limit). "
                             "corpus_size 를 줄일 때 이것도 함께 줄여야 한다")
    parser.add_argument("--random-passages", default=None,
                        help="정답 미포함 표본 경로 (기본: data/random_passages.jsonl)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    banner(f"Week 2 / Subtask 1.1 - building the {cfg.data.corpus_size:,}-passage corpus")

    questions = load_questions(data_dir / "questions_usable.jsonl")

    # 질문 수와 corpus_size 는 함께 줄여야 한다. 축소 리허설에서 corpus_size 만
    # 작게 하면 gold 가 예산을 넘어 build_corpus 가 멈춘다(가드는 정상 동작).
    # 표본은 qid 로 정렬해 뽑으므로 시드와 무관하게 결정적이다.
    limit = args.question_limit or cfg.get_path("data.question_limit", None)
    if limit and limit < len(questions):
        questions = sorted(questions, key=lambda q: q["qid"])[:int(limit)]
        print(f"  question_limit={int(limit):,} applied "
              f"(reduced-scale run; the full set has more)")

    store = PassageStore.from_jsonl(data_dir / "gold_passages.jsonl")

    # remainder 전용 표본. 없으면 코퍼스가 전부 answer-bearing 이 되므로
    # 조용히 넘어가지 않고 멈춘다.
    random_path = Path(args.random_passages or data_dir / "random_passages.jsonl")
    if not random_path.exists():
        print(f"  {random_path} not found.\n"
              "  The random remainder has to come from answer-free passages collected by the\n"
              "  scan. Re-run scripts/scan_corpus.py (it fills the reservoir in the same pass);\n"
              "  without it every corpus passage would contain somebody's answer and retrieval\n"
              "  difficulty would not be what the design intended.")
        return 1

    random_store = PassageStore.from_jsonl(random_path)
    for pid in random_store.ids():
        store.add(random_store.get(pid))
    print(f"  {len(questions):,} usable questions | "
          f"{len(store) - len(random_store):,} answer-bearing + "
          f"{len(random_store):,} answer-free passages available")

    result = build_corpus(
        questions,
        store,
        target_size=cfg.data.corpus_size,
        hard_negatives_per_question=cfg.data.hard_negatives_per_question,
        gold_per_question=cfg.data.gold_per_question_in_corpus,
        hard_negative_questions=args.max_questions or cfg.data.hard_negative_questions,
        bm25_pool_size=cfg.data.bm25_pool_size,
        random_pool=random_store.ids(),
        seed=cfg.partition.partition_seeds[0],
    )

    corpus_store = PassageStore(store.many(result["passage_ids"]))
    corpus_store.to_jsonl(data_dir / "corpus.jsonl")

    # 코퍼스에 gold 가 하나도 남지 않은 질문은 여기서 버린다 - 이후 단계의
    # recall 분모를 오염시키지 않기 위해서다.
    kept = []
    for q in questions:
        golds = [pid for pid in q["gold_passage_ids"] if pid in corpus_store]
        if golds:
            kept.append({**q, "gold_passage_ids": golds})
    save_questions(kept, data_dir / "questions_corpus.jsonl")

    violations = verify_no_answer_bearing_negatives(
        kept, set(result["passage_ids"]) - {p for q in kept for p in q["gold_passage_ids"]}, store
    )
    report = {
        "composition": result["composition"],
        "questions_before": len(questions),
        "questions_after": len(kept),
        "answer_bearing_negative_violations": len(violations),
        "random_pool_available": len(random_store),
    }
    save_json(report, Path(cfg.paths.log_dir) / "week2" / "corpus.json")

    comp = result["composition"]
    print(f"\n  corpus: gold {comp['gold']:,} | hard negatives {comp['hard_negatives']:,} "
          f"| random remainder {comp['random_remainder']:,} | total {comp['total']:,}")
    print(f"  questions retained: {len(kept):,} / {len(questions):,}")
    print(f"  answer-bearing negatives: {len(violations)} (must be 0)")
    if comp["shortfall"]:
        print(f"  SHORTFALL: {comp['shortfall']:,} passages - raise data.random_pool_size")
    return 0 if not violations and not comp["shortfall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
