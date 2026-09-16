#!/usr/bin/env python3
"""Week 1: psgs_w100 을 훑어 answer-bearing passage 를 찾고 usable question 수를 확정.

계획서 Week 1 활동: "scan psgs_w100 for answer-bearing passages; confirm usable
question count". 21M 전체 스캔은 오래 걸리므로 두 가지 모드를 둔다.

  --sample-fraction 0.05  : 표본 스캔 -> usable question 수 추정 (Week 1 go/no-go)
  --sample-fraction 1.0   : 전체 스캔 -> Week 2 코퍼스 구성용 확정 산출물

산출물:
  data/gold_by_qid.json       질문 -> gold passage id 목록
  data/gold_passages.jsonl    gold passage 본문 (Week 2 코퍼스의 씨앗, gate 입력)
  data/random_passages.jsonl  정답 미포함 passage 의 균일 무작위 표본
                              (Week 2 코퍼스의 "random Wikipedia remainder")
  data/questions_usable.jsonl gold 를 가진 질문만
  results/logs/gates/scan_stats.json

gold 와 random 표본을 **같은 한 번의 통과**에서 뽑는다. 21M 을 두 번 훑지
않기 위해서이고, random 성분을 따로 뽑지 않으면 코퍼스가 전부 "누군가의 정답을
담은 passage"가 되어 검색 난이도가 계획서 의도와 달라진다.

Usage:
  python scripts/scan_corpus.py --sample-fraction 0.05
  python scripts/scan_corpus.py --sample-fraction 1.0 --max-gold-per-question 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.corpus_scan import (
    ReservoirSampler,
    plan_parquet_shards,
    scan_sharded,
    extrapolate_usable,
    save_gold_map,
    save_passages,
    scan_for_gold,
    stream_psgs_w100,
)
from src.logging_utils import banner, save_json
from src.nq_data import AnswerIndex, load_nq_open, save_questions, split_usable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="스캔할 코퍼스 비율. Week 1 추정에는 0.02~0.05 면 충분")
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--max-gold-per-question", type=int, default=20)
    parser.add_argument("--question-limit", type=int, default=None,
                        help="디버그용: 사용할 NQ 질문 수 상한")
    parser.add_argument("--random-pool-size", type=int, default=None,
                        help="reservoir 로 모을 answer-free passage 수 (기본: data.random_pool_size)")
    parser.add_argument("--no-resume", action="store_true",
                        help="체크포인트를 무시하고 처음부터 (기본은 이어서 진행)")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="몇 샤드마다 체크포인트를 남길지 (parquet 소스에서만)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    banner("Week 1 / Subtask 1.1 - scanning psgs_w100 for answer-bearing passages")

    print(f"  passages : {cfg.data.passage_dataset} "
          f"(source={cfg.get_path('data.passage_source', 'parquet')}, "
          f"columns={cfg.get_path('data.passage_columns', '-')})")
    print(f"  questions: {cfg.data.questions_dataset}")
    print("loading NQ-open ...")
    train = load_nq_open("train", limit=args.question_limit)
    validation = load_nq_open("validation")
    print(f"  train {len(train):,} | validation {len(validation):,}")

    # 학습 질문과 평가 질문 모두 gold 가 필요하므로 한 인덱스에 함께 넣는다.
    questions = train + validation
    overlap = cfg.get_path("data.gold_min_question_overlap", 2)
    index = AnswerIndex(questions, min_question_overlap=overlap)
    print(f"  answer index: {len(index):,} distinct normalised answer strings "
          f"({index.skipped_answers:,} answers skipped as too long)")
    print(f"  gold criterion: answer containment + >= {overlap} shared question terms"
          + ("  [WARNING: overlap filter disabled; most of the corpus will be gold]"
             if not overlap else ""))

    # 표본 비율은 샤드 선택으로 적용된다 -> 실제 비율을 먼저 확정해야
    # usable question 외삽이 올바른 분모를 쓴다.
    source = cfg.get_path("data.passage_source", "parquet")
    shard_files, effective_fraction = (None, args.sample_fraction)
    if source == "parquet":
        shard_files, effective_fraction = plan_parquet_shards(
            cfg.data.passage_dataset, args.sample_fraction
        )
        print(f"  shards: reading {len(shard_files)} of them "
              f"(requested fraction {args.sample_fraction}, effective {effective_fraction:.4f})")

    gold_passages_path = data_dir / "gold_passages.jsonl"
    random_passages_path = data_dir / "random_passages.jsonl"
    gold_passages_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pids = set()
    random_pool_size = args.random_pool_size or cfg.data.random_pool_size
    sampler = ReservoirSampler(random_pool_size, seed=cfg.partition.partition_seeds[0])
    print(f"  reservoir for the random remainder: {random_pool_size:,} answer-free passages")
    started = time.time()

    checkpoint_path = data_dir / "scan_checkpoint.json"

    if source == "parquet" and not args.max_passages:
        # 샤드 단위 처리 + 샤드 경계마다 체크포인트. 수 시간짜리 스캔이
        # 네트워크 한 번 끊겼다고 통째로 날아가지 않도록.
        gold_by_qid, stats, random_written = scan_sharded(
            shard_files,
            index,
            gold_path=gold_passages_path,
            random_path=random_passages_path,
            checkpoint_path=checkpoint_path,
            dataset=cfg.data.passage_dataset,
            columns=cfg.get_path("data.passage_columns", ("id", "title", "text")),
            max_gold_per_question=args.max_gold_per_question,
            random_pool_size=random_pool_size,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
            seed=cfg.partition.partition_seeds[0],
        )
    else:
        # --max-passages 로 잘라 쓰는 연습 실행이나 datasets 소스는 재개가
        # 필요 없을 만큼 짧다. 판정 로직은 같은 process_passage 를 쓴다.
        with gold_passages_path.open("w") as sink:
            def on_gold_passage(passage):
                if passage["pid"] not in seen_pids:
                    seen_pids.add(passage["pid"])
                    sink.write(json.dumps(passage) + "\n")

            stream = stream_psgs_w100(
                dataset=cfg.data.passage_dataset,
                config=cfg.data.passage_config,
                sample_fraction=args.sample_fraction,
                max_passages=args.max_passages,
                source=source,
                columns=cfg.get_path("data.passage_columns", ("id", "title", "text")),
                files=shard_files,
            )
            gold_by_qid, stats = scan_for_gold(
                stream,
                index,
                max_gold_per_question=args.max_gold_per_question,
                on_gold_passage=on_gold_passage,
                random_sampler=sampler,
            )
    
    stats.sample_fraction = effective_fraction
    elapsed = time.time() - started

    usable_train, discarded_train = split_usable(train, gold_by_qid)
    usable_val, discarded_val = split_usable(validation, gold_by_qid)
    estimate = extrapolate_usable(stats, len(questions))

    save_gold_map(gold_by_qid, data_dir / "gold_by_qid.json")
    save_questions(usable_train, data_dir / "questions_usable.jsonl")
    # NQ validation split 의 사용 가능 질문. **현재 어떤 코드도 읽지 않는다** -
    # 평가셋은 학습 질문에서 떼어내기 때문이다(configs 의 eval_set_size 주석 참조).
    # 그래도 남겨 두는 이유는 두 가지다: 스캔이 validation split 까지 덮었다는
    # 기록이고, 설계를 바꿔 이쪽으로 평가하려 할 때 19 시간 재스캔을 면한다.
    save_questions(usable_val, data_dir / "questions_usable_val.jsonl")

    report = {
        "scan": stats.to_dict(),
        "elapsed_seconds": elapsed,
        "questions_total": len(questions),
        "usable_train": len(usable_train),
        "discarded_train": len(discarded_train),
        "usable_validation": len(usable_val),
        "gold_passages_written": stats.passages_with_answer,
        "random_passages_written": random_written,
        "random_pool_size_requested": random_pool_size,
        "extrapolation": estimate,
        "expected_usable_in_plan": 60000,
        "gold_min_question_overlap": overlap,
    }
    save_json(report, Path(cfg.paths.log_dir) / "gates" / "scan_stats.json")

    print(f"\nscanned {stats.passages_seen:,} passages in {elapsed/60:.1f} min "
          f"(effective sample fraction {effective_fraction:.4f})")
    print(f"  gold passages kept      : {stats.passages_with_answer:,}")
    print(f"  dropped at the gold cap : {stats.passages_capped_out:,} passages "
          f"(every question they matched already had {args.max_gold_per_question})")
    print(f"  rejected as unrelated   : {stats.passages_answer_string_only:,} passages / "
          f"{stats.rejected_pairs:,} (question, passage) pairs "
          "(answer string present but the question's terms were not)")
    print(f"  usable train questions  : {len(usable_train):,} "
          f"(discarded {len(discarded_train):,})")
    print(f"  usable validation       : {len(usable_val):,}")
    if args.sample_fraction < 1.0:
        print(f"  extrapolated usable @ full scan: ~{estimate['estimated_usable']:,} "
              f"(plan expects ~60k)")
    if checkpoint_path.exists():
        checkpoint_path.unlink()      # 완주했으므로 체크포인트는 필요 없다

    print(f"\nwrote {gold_passages_path} ({stats.passages_with_answer:,} gold passages)")
    print(f"wrote {random_passages_path} ({random_written:,} answer-free passages "
          f"sampled from {stats.answer_free_seen:,} seen)")
    print(f"checkpointing: {'on (resumable)' if source == 'parquet' and not args.max_passages else 'off (short run)'}")
    if random_written < cfg.data.corpus_size // 2:
        print("  WARNING: the random pool is small relative to the corpus budget; "
              "the remainder may not fill. Raise data.random_pool_size or scan more passages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
