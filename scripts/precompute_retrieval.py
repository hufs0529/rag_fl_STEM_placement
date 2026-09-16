#!/usr/bin/env python3
"""Week 2: 후보 리스트와 재채점 점수를 한 번 계산해 캐시 (Subtask 1.2).

이 스크립트가 만드는 것이 계획서의 "frozen data artefact"다. 임베더와
재채점기가 동결이고 코퍼스가 정적이므로, **모든 run 이 이 한 번의 retrieval
pass 를 공유**한다. depth 조건 간 차이가 retrieval 무작위성에서 오는 일이
구조적으로 불가능해진다.

공통 top-50 풀 하나만 저장한다: d=10 의 점수는 d=50 점수의 부분집합이므로
네 조건 모두 같은 파일에서 나온다.

비용 회계: 여기서 절약한 계산은 **할인이 아니다**. 배포 시점의 쿼리당 비용은
Subtask 1.4 의 비용 모델이 그대로 청구하고, 실제 forward pass 수와 wall-clock 을
함께 기록해 둔다.

Usage:
  python scripts/precompute_retrieval.py --seed 1001
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.embedding import Embedder
from src.index import QdrantIndex, QdrantRetriever, collection_name
from src.logging_utils import banner, load_json, save_json
from src.metrics import gold_recall_at_k, headroom_summary, retrieval_bucket
from src.rerank import CrossEncoderReranker
from src.retrieval_cache import RetrievalCache, cache_path
from src.selection import select_passages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    seed = args.seed or cfg.partition.partition_seeds[0]
    pool = cfg.retrieval.candidate_pool

    banner(f"Week 2 / Subtask 1.2 - precomputing the retrieval cache | partition {seed}")

    store = PassageStore.from_jsonl(data_dir / "corpus.jsonl")
    partition = load_json(data_dir / f"partition_{seed}.json")
    embedder = Embedder(cfg.retrieval.embedding_model, cfg.retrieval.query_prefix, args.device)
    reranker = CrossEncoderReranker(
        cfg.rerank.model, cfg.rerank.max_length, cfg.rerank.batch_size, args.device
    )

    index = QdrantIndex(
        path=str(Path(cfg.retrieval.qdrant_path) / f"seed_{seed}"),
        dim=cfg.retrieval.embedding_dim,
        distance=cfg.retrieval.distance,
    )

    summary = {}
    started = time.time()
    try:
        for cid in range(cfg.partition.num_clients):
            questions = partition["client_questions"][str(cid)]
            if not questions:
                continue
            retriever = QdrantRetriever(index, collection_name(cid))

            query_vectors = embedder.encode_queries([q["question"] for q in questions])
            hits = retriever.search(query_vectors, pool)

            rerank_started = time.time()
            scores = reranker.score_batch(
                [q["question"] for q in questions],
                [store.many([pid for pid, _ in h]) for h in hits],
            )
            rerank_seconds = time.time() - rerank_started

            cache = RetrievalCache.build(
                [q["qid"] for q in questions],
                hits,
                scores,
                meta={
                    "client_id": cid,
                    "partition_seed": seed,
                    "candidate_pool": pool,
                    "embedding_model": cfg.retrieval.embedding_model,
                    "rerank_model": cfg.rerank.model,
                    # 배포 시점 비용의 실측 근거 (캐싱은 할인이 아님)
                    "rerank_seconds_measured": rerank_seconds,
                    "rerank_forward_passes": sum(len(h) for h in hits),
                },
            )
            path = cache.save(cache_path(cfg.paths.cache_dir, cid, seed))

            # 이 캐시로 실제 recall 이 depth 에 따라 오르는지 즉시 확인한다.
            # 질문을 세 통으로 분해한다 (gate (ii) 와 같은 기준):
            #   already     - 이미 상위 3개 안에 정답지 -> 모든 조건이 동일
            #   recoverable - 풀 안에 있지만 상위 3개 밖 -> **처리가 일하는 유일한 구간**
            #   out_of_pool - 풀 밖 -> d=50 도 닿지 못함
            # 상한(ceiling)만 보면 "이미 쉬운 질문"과 "건질 수 있는 질문"이 섞인다.
            headroom = headroom_summary([
                retrieval_bucket(cache.get(q["qid"])[0], q["gold_passage_ids"],
                                 cfg.retrieval.top_k)
                for q in questions
            ])
            pool_recall = headroom["ceiling"]

            recalls = {}
            for depth in cfg.treatment.depths:
                hit_count = 0
                for q in questions:
                    candidates, score_map = cache.get(q["qid"])
                    selected = select_passages(candidates, score_map, depth, cfg.retrieval.top_k)
                    # gold id 대조 - 문자열 재매칭은 코퍼스의 "아무 책"까지
                    # 성공으로 세어 네 조건 전부의 recall 을 부풀린다
                    hit_count += gold_recall_at_k(
                        selected, q["gold_passage_ids"], cfg.retrieval.top_k
                    )
                recalls[str(depth)] = hit_count / len(questions)

            best = max(recalls.values())
            summary[str(cid)] = {
                "questions": len(questions),
                "cache": path,
                # 상한: 후보 50개 안에 정답지가 있는 질문의 비율
                "headroom": headroom,
                "gold_recall_at_pool": pool_recall,
                "gold_recall_at_3": recalls,
                # 상한 대비, 그리고 **일할 수 있는 폭 대비** 얼마나 건졌는지
                "headroom_captured": (best / pool_recall) if pool_recall else None,
                "recoverable_captured": (
                    (best - headroom["already_at_top_k"]) / headroom["recoverable"]
                    if headroom["recoverable"] else None
                ),
                "rerank_seconds": rerank_seconds,
            }
            print(f"  client {cid}: {len(questions):,} questions | "
                  + " ".join(f"d{d}:{recalls[str(d)]:.3f}" for d in cfg.treatment.depths)
                  + f" | recoverable:{headroom['recoverable']:.3f}"
                  + f" ceiling:{pool_recall:.3f}"
                  + (f" | captured:{(best - headroom['already_at_top_k']) / headroom['recoverable']:.0%}"
                     if headroom["recoverable"] else ""))
    finally:
        index.close()

    save_json(
        {"partition_seed": seed, "elapsed_seconds": time.time() - started,
         "total_reranker_forward_passes": reranker.forward_passes, "clients": summary},
        Path(cfg.paths.log_dir) / "week2" / f"retrieval_cache_{seed}.json",
    )
    print(f"\n  cached {sum(v['questions'] for v in summary.values()):,} questions "
          f"in {(time.time() - started) / 60:.1f} min")
    print(f"  reranker forward passes: {reranker.forward_passes:,} "
          "(charged at deployment cost in Subtask 1.4, not discounted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
