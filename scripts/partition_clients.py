#!/usr/bin/env python3
"""Week 2: 주제 클러스터 기반 8 클라이언트 파티션 (Subtask 1.1).

passage 를 주제로 묶고, 클러스터를 클라이언트에 균등 배분하고, 각 질문을
자기 gold 를 가진 클라이언트에 배정한다. 계획서대로 **파티션 2개**를 만들어
5개 시드에 배정한다 (결과가 한 배정에 묶이지 않도록).

Usage:
  python scripts/partition_clients.py                 # partition_seeds 전부
  python scripts/partition_clients.py --seed 1001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.embedding import Embedder, passage_text_for_embedding
from src.embedding_cache import load_or_encode_at
from src.logging_utils import banner, save_json
from src.nq_data import load_questions
from src.partitioning import partition, partition_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--seed", type=int, default=None, help="단일 파티션만 생성")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    seeds = [args.seed] if args.seed else list(cfg.partition.partition_seeds)

    banner(f"Week 2 / Subtask 1.1 - topic-cluster partition | seeds {seeds}")

    store = PassageStore.from_jsonl(data_dir / "corpus.jsonl")
    questions = load_questions(data_dir / "questions_corpus.jsonl")
    passages = store.many(store.ids())
    print(f"  {len(passages):,} passages | {len(questions):,} questions")

    # 임베딩은 파티션과 인덱싱이 공유한다 - 200k 를 두 번 임베딩하지 않기 위해
    # 여기서 한 번 만들어 디스크에 남긴다 (비용 모델의 일회성 indexing 항).
    vectors_path = data_dir / "corpus_embeddings.npy"
    # 존재 여부만 보면 **다른 코퍼스의 벡터를 재사용**한다. 축소 실행(2,000 권)의
    # 벡터가 남은 상태로 운영(200,000 권)을 돌리면 zip(passages, labels) 가 2,000
    # 으로 조용히 잘려, 20 만 권 중 2 천 권만으로 분할이 만들어진다 - 오류도 없이.
    # 그래서 모델·행 수·본문 해시를 사이드카에 남기고 대조한다.
    texts = [passage_text_for_embedding(p) for p in passages]
    pids = [p["pid"] for p in passages]

    def _encode():
        embedder = Embedder(cfg.retrieval.embedding_model, cfg.retrieval.query_prefix, args.device)
        return embedder.encode_passages(texts)

    vectors = load_or_encode_at(
        vectors_path, cfg.retrieval.embedding_model, pids, texts, _encode,
    )

    for seed in seeds:
        print(f"\n  --- partition seed {seed} ---")
        result = partition(
            passages,
            vectors,
            questions,
            num_clients=cfg.partition.num_clients,
            n_clusters=cfg.partition.clustering.n_clusters,
            batch_size=cfg.partition.clustering.batch_size,
            seed=seed,
            equal_size=cfg.partition.equal_size,
        )
        summary = partition_summary(result)
        save_json(result, data_dir / f"partition_{seed}.json")
        save_json(summary, Path(cfg.paths.log_dir) / "week2" / f"partition_{seed}_summary.json")
        print(f"    size spread {summary['question_size_spread']} "
              f"| gold reachable {summary['gold_reachable_fraction']:.4f}")
        if summary["gold_reachable_fraction"] < 1.0:
            print("    WARNING: some questions cannot reach their own gold evidence; "
                  "measured retrieval failure would reflect partition luck, not ranking difficulty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
