#!/usr/bin/env python3
"""Week 2: 클라이언트별 Qdrant 컬렉션 구축 (Subtask 1.2).

각 클라이언트는 자기 사설 코퍼스에 대한 자기 인덱스만 갖는다.
임베딩은 partition_clients.py 가 남긴 corpus_embeddings.npy 를 재사용한다.

Usage:
  python scripts/build_indexes.py --seed 1001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.embedding_cache import verify_embeddings_match
from src.index import QdrantIndex, build_client_index
from src.logging_utils import banner, load_json, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    seed = args.seed or cfg.partition.partition_seeds[0]

    banner(f"Week 2 / Subtask 1.2 - per-client Qdrant collections | partition {seed}")

    store = PassageStore.from_jsonl(data_dir / "corpus.jsonl")
    partition = load_json(data_dir / f"partition_{seed}.json")
    # 행 수가 코퍼스와 맞는지 확인한다. 맞지 않으면 다른 코퍼스의 벡터이고,
    # 운 좋으면 IndexError, 나쁘면 조용히 어긋난 인덱스가 만들어진다.
    vectors = verify_embeddings_match(data_dir / "corpus_embeddings.npy", store.ids())
    row_of = {pid: i for i, pid in enumerate(store.ids())}

    index = QdrantIndex(
        path=str(Path(cfg.retrieval.qdrant_path) / f"seed_{seed}"),
        dim=cfg.retrieval.embedding_dim,
        distance=cfg.retrieval.distance,
    )
    built = {}
    try:
        for cid in range(cfg.partition.num_clients):
            pids = partition["client_passages"][str(cid)]
            rows = np.array([row_of[pid] for pid in pids], dtype=np.int64)
            name = build_client_index(index, cid, pids, vectors[rows])
            built[name] = len(pids)
    finally:
        index.close()

    save_json(
        {"partition_seed": seed, "collections": built,
         "qdrant_path": str(Path(cfg.retrieval.qdrant_path) / f"seed_{seed}")},
        Path(cfg.paths.log_dir) / "week2" / f"indexes_{seed}.json",
    )
    print(f"\n  built {len(built)} collections, {sum(built.values()):,} passages total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
