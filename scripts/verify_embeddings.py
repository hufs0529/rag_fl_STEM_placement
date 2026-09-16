#!/usr/bin/env python3
"""corpus_embeddings.npy 가 현재 corpus.jsonl 과 **행 정렬**되어 있는지 확인하고,
맞으면 지문 사이드카를 써서 재임베딩을 건너뛰게 한다.

왜 필요한가: corpus.jsonl 을 재생성했을 때 구성 개수가 같아도 passage **순서**가
같은지는 별개다. 임베딩은 원래 순서에 맞춰져 있고, 순서가 어긋나면 분할·인덱스·
캐시가 전부 무효가 된다 - 그런데 조용히 무효가 된다.

지문 사이드카(.meta.json)가 없는 임베딩은 load_or_encode_at 이 "확인 불가" 로
보고 무조건 다시 만든다. 수정 전 코드로 만든 임베딩이 그렇다. 여기서 정렬을
실제로 검증한 뒤에만 사이드카를 써준다.

검증 방법: 표본 행 몇 개를 다시 임베딩해 저장된 벡터와 비교한다. 같은 모델·같은
본문이면 같은 벡터가 나온다. 순서가 어긋나면 거의 확실히 불일치한다.

Usage:
  python scripts/verify_embeddings.py --device cuda
  python scripts/verify_embeddings.py --device cuda --write-fingerprint
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.embedding import Embedder, passage_text_for_embedding
from src.embedding_cache import corpus_fingerprint, texts_digest
from src.logging_utils import banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples", type=int, default=12,
                        help="다시 임베딩해 비교할 행 수")
    parser.add_argument("--tolerance", type=float, default=2e-3,
                        help="허용 최대 절대 오차 (GPU/CPU 간 부동소수 차이)")
    parser.add_argument("--write-fingerprint", action="store_true",
                        help="검증을 통과하면 .meta.json 을 써서 재임베딩을 건너뛰게 한다")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    vec_path = data_dir / "corpus_embeddings.npy"
    corpus_path = data_dir / "corpus.jsonl"
    banner("verify corpus_embeddings.npy is row-aligned with corpus.jsonl")

    if not vec_path.exists():
        print(f"  {vec_path} 없음 - 검증할 것이 없다 (partition_clients 가 새로 만든다)")
        return 1
    vectors = np.load(vec_path)
    store = PassageStore.from_jsonl(corpus_path)
    pids = store.ids()
    print(f"  embeddings {vectors.shape}  corpus {len(pids):,} passages")

    if vectors.shape[0] != len(pids):
        print(f"  ❌ 행 수 불일치: {vectors.shape[0]:,} vs {len(pids):,}")
        print("     다른 코퍼스의 임베딩이다. partition_clients 가 다시 만들게 두면 된다.")
        return 1

    texts = [passage_text_for_embedding(store.get(p)) for p in pids]

    # 양 끝과 중간을 섞어 뽑는다. 순서가 밀리면 어디서든 걸린다.
    n = len(pids)
    rows = sorted({0, 1, n - 1, n - 2, *(int(i * (n - 1) / max(1, args.samples - 1))
                                         for i in range(args.samples))})
    rows = [r for r in rows if 0 <= r < n][: args.samples + 4]
    print(f"  표본 {len(rows)} 행 재임베딩: {rows[:6]}{' ...' if len(rows) > 6 else ''}")

    embedder = Embedder(cfg.retrieval.embedding_model, cfg.retrieval.query_prefix, args.device)
    fresh = embedder.encode_passages([texts[r] for r in rows], show_progress=False)

    worst = 0.0
    bad = []
    for i, r in enumerate(rows):
        err = float(np.max(np.abs(fresh[i] - vectors[r])))
        worst = max(worst, err)
        if err > args.tolerance:
            bad.append((r, err))
    print(f"  최대 절대 오차 {worst:.2e}  (허용 {args.tolerance:.0e})")

    if bad:
        print(f"  ❌ {len(bad)} 행 불일치: {bad[:5]}")
        print("     corpus.jsonl 의 순서가 임베딩과 다르다. 임베딩·인덱스·캐시를")
        print("     모두 다시 만들어야 한다 (partition_clients -> build_indexes ->")
        print("     precompute_retrieval).")
        return 1

    print("  ✅ 정렬 확인 - 임베딩을 그대로 쓸 수 있다")
    if args.write_fingerprint:
        import json
        meta = vec_path.with_name(vec_path.stem + ".meta.json")
        payload = {
            "model": cfg.retrieval.embedding_model,
            "n": len(pids),
            "digest": corpus_fingerprint(cfg.retrieval.embedding_model, pids, texts_digest(texts)),
        }
        meta.write_text(json.dumps(payload, indent=2))
        print(f"  지문 기록 -> {meta}")
        print("     partition_clients 가 이제 'reusing cached embeddings' 로 건너뛴다")
    else:
        print("  --write-fingerprint 를 주면 재임베딩을 건너뛰게 사이드카를 쓴다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
