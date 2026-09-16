#!/usr/bin/env python3
"""Week 4: reranker/embedder 의 forward FLOPs 실측 (Subtask 1.4 입력).

비용 모델은 reranking FLOPs 를 절약된 바이트와 맞바꾼다. 그 환율의 분모가 되는
"쌍당 FLOPs"를 여기서 확정하고, 실측인지 해석적 추정인지를 함께 기록한다.
가정을 숨기지 않는 것이 이 실험의 결론(운영 구간)의 신뢰도를 좌우한다.

측정 결과는 configs 에 되쓰지 않고 별도 파일로 남긴다 - Week 5 분석이 이 파일을
읽는다.

Usage:
  python scripts/measure_flops.py
  python scripts/measure_flops.py --analytic-only
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.flops import (
    cross_encoder_flops_per_pair, describe, embedder_flops_per_passage,
    measure_forward_flops,
)
from src.logging_utils import banner, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pairs", type=int, default=64, help="wall-clock 실측용 쌍 수")
    args = parser.parse_args()
    cfg = config_from_args(args)

    banner("Week 4 - measuring reranker and embedder FLOPs")

    ce_measured = em_measured = None
    throughput = {}

    if not args.analytic_only:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        ce_tok = AutoTokenizer.from_pretrained(cfg.rerank.model)
        ce_model = AutoModelForSequenceClassification.from_pretrained(cfg.rerank.model)
        sample = ce_tok(
            ["what is the capital of france?"], ["Paris is the capital of France." * 8],
            return_tensors="pt", truncation=True, max_length=cfg.rerank.max_length,
        )
        ce_measured = measure_forward_flops(ce_model, dict(sample))

        # 실제 처리량도 잰다: FLOPs 추정이 어긋나도 "초당 몇 쌍"은 직접 쓸 수 있다.
        import torch

        batch = ce_tok(
            ["what is the capital of france?"] * args.pairs,
            ["Paris is the capital of France." * 8] * args.pairs,
            return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.rerank.max_length,
        )
        with torch.no_grad():
            started = time.time()
            ce_model(**batch)
            elapsed = time.time() - started
        throughput["rerank_pairs_per_second"] = args.pairs / elapsed
        throughput["device"] = args.device

        emb_tok = AutoTokenizer.from_pretrained(cfg.retrieval.embedding_model)
        emb_model = AutoModel.from_pretrained(cfg.retrieval.embedding_model)
        emb_sample = emb_tok(
            ["Paris is the capital of France." * 8],
            return_tensors="pt", truncation=True, max_length=512,
        )
        em_measured = measure_forward_flops(emb_model, dict(emb_sample))

    report = {
        "rerank": describe(ce_measured, cross_encoder_flops_per_pair(), cfg.rerank.model),
        "embed": describe(em_measured, embedder_flops_per_passage(), cfg.retrieval.embedding_model),
        "throughput": throughput,
        "note": (
            "Cached rerank scores are an implementation optimisation; the cost model "
            "charges the deployment-time per-query cost using these figures."
        ),
    }
    save_json(report, Path(cfg.paths.log_dir) / "week4" / "flops.json")

    for key in ("rerank", "embed"):
        info = report[key]
        print(f"  {key:>6}: {info['flops']:.3e} FLOPs ({info['source']}) - {info['label']}")
    if throughput:
        print(f"  measured throughput: {throughput['rerank_pairs_per_second']:.1f} pairs/s")
    print(f"\n  wrote {Path(cfg.paths.log_dir) / 'week4' / 'flops.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
