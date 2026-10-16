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
    CE_TOKENS, EMBED_TOKENS, cross_encoder_flops_per_pair, describe,
    embedder_flops_per_passage, measure_forward_flops,
)
from src.logging_utils import banner, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pairs", type=int, default=64, help="wall-clock 실측용 쌍 수")
    parser.add_argument("--repeats", type=int, default=5, help="타이밍 반복 횟수")
    args = parser.parse_args()
    cfg = config_from_args(args)

    banner("Week 4 - measuring reranker and embedder FLOPs")

    ce_measured = em_measured = None
    ce_tokens = emb_tokens = None
    throughput = {}

    if not args.analytic_only:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        ce_tok = AutoTokenizer.from_pretrained(cfg.rerank.model)
        ce_model = AutoModelForSequenceClassification.from_pretrained(cfg.rerank.model)
        # **추정식과 같은 길이로 패딩한다.** 그러지 않으면 두 값을 비교할 수 없다:
        # 이 샘플은 패딩 없이 66 토큰이고 추정식은 256 을 가정하므로, 실측이
        # 3.9 배 작게 나와 환율의 분모를 그만큼 낮춘다.
        sample = ce_tok(
            ["what is the capital of france?"], ["Paris is the capital of France." * 8],
            return_tensors="pt", truncation=True,
            padding="max_length", max_length=CE_TOKENS,
        )
        ce_tokens = int(sample["input_ids"].shape[1])
        ce_measured = measure_forward_flops(ce_model, dict(sample))

        # 실제 처리량도 잰다: FLOPs 추정이 어긋나도 "초당 몇 쌍"은 직접 쓸 수 있다.
        #
        # 측정을 **워밍업 + 동기화 + 반복**으로 한다. 원래는 동기화 없이 한 번만
        # 쟀다. CUDA 커널은 비동기로 큐에 들어가므로 ce_model(**batch) 은 계산이
        # 끝나기 전에 돌아오고, 그 한 번에는 모델 전송과 커널 컴파일까지 섞인다.
        # 그렇게 재서 A10G 가 81 쌍/초, CPU 가 84 쌍/초로 나왔다 - GPU 가 CPU 보다
        # 느리다는 값이 나오면 그건 측정이 틀렸다는 뜻이다.
        import torch

        batch = ce_tok(
            ["what is the capital of france?"] * args.pairs,
            ["Paris is the capital of France." * 8] * args.pairs,
            return_tensors="pt", padding="max_length", truncation=True,
            max_length=CE_TOKENS,
        )
        dev = args.device or "cpu"
        ce_model = ce_model.to(dev)
        batch = {k: v.to(dev) for k, v in batch.items()}

        def sync():
            if dev.startswith("cuda"):
                torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(2):                 # 워밍업: 전송과 커널 컴파일을 제외
                ce_model(**batch)
            sync()
            started = time.time()
            for _ in range(args.repeats):
                ce_model(**batch)
            sync()
            elapsed = time.time() - started
        throughput["rerank_pairs_per_second"] = args.pairs * args.repeats / elapsed
        throughput["device"] = dev
        throughput["pairs_per_batch"] = args.pairs
        throughput["repeats"] = args.repeats
        throughput["tokens_per_pair"] = CE_TOKENS
        throughput["method"] = "2 warmup passes, then timed repeats with device sync"

        emb_tok = AutoTokenizer.from_pretrained(cfg.retrieval.embedding_model)
        emb_model = AutoModel.from_pretrained(cfg.retrieval.embedding_model)
        emb_sample = emb_tok(
            ["Paris is the capital of France." * 8],
            return_tensors="pt", truncation=True,
            padding="max_length", max_length=EMBED_TOKENS,
        )
        emb_tokens = int(emb_sample["input_ids"].shape[1])
        em_measured = measure_forward_flops(emb_model, dict(emb_sample))

    report = {
        "rerank": describe(ce_measured, cross_encoder_flops_per_pair(), cfg.rerank.model,
                           n_tokens_measured=ce_tokens, n_tokens_assumed=CE_TOKENS),
        "embed": describe(em_measured, embedder_flops_per_passage(),
                          cfg.retrieval.embedding_model,
                          n_tokens_measured=emb_tokens, n_tokens_assumed=EMBED_TOKENS),
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
        if info.get("measured_estimate") and info.get("analytic_estimate"):
            print(f"          measured {info['measured_estimate']:.3e} at "
                  f"{info['n_tokens_measured']} tokens | "
                  f"analytic {info['analytic_estimate']:.3e} at "
                  f"{info['n_tokens_assumed']} tokens | "
                  f"ratio {info['measured_over_analytic']:.2f}x")
        if info.get("warning"):
            print(f"          WARNING {info['warning']}")
    if throughput:
        print(f"  measured throughput: {throughput['rerank_pairs_per_second']:.1f} pairs/s")
    print(f"\n  wrote {Path(cfg.paths.log_dir) / 'week4' / 'flops.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
