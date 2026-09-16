#!/usr/bin/env python3
"""Week 2 전체를 순서대로: 코퍼스 -> 파티션 -> 인덱스 -> 캐시.

산출물은 이후 모든 주차가 읽기만 하는 **동결 데이터 산출물(frozen data
artefact)**이다. 본실험 중에는 이 단계로 되돌아가지 않는다.

Usage:
  python scripts/run_week2_pipeline.py
  python scripts/run_week2_pipeline.py --dev
  python scripts/run_week2_pipeline.py --set rerank.batch_size=256     # GPU
  python scripts/run_week2_pipeline.py --start-at build_indexes.py     # 중단 지점부터
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.logging_utils import banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    # 어느 단계에서 멈췄든 거기서 이어서 돌 수 있어야 한다. build_corpus 는
    # 30~50 분(BM25)이고 결정적이므로, 4 단계에서 실패했을 때 처음부터 다시
    # 돌리는 것은 순수한 낭비다.
    parser.add_argument("--start-at", default=None,
                        help="이 스크립트부터 실행 (예: build_indexes.py). "
                             "앞 단계의 산출물이 이미 있을 때 쓴다")
    parser.add_argument("--only", default=None,
                        help="이 스크립트만 실행")
    args = parser.parse_args()
    cfg = config_from_args(args)

    here = Path(__file__).resolve().parent
    shared = ["--config", args.config] + (["--dev"] if args.dev else [])
    for override in args.overrides:
        shared += ["--set", override]

    steps = [("build_corpus.py", [])]
    for seed in cfg.partition.partition_seeds:
        steps += [
            ("partition_clients.py", ["--seed", str(seed)]),
            ("build_indexes.py", ["--seed", str(seed)]),
            ("precompute_retrieval.py", ["--seed", str(seed)]),
        ]

    steps_all = list(steps)
    if args.only:
        steps = [(s_, e) for s_, e in steps if s_ == args.only]
        if not steps:
            print(f"  --only {args.only} matches no step; "
                  f"choose one of {sorted({s_ for s_, _ in steps_all})}")
            return 1
    elif args.start_at:
        names = [s_ for s_, _ in steps]
        if args.start_at not in names:
            print(f"  --start-at {args.start_at} matches no step; "
                  f"choose one of {sorted(set(names))}")
            return 1
        steps = steps[names.index(args.start_at):]
        print(f"  starting at {args.start_at}: "
              f"{len(steps)} of {len(names)} steps "
              "(earlier artefacts are assumed present and are not rebuilt)")

    for script, extra in steps:
        banner(f"running {script} {' '.join(extra)}")
        # -u 를 자식에게도 준다. 파이썬은 stdout 이 파일이면 블록 버퍼링을 하므로,
        # 이게 없으면 `> week2.log` 로 돌릴 때 자식이 끝날 때까지 한 줄도 안 보인다.
        # BM25 단계가 30~50 분이라, 그 동안 멈춘 것과 구별할 수 없었다.
        proc = subprocess.run(
            [sys.executable, "-u", str(here / script)] + shared + extra,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.returncode != 0:
            print(f"\n{script} failed (exit {proc.returncode}); stopping.")
            return proc.returncode

    banner("Week 2 complete - frozen data artefact ready")
    print("  data/corpus.jsonl, data/partition_*.json, data/qdrant_storage/, results/cache/retrieval/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
