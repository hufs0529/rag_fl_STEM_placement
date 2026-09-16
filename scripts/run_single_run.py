#!/usr/bin/env python3
"""한 run(depth x seed)을 고정 통신 예산으로 끝까지 실행 (Subtask 1.3).

early stopping 은 없다. 모든 조건이 같은 라운드 수 R 을 소모하고, 곡선은
예약된 라운드에서 평가된다. K, S, R 은 gate 3 산출물에서 주입된다.

Usage:
  python scripts/run_single_run.py --depth 10 --seed 1
  python scripts/run_single_run.py --depth 0 --seed 1 --backend inprocess --dev
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.experiment import make_run_spec
from src.fl_runner import run_experiment, run_path
from src.logging_utils import banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backend", choices=["flower", "inprocess"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--calibrated", default=None,
                        help="gate 3 산출물 경로 (기본: results/logs/gates/calibrated_schedule.json)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)

    spec = make_run_spec(cfg, args.depth, args.seed, args.calibrated)
    backend = args.backend or cfg.get_path("federated.backend", "flower")
    path = run_path(cfg, spec)

    if path.exists() and not args.overwrite:
        print(f"{path} already exists; pass --overwrite to redo this run.")
        return 0

    banner(f"run {spec.name} | depth {spec.depth} seed {spec.seed} "
           f"| partition {spec.partition_seed} | K={spec.local_steps_per_round} R={spec.rounds}")
    run_experiment(cfg, spec, backend=backend, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
