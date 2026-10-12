#!/usr/bin/env python3
"""Week 4: 본실험 20 run 실행 (Subtask 1.3).

4 depth x 5 seed, 모든 run 이 **동일한 통신 예산**을 소모하고 early stopping 은
없다. 실행 순서는 seed 바깥 / depth 안쪽이라, 중간에 멈춰도 완료된 시드에
대해서는 네 조건이 모두 갖춰진다.

이미 완료된 run 은 건너뛴다(--overwrite 로 강제 재실행). 한 run 이 실패해도
나머지는 계속 돌리고, 마지막에 실패 목록을 보여준다 - 6주 일정에서 밤새 돌리는
작업이 하나의 예외로 통째로 날아가면 안 되기 때문이다.

Usage:
  python scripts/run_main_runs.py
  python scripts/run_main_runs.py --only-depth 50 --only-seed 1
  python scripts/run_main_runs.py --dry-run
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.fl_runner import run_experiment
from src.logging_utils import banner, save_json
from src.run_matrix import (
    acquire_sweep_lock, check_integrity, enumerate_runs, is_complete, pending,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--backend", choices=["flower", "inprocess"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--calibrated", default=None)
    parser.add_argument("--only-depth", type=int, action="append", default=None)
    parser.add_argument("--only-seed", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)

    specs = enumerate_runs(cfg, args.calibrated)
    if args.only_depth:
        specs = [s for s in specs if s.depth in args.only_depth]
    if args.only_seed:
        specs = [s for s in specs if s.seed in args.only_seed]

    todo = specs if args.overwrite else pending(cfg, specs)
    backend = args.backend or cfg.get_path("federated.backend", "flower")
    budget = specs[0] if specs else None

    banner(f"Week 4 / Subtask 1.3 - {len(specs)} main runs "
           f"({len(todo)} to execute, {len(specs) - len(todo)} already complete)")
    if budget:
        print(f"  fixed budget per run: R={budget.rounds} rounds x "
              f"N={cfg.partition.num_clients} clients, K={budget.local_steps_per_round}, "
              f"no early stopping")
    print("  order: seed outer, depth inner "
          "(an interrupted sweep still has all four conditions per finished seed)")
    for spec in specs:
        state = "done" if is_complete(cfg, spec) else "pending"
        print(f"    {spec.name:>8}  partition {spec.partition_seed}  [{state}]")

    if args.dry_run:
        return 0

    # --dry-run 뒤에 둔다: 상태 확인은 스윕이 돌고 있어도 할 수 있어야 한다.
    lock = acquire_sweep_lock(cfg)

    started = time.time()
    failures = []
    for i, spec in enumerate(todo, 1):
        banner(f"[{i}/{len(todo)}] {spec.name} | depth {spec.depth} seed {spec.seed}")
        try:
            run_experiment(cfg, spec, backend=backend, device=args.device)
        except Exception as exc:                     # 한 run 의 실패로 밤샘 작업을 잃지 않는다
            traceback.print_exc()
            failures.append({"run": spec.name, "error": repr(exc)})
            print(f"  {spec.name} FAILED; continuing with the rest")

    integrity = check_integrity(cfg, specs)
    report = {
        "runs_requested": len(specs),
        "runs_executed": len(todo),
        "failures": failures,
        "elapsed_hours": (time.time() - started) / 3600,
        "integrity": integrity,
        "backend": backend,
    }
    save_json(report, Path(cfg.paths.log_dir) / "week4" / "main_runs.json")

    banner("Week 4 summary")
    print(f"  executed {len(todo)} runs in {report['elapsed_hours']:.2f} h")
    print(f"  complete: {integrity['runs_complete']}/{integrity['runs_expected']}")
    print(f"  identical communication budget across runs: {integrity['identical_budget']}")
    for issue in integrity["issues"]:
        print(f"    ISSUE {issue['field']}: {issue['problem']}")
    if failures:
        print(f"  failed runs: {', '.join(f['run'] for f in failures)}")
    lock.unlink(missing_ok=True)
    return 0 if not failures and integrity["identical_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
